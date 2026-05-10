"""GENAI-3 -- LLM-driven critic for the multi-critic ensemble.

Adds a fourth critic ("llm_judge") to the existing rule-based ensemble
(clinical_safety + fairness + evidence). The LLM examines the candidate
DecisionCard for divergent reasoning patterns the rule-based critics miss
-- inconsistencies between the LACE breakdown and the recommendation,
counterfactuals that contradict the action, hidden assumptions in the
rationale, etc.

Deterministic floor: when LLM is unavailable (Ollama not installed,
TRUSTEDRISK_DISABLE_LLM=1, or call timeout), the critic emits a
neutral `approved` verdict tagged as deterministic-fallback. Other
critics' votes still drive the ensemble -- the LLM never blocks on its
own absence.

Output verdicts follow the existing `CritiqueDecision` contract so the
ensemble aggregator and the orchestrator's apply_critique pipeline work
unchanged.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from shared.schemas import (
    AbstainTrigger,
    CritiqueDecision,
    DecisionCard,
)


# ─────────────────────── Card serialization for the prompt ───────────────────────

def _summarize_card_for_prompt(card: DecisionCard) -> dict[str, Any]:
    """Produce a compact JSON-serializable summary that fits in an LLM context.

    Strips all PHI-adjacent fields (we never send raw FHIR data); keeps the
    structured signals an external clinician would actually read.
    """
    summary: dict[str, Any] = {}

    if card.recommendation is not None:
        summary["action"] = card.recommendation.action.value if hasattr(
            card.recommendation.action, "value") else str(
            card.recommendation.action)
        summary["confidence"] = card.recommendation.confidence

    if card.reasoning is not None:
        risk = card.reasoning.risk_estimate
        if risk is not None:
            summary["risk_estimate"] = {
                "outcome_id": risk.outcome_id,
                "probability_mean": risk.probability_mean,
                "probability_ci95": list(risk.probability_ci95),
                "lace_raw_score": risk.lace_raw_score,
                "model_name": risk.model_name,
                "model_version": risk.model_version,
                "contributing_factors": [
                    {"name": f.name,
                     "lace_points": f.lace_points,
                     "weight": f.weight}
                    for f in risk.contributing_factors
                ],
            }
        if card.reasoning.utility_analysis is not None:
            ua = card.reasoning.utility_analysis
            summary["utility"] = {
                "dominant_action": str(ua.dominant_action) if ua.dominant_action
                                       else None,
                "dominance_confidence": ua.dominance_confidence,
            }

    if card.abstain:
        summary["abstain_triggers"] = [
            {"type": getattr(t, "type", "unknown"),
             "detail": (getattr(t, "detail", "") or "")[:120]}
            for t in card.abstain
        ]

    if card.validation:
        if card.validation.grounding is not None:
            summary["grounding_overall"] = \
                card.validation.grounding.overall_verdict
        if card.validation.phi_check is not None:
            summary["phi_risk_level"] = card.validation.phi_check.risk_level

    return summary


# ─────────────────────── LLM call (with deterministic floor) ───────────────────────

_DEFAULT_MODEL = os.environ.get("TRUSTEDRISK_LLM_CRITIC_MODEL", "llama3.1:8b")


_PROMPT_TEMPLATE = """You are a senior hospitalist reviewing an automated discharge \
recommendation for internal consistency and missed risks. The recommendation \
was produced by a calibrated risk model + 3 rule-based critics.

Your job: identify divergent reasoning patterns the rule-based critics may \
have missed. Examples of issues to flag:
  - Recommendation contradicts the LACE risk + CI95 (e.g. discharge_home \
    with risk_estimate.probability_mean > 0.4)
  - Confidence "high" on a recommendation with abstain_triggers present
  - Counterfactual factors that argue against the recommended action
  - Hidden assumptions in the action choice that don't match the data

Return a JSON object with EXACTLY these fields:
  verdict: one of "approved" | "downgrade_confidence" | "force_abstain" | \
"request_replay"
  rationale: ≤ 200-character explanation
  confidence_after: only when verdict=downgrade_confidence -- one of \
"low" | "medium" | "none"

DecisionCard summary (PHI-redacted):
{card_summary_json}

JSON response:"""


def _call_ollama(prompt: str) -> str | None:
    """Invoke Ollama. Returns the raw response text, or None if unavailable."""
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return None
    try:
        import ollama  # type: ignore
    except ImportError:
        return None
    try:
        resp = ollama.generate(model=_DEFAULT_MODEL, prompt=prompt,
                                  options={"temperature": 0.0})
        return str(resp.get("response", "")).strip()
    except Exception:
        return None


_VALID_VERDICTS = {
    "approved", "downgrade_confidence", "force_abstain", "request_replay",
}


def _parse_response(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    if verdict not in _VALID_VERDICTS:
        return None
    return parsed


# ─────────────────────── Public API ───────────────────────

def llm_judge_critic(card: DecisionCard) -> CritiqueDecision:
    """Return the LLM-driven critic's verdict on the candidate DecisionCard.

    Always returns a CritiqueDecision -- never raises. When the LLM is
    unavailable, the critic abstains by emitting a neutral `approved`
    verdict with the rationale flagging the deterministic-fallback path
    so audit logs surface the absence.
    """
    summary = _summarize_card_for_prompt(card)
    prompt = _PROMPT_TEMPLATE.format(
        card_summary_json=json.dumps(summary, indent=2, default=str))
    raw = _call_ollama(prompt)
    parsed = _parse_response(raw)

    if parsed is None:
        # Deterministic fallback: never block the recommendation when the
        # LLM is unavailable -- leave the gating to the rule-based critics.
        return CritiqueDecision(
            verdict="approved",
            rationale=(
                "LLM critic unavailable (model not installed, disabled, or "
                "parse failure) -- defaulting to neutral approval. "
                "Rule-based critics remain in force."
            ),
            critic_role="llm_judge",
        )

    rationale = str(parsed.get("rationale", ""))[:200]
    verdict = parsed["verdict"]

    # Optional fields
    abstain_trigger: AbstainTrigger | None = None
    if verdict == "force_abstain":
        abstain_trigger = AbstainTrigger(
            type="evidence_insufficient",
            detail=("LLM judge: " + (rationale
                                          or "divergent reasoning flagged"))[:300],
        )

    confidence_after = None
    if verdict == "downgrade_confidence":
        ca = parsed.get("confidence_after")
        if ca in ("low", "medium", "none"):
            confidence_after = ca

    return CritiqueDecision(
        verdict=verdict,         # type: ignore[arg-type]
        rationale=rationale or "LLM critic returned no rationale.",
        abstain_trigger_to_add=abstain_trigger,
        confidence_after=confidence_after,
        critic_role="llm_judge",
    )


# ─────────────────────── Phase 6.2 polish-client passthrough ───────────────────────
#
# Phase 2 polish hooks across the tools import `_resolve_llm_model`
# from this module. The real implementation lives in
# `a2a_agent.llm_polish` (Phase 6.2). Re-exported here so legacy import
# paths keep working.

from .llm_polish import _resolve_llm_model  # noqa: E402, F401
