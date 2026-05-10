"""AUDIT-3 -- Right-to-explanation generator (GDPR Art. 22 / EU AI Act).

Composes a plain-language rationale for an automated discharge decision,
satisfying the disclosure obligations under:

  - GDPR Art. 22 (data subjects' right to a meaningful explanation of
    decisions made by automated processing)
  - EU AI Act Art. 13 + Annex IV (transparency requirements for high-
    risk AI systems)

The output is paired with a structured factor table (factor name + weight
+ direction) so a regulator or patient advocate can audit the basis.

Implementation:
  - Factor extraction: pulls `contributing_factors` from the
    DecisionCard's `risk_estimate` + any `abstain` triggers.
  - Plain-language rationale: LLM-driven (Ollama) with a deterministic
    fallback that emits a structured-but-still-readable template.
"""

from __future__ import annotations

import json
import os
from typing import Any

from shared.schemas import (
    DecisionCard,
    RightToExplanationReport,
)


_DEFAULT_MODEL = os.environ.get(
    "TRUSTEDRISK_RTE_LLM_MODEL", "llama3.1:8b")


def _call_ollama(prompt: str) -> str | None:
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


# ─────────────────────── Factor extraction ───────────────────────

def _extract_factors(card: dict[str, Any]
                          ) -> tuple[list[str], dict[str, float], list[str]]:
    factors: list[str] = []
    weights: dict[str, float] = {}
    abstain_triggers: list[str] = []

    reasoning = card.get("reasoning") or {}
    risk = reasoning.get("risk_estimate") or {}
    for f in risk.get("contributing_factors", []) or []:
        if isinstance(f, dict):
            name = str(f.get("name", ""))
            w = float(f.get("weight", 0.0))
            factors.append(name)
            weights[name] = round(w, 4)

    for t in card.get("abstain", []) or []:
        if isinstance(t, dict):
            abstain_triggers.append(str(
                t.get("type") or t.get("trigger") or "unknown"))

    return factors, weights, abstain_triggers


def _data_sources(card: dict[str, Any]) -> list[str]:
    sources: set[str] = set()
    reasoning = card.get("reasoning") or {}
    risk = reasoning.get("risk_estimate") or {}
    if risk.get("model_name"):
        sources.add(f"calibrated risk model: {risk['model_name']}")
    obs_used = risk.get("fhir_observations_used") or []
    if obs_used:
        sources.add(f"{len(obs_used)} FHIR Observation resources")
    validation = card.get("validation") or {}
    grounding = validation.get("grounding") or {}
    if grounding.get("sub_claims"):
        sources.add("clinical guideline corpus (literature grounding)")
    if validation.get("phi_check"):
        sources.add("PHI privacy scrubber")
    return sorted(sources) or ["calibrated risk model"]


# ─────────────────────── Deterministic template ───────────────────────

def _deterministic_rationale(
    card: dict[str, Any],
    factors: list[str],
    weights: dict[str, float],
    abstain_triggers: list[str],
) -> str:
    rec = card.get("recommendation") or {}
    action = rec.get("action") if isinstance(rec, dict) else None
    confidence = rec.get("confidence") if isinstance(rec, dict) else None

    parts: list[str] = []
    if action:
        parts.append(
            f"The system recommended action {action!r} with "
            f"{confidence or 'unspecified'} confidence."
        )
    else:
        parts.append(
            "The system did NOT make a final recommendation; the case "
            "was referred to a clinician."
        )

    if factors:
        ranked = sorted(weights.items(), key=lambda kv: -abs(kv[1]))
        top3 = ", ".join(f"{n} (weight {w:+.2f})" for n, w in ranked[:3])
        parts.append(f"Top factors driving the recommendation: {top3}.")

    if abstain_triggers:
        parts.append(
            f"Abstain triggers fired: {', '.join(abstain_triggers)}. "
            "When triggered, the system recommends clinician review "
            "rather than acting on the model output alone."
        )

    parts.append(
        "You have the right under GDPR Article 22 (and equivalent "
        "regulations) to ask for a human review of this decision and "
        "to contest it if you disagree."
    )

    return " ".join(parts)


# ─────────────────────── LLM enhancement ───────────────────────

_LLM_PROMPT = """Write a plain-language rationale for a hospital discharge \
recommendation produced by an automated decision-support system.

Rules:
  1. Address the patient directly ("you / your"). Reading level: 8th grade.
  2. ≤ 200 words.
  3. List the TOP 3 factors that drove the recommendation, with their \
relative importance (use the supplied weights verbatim).
  4. Explicitly mention any abstain triggers -- these mean the system \
DEFERRED to a human clinician.
  5. Close with the patient's right under GDPR Art. 22 / EU AI Act to \
ask for human review.
  6. Do NOT add new clinical advice; you are explaining a past decision.

Decision summary (JSON):
{summary_json}

Plain-language rationale:"""


# ─────────────────────── Public API ───────────────────────

def compute_right_to_explanation(
    decision_card: DecisionCard | dict[str, Any],
) -> RightToExplanationReport:
    """Compose a GDPR-Art.-22-compliant explanation for a DecisionCard."""
    if isinstance(decision_card, DecisionCard):
        card = decision_card.model_dump(mode="json")
    elif isinstance(decision_card, dict):
        card = decision_card
    else:
        raise ValueError("decision_card must be DecisionCard or dict.")

    factors, weights, abstain = _extract_factors(card)
    sources = _data_sources(card)
    rec = card.get("recommendation") or {}
    action = rec.get("action") if isinstance(rec, dict) else None
    confidence = rec.get("confidence") if isinstance(rec, dict) else None
    audit = card.get("audit") or {}
    request_id = audit.get("request_id")

    summary = {
        "action": action, "confidence": confidence,
        "factors": factors, "weights": weights,
        "abstain_triggers": abstain, "data_sources": sources,
    }
    raw = _call_ollama(_LLM_PROMPT.format(
        summary_json=json.dumps(summary, indent=2, default=str)))
    safety: list[str] = []
    if raw is None or not raw.strip():
        method = "deterministic_template"
        rationale = _deterministic_rationale(card, factors, weights, abstain)
        safety.append("llm_unavailable_template_used")
    else:
        # Defense in depth: LLM must NOT introduce new clinical claims.
        # Block known dose-change / new-medication phrases.
        bad_phrases = ("take more", "increase your dose",
                          "stop taking", "double the dose")
        if any(b in raw.lower() for b in bad_phrases):
            method = "deterministic_template"
            rationale = _deterministic_rationale(card, factors, weights,
                                                       abstain)
            safety.append("llm_dose_change_phrase_blocked")
        else:
            method = "llm"
            rationale = raw[:1500]

    return RightToExplanationReport(
        request_id=request_id,
        decision_action=action,
        decision_confidence=confidence,
        plain_language_rationale=rationale,
        factors_considered=factors,
        factors_weight=weights,
        data_sources=sources,
        method=method,                          # type: ignore[arg-type]
        safety_warnings=safety,
    )
