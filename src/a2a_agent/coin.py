"""COIN -- Conversational Interoperability between A2A agents.

Two parts:

  COIN-3 -- handshake + skill auto-discovery: read a peer agent's
           agent-card.json (via the registry or, in production, via the
           peer's `/.well-known/agent-card.json`) and semantic-match an
           NL prompt to its declared skills.

  COIN-1 -- natural-language dialog: invoke the matched skill in-process
           and translate the structured result back to NL via LLM (or
           a deterministic template when LLM is unavailable).

Together they let one agent ask another in plain English ("when should
this patient come back?") and receive both the structured tool output AND
a NL paraphrase usable for chat / phone / chart.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from shared.schemas import (
    COINDialogResult,
    COINSkillMatch,
)


# ─────────────────────── Skill matching (COIN-3) ───────────────────────

def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) >= 3}


def _keyword_skill_score(prompt: str, skill: dict[str, Any]) -> float:
    """Token-overlap score with id/name boost.

    Two terms (clipped to [0, 1]):
      - id_name_bonus: fraction of prompt tokens that appear in skill.id +
        skill.name (these are the canonical signals -- heavily weighted).
      - corpus_overlap: cosine-style overlap on (id + name + description +
        tags + examples).

    The id/name boost prevents short-description skills from outranking
    long-description skills purely due to denominator effects.
    """
    prompt_tokens = _tokenize(prompt)
    if not prompt_tokens:
        return 0.0

    id_name_text = (skill.get("id", "") + " " + skill.get("name", ""))
    id_name_tokens = _tokenize(id_name_text)
    id_name_overlap = len(prompt_tokens & id_name_tokens)
    id_name_bonus = id_name_overlap / max(1, len(prompt_tokens))

    parts = [skill.get("id", ""), skill.get("name", ""),
                skill.get("description", "")]
    for t in skill.get("tags", []) or []:
        parts.append(str(t))
    for ex in skill.get("examples", []) or []:
        parts.append(str(ex))
    skill_tokens = _tokenize(" ".join(parts))
    if not skill_tokens:
        return min(1.0, id_name_bonus)
    overlap = len(prompt_tokens & skill_tokens)
    cosine = overlap / (
        len(prompt_tokens) ** 0.5 * len(skill_tokens) ** 0.5)

    # Weighted blend -- id/name is the primary signal, corpus overlap is
    # the tiebreaker.
    score = 0.65 * id_name_bonus + 0.35 * cosine
    return min(1.0, score)


def match_skill_to_prompt(
    prompt: str,
    agent_card: dict[str, Any],
    *,
    top_k: int = 3,
) -> list[COINSkillMatch]:
    """Semantic-match an NL prompt against a peer's declared skills.

    Pure-keyword implementation in v1 -- deterministic + tested. The
    embedder-backed version reuses `tool_discovery._load_embedder()` and
    is the natural extension when corpus search is wanted.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        return []
    skills = agent_card.get("skills", []) or []
    matches: list[COINSkillMatch] = []
    for skill in skills:
        if not isinstance(skill, dict):
            continue
        sid = skill.get("id")
        if not isinstance(sid, str) or not sid:
            continue
        score = _keyword_skill_score(prompt, skill)
        if score <= 0.0:
            continue
        matches.append(COINSkillMatch(
            skill_id=sid,
            score=round(score, 4),
            rationale=(
                f"keyword overlap on tokens of {sid!r} "
                f"({skill.get('name', '')[:60]})"
            ),
        ))
    matches.sort(key=lambda m: -m.score)
    return matches[:top_k]


# ─────────────────────── Agent invocation table (COIN-1 dispatch) ───────────────────────

# Maps (agent_id, skill_id) -> callable that takes the structured input dict
# and returns a Pydantic-serializable structured result.
#
# We import lazily so test environments that don't load every partner-agent
# module still work.

async def _invoke_scheduler_followup(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    from apps.scheduler_agent import server as srv

    class _StubReq:
        def __init__(self, payload: dict[str, Any]):
            self._p = payload
            self.headers = {"content-length": "1"}

        async def json(self) -> dict[str, Any]:
            return self._p

    resp = await srv.propose_followup_visits(_StubReq(inputs))
    import json as _json
    return _json.loads(resp.body)


async def _invoke_alert_check_drift(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    from apps.alert_agent import server as srv

    class _StubReq:
        def __init__(self, payload: dict[str, Any]):
            self._p = payload
            self.headers = {"content-length": "1"}

        async def json(self) -> dict[str, Any]:
            return self._p

    resp = await srv.check_drift_now(_StubReq(inputs))
    import json as _json
    return _json.loads(resp.body)


async def _invoke_trustedrisk_safe_discharge(
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Stub -- the trustedrisk-agent's full discharge skill is bundled in
    the showcase scenarios, not exposed as a single MCP tool. We surface
    a minimal in-process risk computation here."""
    from mcp_server.tools.readmission_risk import compute_readmission_risk

    patient_id = inputs.get("patient_id")
    risk = await compute_readmission_risk(patient_id=patient_id)
    return risk.model_dump(mode="json")


_INVOKERS: dict[tuple[str, str], Any] = {
    ("trustedrisk-scheduler-agent", "propose_followup_visits"):
        _invoke_scheduler_followup,
    ("trustedrisk-alert-agent", "check_drift_now"):
        _invoke_alert_check_drift,
    ("trustedrisk-agent", "safe_discharge_review"):
        _invoke_trustedrisk_safe_discharge,
}


# ─────────────────────── NL response generation ───────────────────────

_DEFAULT_MODEL = os.environ.get(
    "TRUSTEDRISK_COIN_LLM_MODEL", "llama3.1:8b")


def _call_ollama(prompt: str) -> str | None:
    if os.environ.get("TRUSTEDRISK_DISABLE_LLM", "0") == "1":
        return None
    try:
        import ollama  # type: ignore
    except ImportError:
        return None
    try:
        resp = ollama.generate(model=_DEFAULT_MODEL, prompt=prompt,
                                  options={"temperature": 0.1})
        return str(resp.get("response", "")).strip()
    except Exception:
        return None


_NL_PROMPT_TEMPLATE = """You are an A2A agent translating a structured tool \
result into natural language for another agent. Rules:

  1. Answer the original question directly.
  2. Quote concrete numbers / dates / actions from the structured result \
verbatim -- do NOT round, paraphrase, or invent.
  3. ≤ 120 words.
  4. NEVER add new clinical advice that isn't already in the structured result.

Original question (from peer agent {source_agent}):
{prompt}

Structured result (from skill {skill_id}):
{result_json}

Plain-language reply:"""


# Deterministic templates per skill -- the safety floor for COIN-1
def _deterministic_nl(skill_id: str, result: dict[str, Any]) -> str:
    if skill_id == "propose_followup_visits":
        n = len(result.get("visits", []) or [])
        tier = result.get("risk_tier") or "unknown"
        return (
            f"I propose {n} follow-up visit(s) at the {tier} priority tier "
            f"for this patient. See the structured plan_id "
            f"{result.get('plan_id', '?')} for the per-visit timing + "
            f"specialty assignment."
        )
    if skill_id == "check_drift_now":
        tier = result.get("severity") or result.get("tier") or "ok"
        if tier == "ok":
            return "Calibration is nominal -- no alerts dispatched."
        kpi = result.get("kpi") or "an unspecified KPI"
        return (
            f"Calibration tier escalated to {tier}. The driver is "
            f"{kpi} (value {result.get('kpi_value', '?')} vs baseline "
            f"{result.get('kpi_baseline', '?')}). "
            f"Subscribers were notified."
        )
    if skill_id == "safe_discharge_review":
        prob = result.get("probability_mean")
        return (
            f"30-day readmission probability: "
            f"{prob:.3f}" if isinstance(prob, (int, float)) else
            "n/a"
        ) + (
            f" (LACE={result.get('lace_raw_score', '?')}). "
            f"Confidence={result.get('confidence', '?')}."
        )
    return f"Result of {skill_id}: " + json.dumps(result, default=str)[:300]


def _llm_nl(skill_id: str, prompt: str, source_agent: str,
              result: dict[str, Any]) -> tuple[str, str]:
    """Return (nl_text, method)."""
    raw = _call_ollama(_NL_PROMPT_TEMPLATE.format(
        skill_id=skill_id,
        source_agent=source_agent,
        prompt=prompt,
        result_json=json.dumps(result, default=str)[:2500],
    ))
    if raw is None or not raw.strip():
        return _deterministic_nl(skill_id, result), "deterministic_template"
    return raw[:1500], "llm"


# ─────────────────────── Public API (COIN-1 + COIN-3 unified) ───────────────────────

async def dialog_with_partner(
    target_agent_id: str,
    prompt: str,
    *,
    structured_inputs: dict[str, Any] | None = None,
    source_agent_id: str = "trustedrisk-agent",
) -> COINDialogResult:
    """End-to-end COIN dialog: discover -> match skill -> invoke -> translate.

    Args:
        target_agent_id: peer agent id from the registry.
        prompt: natural-language request.
        structured_inputs: payload to feed to the matched skill (passed
            through as the structured request body).
        source_agent_id: id of the calling agent (logged in the audit trail).

    Returns:
        COINDialogResult with the matched skill, structured result (when
        available), NL paraphrase, and any safety_warnings.
    """
    from a2a_agent.registry import find_agent

    if not isinstance(prompt, str) or not prompt.strip():
        return COINDialogResult(
            target_agent_id=target_agent_id,
            source_prompt=prompt or "",
            candidate_skills=[],
            matched_skill_id=None,
            structured_result=None,
            nl_response="I didn't see a question. Please send one.",
            method="deterministic_template",
            safety_warnings=["empty_prompt"],
        )

    entry = find_agent(target_agent_id)
    if entry is None:
        return COINDialogResult(
            target_agent_id=target_agent_id,
            source_prompt=prompt,
            candidate_skills=[],
            matched_skill_id=None,
            structured_result=None,
            nl_response=f"Unknown peer agent {target_agent_id!r}.",
            method="deterministic_template",
            safety_warnings=["unknown_target_agent"],
        )

    card = entry.load_agent_card()
    candidates = match_skill_to_prompt(prompt, card, top_k=3)

    if not candidates:
        return COINDialogResult(
            target_agent_id=target_agent_id,
            source_prompt=prompt,
            candidate_skills=[],
            matched_skill_id=None,
            structured_result=None,
            nl_response=(
                f"None of {target_agent_id!r}'s declared skills matched the "
                f"request. Try a more specific phrasing."
            ),
            method="deterministic_template",
            safety_warnings=["no_skill_match"],
        )

    matched_id = candidates[0].skill_id
    invoker = _INVOKERS.get((target_agent_id, matched_id))
    if invoker is None:
        return COINDialogResult(
            target_agent_id=target_agent_id,
            source_prompt=prompt,
            candidate_skills=candidates,
            matched_skill_id=matched_id,
            structured_result=None,
            nl_response=(
                f"Skill {matched_id!r} on {target_agent_id!r} is declared "
                f"in the agent-card but no in-process invoker is registered "
                f"in this build."
            ),
            method="deterministic_template",
            safety_warnings=["invoker_not_registered"],
        )

    inputs = dict(structured_inputs or {})
    try:
        structured = await invoker(inputs)
    except Exception as exc:  # noqa: BLE001
        return COINDialogResult(
            target_agent_id=target_agent_id,
            source_prompt=prompt,
            candidate_skills=candidates,
            matched_skill_id=matched_id,
            structured_result=None,
            nl_response=(
                f"Skill {matched_id!r} on {target_agent_id!r} raised "
                f"{type(exc).__name__}: {str(exc)[:150]}"
            ),
            method="deterministic_template",
            safety_warnings=[f"invoker_error:{type(exc).__name__}"],
        )

    nl_text, method = _llm_nl(matched_id, prompt, source_agent_id, structured)

    return COINDialogResult(
        target_agent_id=target_agent_id,
        source_prompt=prompt,
        candidate_skills=candidates,
        matched_skill_id=matched_id,
        structured_result=structured,
        nl_response=nl_text,
        method=method,                    # type: ignore[arg-type]
        safety_warnings=[],
    )
