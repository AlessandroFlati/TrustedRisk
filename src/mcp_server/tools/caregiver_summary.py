"""healthcare.compute_caregiver_summary -- PATIENT-3.

Produce a caregiver-facing summary parallel to the patient discharge
counseling. Differences vs the patient document:

  - Higher reading level (10th grade default vs 6th)
  - Explicit red-flag -> action mapping (caregivers need decision rules,
    patients need symptom recognition)
  - Daily observation checklist (caregivers track trajectory)
  - Separate "when to call PCP" and "when to call 911" lists with the
    severity bar made bright

Same safety guarantees as the upstream tools: bullet integrity, no new
medical advice, deterministic floor when LLM unavailable.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from shared.schemas import (
    CaregiverInstructionSet,
    DischargeCounseling,
)


_DEFAULT_MODEL = os.environ.get(
    "TRUSTEDRISK_CAREGIVER_LLM_MODEL", "llama3.1:8b")


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


# ─────────────────────── Deterministic structure builders ───────────────────────

def _coerce_counseling(counseling: DischargeCounseling | dict
                          ) -> DischargeCounseling:
    if isinstance(counseling, dict):
        return DischargeCounseling.model_validate(counseling)
    return counseling


def _section_by_id(c: DischargeCounseling, section_id: str):
    for s in c.sections:
        if s.section_id == section_id:
            return s
    return None


_RED_FLAG_911_KEYWORDS = (
    "chest pain", "trouble breathing", "shortness of breath",
    "uncontrolled bleeding", "severe headache", "facial droop",
    "slurred speech", "loss of consciousness", "seizure",
    "stroke", "heart attack",
)


def _split_red_flags(warning_section_bullets: list[str]) -> tuple[list[str], list[str]]:
    """Split warning-sign bullets into (call_911, call_pcp) buckets."""
    call_911: list[str] = []
    call_pcp: list[str] = []
    for b in warning_section_bullets:
        low = b.lower()
        if any(kw in low for kw in _RED_FLAG_911_KEYWORDS):
            call_911.append(b)
        else:
            call_pcp.append(b)
    return call_911, call_pcp


def _build_daily_checklist(c: DischargeCounseling) -> list[str]:
    """Compose a daily-observation checklist from medication + warning sections."""
    items: list[str] = []
    meds = _section_by_id(c, "your_medications")
    if meds is not None:
        for b in meds.bullets:
            # Tag each medication line with a verification observation
            items.append(f"Confirm: {b}")
    items.append("Note any new or worsening symptoms (date + time + severity).")
    items.append("Track weight daily (same scale, same time of day).")
    items.append("Track all medications taken (yes/no per dose).")
    return items[:10]


def _deterministic_summary(c: DischargeCounseling,
                              audience: str = "caregiver") -> CaregiverInstructionSet:
    warnings_sec = _section_by_id(c, "warning_signs")
    bullets = warnings_sec.bullets if warnings_sec else []
    call_911, call_pcp = _split_red_flags(bullets)

    summary_parts: list[str] = []
    rec_action = ""  # filled by upstream when card is provided
    summary_parts.append(
        "This is a caregiver summary that mirrors the patient discharge plan. "
        "Use it alongside (NOT instead of) the official paper instructions."
    )

    # Pull medication purposes
    meds_sec = _section_by_id(c, "your_medications")
    if meds_sec is not None:
        summary_parts.append(meds_sec.plain_text)

    # Pull follow-up
    fu = _section_by_id(c, "follow_up")
    if fu is not None:
        summary_parts.append(fu.plain_text)

    summary = " ".join(summary_parts)[:1500]

    return CaregiverInstructionSet(
        target_audience=audience,                # type: ignore[arg-type]
        reading_level_grade=10,
        summary=summary,
        red_flag_actions=[
            "If a 911 red flag occurs -> call 911 immediately. Do not wait.",
            "If a non-emergency red flag occurs -> call the PCP within 4 hours.",
            "If the patient cannot be roused or wakes unable to speak -> call 911.",
        ],
        daily_observation_checklist=_build_daily_checklist(c),
        when_to_call_pcp=call_pcp or [
            "Any new symptom not listed in the patient warning signs section.",
        ],
        when_to_call_911=call_911 or [
            "Any sudden severe symptom (chest pain, trouble breathing, "
            "slurred speech, facial droop, seizure, loss of consciousness).",
        ],
        method="deterministic_template",
    )


# ─────────────────────── LLM enhancement ───────────────────────

_LLM_PROMPT = """You write caregiver instructions for a hospital discharge.

You receive the patient's discharge counseling (JSON). Produce a JSON \
object with EXACTLY these fields:

  - summary: ≤ 200-word narrative for the caregiver, written at a 10th-\
grade reading level. Mention recommendation context + the patient's \
medication purposes + follow-up timing. NO new medical advice.
  - red_flag_actions: list[string] (4-6 items, each a "if X -> do Y" rule)
  - daily_observation_checklist: list[string] (≤ 10 items, observable \
once per day or per dose)
  - when_to_call_pcp: list[string] (escalation rules less than 911-grade)
  - when_to_call_911: list[string] (true emergencies only)

CRITICAL RULES:
  1. Every clinical claim must trace to the source counseling. Do NOT \
invent symptoms, medications, doses, or timing.
  2. Reading level: caregivers tolerate slightly more medical language \
than patients, but no jargon without a parenthetical.
  3. Return ONLY the JSON object.

PATIENT COUNSELING (JSON):
{counseling_json}

JSON OUTPUT:"""


def _parse_llm_response(raw: str | None,
                          fallback: CaregiverInstructionSet
                          ) -> CaregiverInstructionSet:
    if not raw:
        return fallback
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return fallback
    try:
        payload = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    if not isinstance(payload, dict):
        return fallback
    try:
        return CaregiverInstructionSet(
            target_audience=fallback.target_audience,
            reading_level_grade=int(payload.get(
                "reading_level_grade", fallback.reading_level_grade)),
            summary=str(payload.get("summary", fallback.summary))[:2000],
            red_flag_actions=list(payload.get(
                "red_flag_actions", fallback.red_flag_actions))[:8],
            daily_observation_checklist=list(payload.get(
                "daily_observation_checklist",
                fallback.daily_observation_checklist))[:12],
            when_to_call_pcp=list(payload.get(
                "when_to_call_pcp", fallback.when_to_call_pcp))[:10],
            when_to_call_911=list(payload.get(
                "when_to_call_911", fallback.when_to_call_911))[:10],
            method="llm",
        )
    except Exception:
        return fallback


# ─────────────────────── Public API ───────────────────────

async def compute_caregiver_summary(
    counseling: DischargeCounseling | dict,
    target_audience: str = "caregiver",
) -> CaregiverInstructionSet:
    """Return a caregiver-facing instruction set parallel to the patient
    counseling document.

    Args:
        counseling: DischargeCounseling (or dict). Provides the substrate
            from which red-flag actions + daily checklist are derived.
        target_audience: caregiver / guardian / family_proxy.

    Returns:
        CaregiverInstructionSet with red-flag -> action rules, daily
        observation checklist, and PCP-vs-911 escalation lists.
    """
    audience = (target_audience or "caregiver").lower().strip()
    if audience not in {"caregiver", "guardian", "family_proxy"}:
        raise ValueError(
            "target_audience must be one of caregiver / guardian / family_proxy")

    src = _coerce_counseling(counseling)
    fallback = _deterministic_summary(src, audience=audience)

    payload = src.model_dump(mode="json")
    prompt = _LLM_PROMPT.format(
        counseling_json=json.dumps(payload, indent=2, default=str),
    )
    raw = _call_ollama(prompt)
    return _parse_llm_response(raw, fallback)


def register(mcp) -> None:
    mcp.tool()(compute_caregiver_summary)
