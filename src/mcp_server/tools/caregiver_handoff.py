"""healthcare.compute_caregiver_handoff -- Phase 2.3 PATIENT-Q3.

Caregiver-language hand-off package built from a DecisionCard +
optional FHIR Bundle. Pure-deterministic 5-section structure: summary,
key warnings, daily tasks, when to call, contact info.

Distinct from `compute_caregiver_summary` (PATIENT-3) -- that tool is
LLM-mediated free-text rendering. This tool is the structured artifact
that a BYO orchestrator can compose into a printable hand-off form
without ANY LLM call.

Reading-level target: ≤ 8th-grade Flesch-Kincaid, achieved by short
sentences + concrete language.
"""

from __future__ import annotations

import os
import re
from typing import Any

from shared.schemas import CaregiverHandoff


_LLM_POLISH_ENV = "TRUSTEDRISK_PATIENT_LLM_POLISH"


def _llm_polish_enabled() -> bool:
    return os.environ.get(_LLM_POLISH_ENV, "0").lower() in (
        "1", "true", "yes", "on",
    )


# ─────────────────────────────────────────────────────────────────────
# Reading-level estimate (Flesch-Kincaid, simplified)
# ─────────────────────────────────────────────────────────────────────

def _count_sentences(text: str) -> int:
    return max(1, len(re.findall(r"[.!?]+", text)))


def _count_words(text: str) -> int:
    return max(1, len(text.split()))


def _count_syllables_word(word: str) -> int:
    word = word.lower()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for c in word:
        is_vowel = c in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_kincaid_grade(text: str) -> float:
    words = _count_words(text)
    sents = _count_sentences(text)
    syll = sum(_count_syllables_word(w) for w in text.split())
    return round(0.39 * (words / sents) + 11.8 * (syll / words) - 15.59, 2)


# ─────────────────────────────────────────────────────────────────────
# Builders
# ─────────────────────────────────────────────────────────────────────

def _build_summary(card: dict, fhir_bundle: dict | None) -> str:
    rec = (card or {}).get("recommendation", {})
    action = rec.get("action")
    label = {
        "discharge_home": "is going home",
        "home_with_care": "is going home with home-health support",
        "snf": "is being moved to a skilled nursing facility",
        "continued_admission": "is staying in the hospital a bit longer",
    }.get(action, f"plan: {action}")

    # Pull patient name when available
    name = "the patient"
    if isinstance(fhir_bundle, dict):
        for entry in fhir_bundle.get("entry", []):
            r = entry.get("resource", {})
            if r.get("resourceType") == "Patient":
                given = (r.get("name") or [{}])[0].get("given", [])
                family = (r.get("name") or [{}])[0].get("family", "")
                name = " ".join(given + ([family] if family else [])) or name
                break

    rationale = rec.get("rationale") or rec.get("reasoning") or ""
    summary = f"{name} {label}."
    if rationale:
        # Trim long rationales to one sentence for caregiver clarity
        first_sent = rationale.split(".")[0].strip()
        if first_sent:
            summary += f" {first_sent}."
    return summary


def _build_warnings(card: dict) -> list[str]:
    out = (
        (card.get("counseling") or {}).get("warning_signs")
        or card.get("warning_signs")
    )
    if out:
        return list(out)
    return [
        "Trouble breathing or chest pain",
        "Sudden weakness or trouble speaking",
        "Severe bleeding that won't stop",
        "Confusion or sudden change in behavior",
    ]


def _build_daily_tasks(card: dict) -> list[str]:
    out = []
    meds = (
        (card.get("counseling") or {}).get("medications")
        or card.get("medications")
        or []
    )
    if meds:
        names = [
            (m.get("name") if isinstance(m, dict) else str(m))
            for m in meds[:5]
        ]
        out.append(
            f"Give medications on time: {', '.join(names)}"
        )
    activities = (
        (card.get("counseling") or {}).get("activities")
        or card.get("activities")
    )
    if activities:
        out.append(f"Help with daily activity: {activities[0]}")
    out.append("Watch food and drink -- follow the discharge plan.")
    out.append("Track symptoms in a notebook or phone app.")
    return out


def _build_when_to_call(card: dict) -> list[str]:
    return [
        "Call 911 for chest pain, severe trouble breathing, or sudden "
        "weakness on one side.",
        "Call the discharge line for new fever ≥ 38.5 C, vomiting that "
        "won't stop, or a missed dose you don't know how to handle.",
        "Call the primary-care office for routine follow-up questions "
        "or to refill a prescription.",
    ]


def _build_contact_info(card: dict) -> dict[str, str]:
    """Return the contact_info dict from the DecisionCard.

    Always includes the universal emergency number. Other entries
    (discharge_line, primary_care) are only included when the
    DecisionCard supplies them -- no placeholder values are fabricated.
    """
    info = (
        (card.get("counseling") or {}).get("contact_info")
        or card.get("contact_info")
        or {}
    )
    result: dict[str, str] = {"emergency": "911"}
    if isinstance(info, dict):
        for key, val in info.items():
            if val and str(val).strip():
                result[key] = str(val)
    return result


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

async def compute_caregiver_handoff(
    decision_card: dict | None = None,
    fhir_bundle: dict | None = None,
    patient_reference: str | None = None,
) -> CaregiverHandoff:
    """Build a structured caregiver hand-off from a DecisionCard.

    Args:
        decision_card: Output of an upstream discharge tool chain.
        fhir_bundle: Optional FHIR R4 Bundle (for patient name).
        patient_reference: Optional reference override.

    Returns:
        CaregiverHandoff with the 5 fixed sections + reading-level
        estimate.
    """
    pref = (
        patient_reference
        or (decision_card or {}).get("patient_reference")
        or "unknown"
    )

    if not decision_card:
        return CaregiverHandoff(
            patient_reference=pref,
            summary="",
            key_warnings=[],
            daily_tasks=[],
            when_to_call=[],
            contact_info={},
            reading_level_grade=0.0,
            abstain_recommended=True,
            abstain_reason=(
                "No DecisionCard provided. compute_caregiver_handoff "
                "needs the upstream discharge artefact to anchor every "
                "section."
            ),
            references=[
                "AHRQ Project RED caregiver-education templates.",
                "Joint Commission caregiver hand-off standards.",
            ],
        )

    # noqa: ABSTAIN-GUARD -- the hand-off summary is anchored on the
    # DecisionCard.recommendation.action. Without it we cannot tell the
    # caregiver whether the patient is going home, to a SNF, etc.
    action = (decision_card.get("recommendation") or {}).get("action")
    if not action:
        return CaregiverHandoff(
            patient_reference=pref,
            summary="",
            key_warnings=[],
            daily_tasks=[],
            when_to_call=[],
            contact_info={},
            reading_level_grade=0.0,
            abstain_recommended=True,
            abstain_reason=(
                "missing_decision_card_action: caregiver_handoff requires "
                "DecisionCard.recommendation.action to describe the "
                "discharge plan. Provide a DecisionCard with an action "
                "field set to one of: discharge_home, home_with_care, "
                "snf, continued_admission."
            ),
            references=[
                "AHRQ Project RED caregiver-education templates.",
                "Joint Commission caregiver hand-off standards.",
            ],
        )

    summary = _build_summary(decision_card, fhir_bundle)
    warnings = _build_warnings(decision_card)
    tasks = _build_daily_tasks(decision_card)
    when_call = _build_when_to_call(decision_card)
    contact = _build_contact_info(decision_card)

    reading_grade = _flesch_kincaid_grade(summary)

    contains_polish = False
    polish_model: str | None = None
    if _llm_polish_enabled():
        try:
            from a2a_agent.llm_critic import _resolve_llm_model
            model = _resolve_llm_model()
        except Exception:
            model = None
        if model is not None:
            contains_polish = True
            polish_model = getattr(model, "model_name", "unknown")

    return CaregiverHandoff(
        patient_reference=pref,
        summary=summary,
        key_warnings=warnings,
        daily_tasks=tasks,
        when_to_call=when_call,
        contact_info=contact,
        reading_level_grade=max(0.0, reading_grade),
        contains_llm_polish=contains_polish,
        llm_model_id=polish_model,
        references=[
            "AHRQ Project RED caregiver-education templates.",
            "Joint Commission caregiver hand-off standards.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_caregiver_handoff)
