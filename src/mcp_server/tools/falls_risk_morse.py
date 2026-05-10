"""healthcare.compute_falls_risk_morse -- Morse Falls Scale.

Morse Falls Scale (Morse 2009) is the most-cited inpatient fall risk tool.
6 items, total 0-125. Validated cutoffs (institution-tunable):
  0-24    low risk    -> routine safety
  25-44   moderate    -> standard falls precautions (yellow band, slippers)
  ≥45     high risk   -> high-risk falls protocol (sitter, bed alarm, low bed)

This implementation also flags MEDICATIONS that increase fall risk per the
2023 AGS Beers Criteria (any 1+ from: benzodiazepines, opioids, anticholinergics,
sedating antihistamines, hypnotics, alpha-blockers, sedating antipsychotics,
TCAs, muscle relaxants, sedating SSRIs/SNRIs).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from shared.schemas import MorseFallsReport


# ─────────────────────────────────────────────────────────────────────
# Beers-criteria fall-risk medication patterns
# ─────────────────────────────────────────────────────────────────────

_FALLS_RISK_MED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"diazepam|lorazepam|alprazolam|clonazepam|temazepam|"
                 r"oxazepam|midazolam|triazolam", re.I), "benzodiazepine"),
    (re.compile(r"oxycodone|hydrocodone|morphine|fentanyl|tramadol|"
                 r"hydromorphone|codeine|methadone", re.I), "opioid"),
    (re.compile(r"diphenhydramine|hydroxyzine|chlorpheniramine|"
                 r"meclizine|promethazine", re.I), "sedating_antihistamine"),
    (re.compile(r"zolpidem|zaleplon|eszopiclone|trazodone", re.I), "hypnotic"),
    (re.compile(r"prazosin|terazosin|doxazosin|tamsulosin", re.I),
     "alpha_blocker"),
    (re.compile(r"haloperidol|olanzapine|quetiapine|risperidone|"
                 r"ziprasidone|aripiprazole", re.I), "antipsychotic"),
    (re.compile(r"amitriptyline|nortriptyline|imipramine|doxepin", re.I),
     "tricyclic_antidepressant"),
    (re.compile(r"cyclobenzaprine|baclofen|methocarbamol|tizanidine",
                 re.I), "muscle_relaxant"),
    (re.compile(r"oxybutynin|tolterodine|solifenacin|hyoscyamine|"
                 r"dicyclomine|scopolamine", re.I), "anticholinergic"),
]


def _flag_falls_risk_meds(meds: list[str]) -> list[str]:
    """Return the unique drug-class flags triggered by the medication list."""
    flagged: set[str] = set()
    for med in meds:
        for pat, cls in _FALLS_RISK_MED_PATTERNS:
            if pat.search(med):
                flagged.add(f"{med.strip()} ({cls})")
                break
    return sorted(flagged)


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_falls_risk_morse(
    history_of_falling_3mo: bool | None = None,
    secondary_diagnosis_present: bool | None = None,
    ambulatory_aid: str | None = None,    # none / crutches / cane / walker / furniture / wheelchair
    has_iv_or_heparin_lock: bool | None = None,
    gait: str | None = None,              # normal / weak / impaired
    mental_status: str | None = None,     # oriented / forgets_limitations
    current_medications: list[str] | None = None,
    patient_id: str | None = None,
) -> MorseFallsReport:
    """Compute the Morse Falls Scale.

    Args:
        history_of_falling_3mo: any fall in the past 3 months -> +25.
        secondary_diagnosis_present: more than one active medical diagnosis -> +15.
        ambulatory_aid: "none / bedrest / wheelchair" -> 0; "crutches / cane /
            walker" -> 15; "holds onto furniture" -> 30.
        has_iv_or_heparin_lock: -> +20.
        gait: "normal / bedrest / wheelchair" -> 0; "weak" -> 10; "impaired" -> 20.
        mental_status: "oriented to own ability" -> 0; "forgets limitations /
            overestimates ability" -> 15.
        current_medications: free-text med names -- flagged against the Beers
            falls-risk drug list (informational; doesn't add to the Morse total
            but contributes to the recommended_intervention).

    Returns:
        MorseFallsReport with score_total, risk_tier, recommended_intervention,
        and contributing_medications_flagged.
    """
    # The 6 Morse axes must be assessed by the bedside clinician. A
    # missing value cannot be silently mapped to the benign band -- the
    # absence of an assessment is NOT equivalent to an assessment of
    # "no risk". When any of the 6 is missing, abstain.
    missing = [
        f for f, v in {
            "history_of_falling_3mo": history_of_falling_3mo,
            "secondary_diagnosis_present": secondary_diagnosis_present,
            "ambulatory_aid": ambulatory_aid,
            "has_iv_or_heparin_lock": has_iv_or_heparin_lock,
            "gait": gait,
            "mental_status": mental_status,
        }.items() if v is None
    ]
    if missing:
        return MorseFallsReport(
            patient_id=patient_id,
            history_falls_points=0,
            secondary_diagnosis_points=0,
            ambulatory_aid_points=0,
            iv_or_heparin_lock_points=0,
            gait_points=0,
            mental_status_points=0,
            score_total=0,
            risk_tier="low",
            recommended_intervention="routine_safety",
            contributing_medications_flagged=_flag_falls_risk_meds(
                current_medications or []
            ),
            rationale=(
                "Morse Falls Scale abstained: missing input(s) "
                f"{missing}. The score and tier shown are placeholders, "
                "NOT clinical findings. A missing axis cannot be "
                "mapped to the benign band."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_morse_axes: bedside Morse assessment requires "
                f"explicit values for all 6 axes; missing {missing}."
            ),
        )

    history_pts = 25 if history_of_falling_3mo else 0
    secondary_pts = 15 if secondary_diagnosis_present else 0

    aid = ambulatory_aid.strip().lower()
    if aid in ("none", "bedrest", "wheelchair"):
        aid_pts = 0
    elif aid in ("crutches", "cane", "walker"):
        aid_pts = 15
    elif aid in ("furniture", "holds_furniture"):
        aid_pts = 30
    else:
        aid_pts = 0

    iv_pts = 20 if has_iv_or_heparin_lock else 0

    g = gait.strip().lower()
    if g in ("normal", "bedrest", "wheelchair"):
        gait_pts = 0
    elif g == "weak":
        gait_pts = 10
    elif g == "impaired":
        gait_pts = 20
    else:
        gait_pts = 0

    ms = mental_status.strip().lower()
    if ms in ("oriented", "oriented_to_own_ability"):
        ms_pts = 0
    elif ms in ("forgets_limitations", "overestimates"):
        ms_pts = 15
    else:
        ms_pts = 0

    total = history_pts + secondary_pts + aid_pts + iv_pts + gait_pts + ms_pts

    flagged_meds = _flag_falls_risk_meds(current_medications or [])

    if total >= 45:
        tier = "high"
        intervention = "high_risk_falls_protocol"
    elif total >= 25:
        tier = "moderate"
        intervention = "standard_falls_precautions"
    else:
        tier = "low"
        intervention = "routine_safety"

    # If multiple high-risk meds + already moderate, bump to high
    if tier == "moderate" and len(flagged_meds) >= 3:
        tier = "high"
        intervention = "high_risk_falls_protocol"

    rationale = _build_rationale(total, tier, intervention, history_pts,
                                   secondary_pts, aid_pts, iv_pts, gait_pts,
                                   ms_pts, flagged_meds)

    return MorseFallsReport(
        patient_id=patient_id,
        history_falls_points=history_pts,
        secondary_diagnosis_points=secondary_pts,
        ambulatory_aid_points=aid_pts,
        iv_or_heparin_lock_points=iv_pts,
        gait_points=gait_pts,
        mental_status_points=ms_pts,
        score_total=total,
        risk_tier=tier,  # type: ignore[arg-type]
        recommended_intervention=intervention,  # type: ignore[arg-type]
        contributing_medications_flagged=flagged_meds,
        rationale=rationale,
    )


def _build_rationale(total: int, tier: str, intervention: str,
                       hp: int, sp: int, ap: int, ivp: int, gp: int, msp: int,
                       flagged: list[str]) -> str:
    parts = [
        f"Morse Falls Scale = {total} ({tier} risk -> {intervention}).",
        f"Components: history={hp}, secondary_dx={sp}, ambulatory_aid={ap}, "
        f"IV/heparin_lock={ivp}, gait={gp}, mental_status={msp}.",
    ]
    if flagged:
        parts.append(
            f"Beers-criteria falls-risk medications ({len(flagged)}): "
            + "; ".join(flagged[:5])
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_falls_risk_morse)
