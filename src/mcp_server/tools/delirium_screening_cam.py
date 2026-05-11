"""healthcare.compute_delirium_screening_cam -- Confusion Assessment Method.

CAM (Inouye 1990) is the field-standard delirium screening tool, validated
in inpatient + ICU + ED + nursing-home populations. Diagnostic algorithm:

  CAM POSITIVE = Feature 1 + Feature 2 + (Feature 3 OR Feature 4)

where:
  F1 -- Acute onset and fluctuating course
  F2 -- Inattention
  F3 -- Disorganized thinking
  F4 -- Altered level of consciousness

Subtype (Lipowski 1989):
  Hyperactive  -- agitated, hypervigilant, restless
  Hypoactive   -- withdrawn, lethargic, decreased responsiveness
                 (often missed; worse prognosis than hyperactive)
  Mixed        -- features of both, fluctuating

Common contributing factors (DELIRIUM mnemonic):
  Drugs (anticholinergics, BZD, opioids, steroids), Electrolytes (Na, Ca),
  Lack of drugs (withdrawal -- alcohol, BZD), Infection (UTI, pneumonia),
  Reduced sensory input (hearing, vision), Intracranial (CVA, hemorrhage,
  seizure), Urinary retention / fecal impaction, Myocardial / pulmonary.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from shared.schemas import CAMReport

from ._chart_inputs import harden_clinical_inputs


# ─────────────────────────────────────────────────────────────────────
# Drug-class patterns commonly precipitating delirium (DELIRIUM mnemonic)
# ─────────────────────────────────────────────────────────────────────

_DELIRIOGENIC_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"diazepam|lorazepam|alprazolam|clonazepam|temazepam|"
                 r"oxazepam|midazolam|triazolam", re.I), "benzodiazepine"),
    (re.compile(r"oxycodone|hydrocodone|morphine|fentanyl|tramadol|"
                 r"hydromorphone", re.I), "opioid"),
    (re.compile(r"diphenhydramine|hydroxyzine|chlorpheniramine|"
                 r"meclizine|promethazine", re.I), "anticholinergic_antihistamine"),
    (re.compile(r"oxybutynin|tolterodine|solifenacin|hyoscyamine|"
                 r"dicyclomine|scopolamine", re.I), "anticholinergic"),
    (re.compile(r"prednisone|methylprednisolone|dexamethasone", re.I),
     "corticosteroid"),
    (re.compile(r"amitriptyline|nortriptyline|imipramine|doxepin", re.I),
     "tricyclic_antidepressant"),
    (re.compile(r"zolpidem|zaleplon|eszopiclone", re.I), "z_drug"),
    (re.compile(r"haloperidol|chlorpromazine", re.I),
     "typical_antipsychotic"),  # paradoxically can worsen at high dose
]


def _flag_deliriogenic_meds(meds: list[str]) -> list[str]:
    flagged: list[str] = []
    seen: set[str] = set()
    for med in meds:
        for pat, cls in _DELIRIOGENIC_PATTERNS:
            if pat.search(med):
                key = f"{med.strip()} ({cls})"
                if key not in seen:
                    flagged.append(key)
                    seen.add(key)
                break
    return flagged


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_delirium_screening_cam(
    feature1_acute_onset_or_fluctuating: bool,
    feature2_inattention: bool,
    feature3_disorganized_thinking: bool | None = None,
    feature4_altered_consciousness: bool | None = None,
    motor_subtype: str = "unknown",       # hyperactive / hypoactive / mixed / unknown
    current_medications: list[str] | None = None,
    suspected_contributors: list[str] | None = None,
    patient_id: str | None = None,
) -> CAMReport:
    """Apply the CAM diagnostic algorithm.

    Args:
        feature1_acute_onset_or_fluctuating: F1 -- required for CAM positive.
        feature2_inattention: F2 -- required.
        feature3_disorganized_thinking: F3 -- at least one of F3/F4 needed.
        feature4_altered_consciousness: F4 -- at least one of F3/F4 needed.
        motor_subtype: hyperactive / hypoactive / mixed / unknown.
        current_medications: free-text med list -- flagged against DELIRIUM
            mnemonic drug classes.
        suspected_contributors: free-text contributors the clinician already
            identified (UTI, hyponatremia, hypoxia, etc.).

    Returns:
        CAMReport with cam_positive, subtype, contributing_factors,
        next_steps.
    """
    _r, _sb, _ = await harden_clinical_inputs(
        {"current_medications": current_medications},
        chart_derivable={"current_medications"},
    )
    if _sb and _r.get("current_medications"):
        current_medications = _r["current_medications"]
    # CAM requires F3 OR F4. If both are missing the assessment is
    # incomplete -- a missing feature cannot be silently mapped to
    # False, since that would let an unassessed patient fail CAM and
    # be reported as 'cam_negative' without justification.
    if feature3_disorganized_thinking is None and feature4_altered_consciousness is None:
        return CAMReport(
            patient_id=patient_id,
            feature1_acute_onset_or_fluctuating=feature1_acute_onset_or_fluctuating,
            feature2_inattention=feature2_inattention,
            feature3_disorganized_thinking=False,
            feature4_altered_consciousness=False,
            cam_positive=False,
            delirium_subtype="not_applicable",
            contributing_factors=[],
            next_steps=[
                "Complete CAM assessment: feature 3 (disorganized "
                "thinking) and feature 4 (altered consciousness) were "
                "not evaluated. Repeat assessment with a bedside "
                "clinician before any disposition.",
            ],
            rationale=(
                "CAM screening abstained: features F3 and F4 were both "
                "unassessed. CAM positivity requires F1 + F2 + (F3 or "
                "F4); without F3/F4 the result is uninterpretable."
            ),
            abstain_recommended=True,
            abstain_reason=(
                "missing_cam_features: F3 (disorganized thinking) and "
                "F4 (altered consciousness) must be explicitly "
                "evaluated; absence is not equivalent to False."
            ),
        )
    f3 = bool(feature3_disorganized_thinking)
    f4 = bool(feature4_altered_consciousness)

    cam_positive = (
        feature1_acute_onset_or_fluctuating
        and feature2_inattention
        and (f3 or f4)
    )

    sub = motor_subtype.strip().lower()
    if sub not in ("hyperactive", "hypoactive", "mixed"):
        subtype: Literal["hyperactive", "hypoactive", "mixed", "not_applicable"] = (
            "not_applicable" if not cam_positive else "mixed"
        )
    else:
        subtype = sub  # type: ignore[assignment]

    flagged_meds = _flag_deliriogenic_meds(current_medications or [])
    contributors: list[str] = list(suspected_contributors or [])
    if flagged_meds:
        contributors.append(
            f"deliriogenic medications ({len(flagged_meds)}): "
            + "; ".join(flagged_meds[:4])
        )

    next_steps: list[str] = []
    if cam_positive:
        next_steps.append("Initiate non-pharmacologic delirium bundle (HELP / "
                           "ABCDEF: orient, sleep hygiene, mobilize, hydration, "
                           "sensory aids).")
        next_steps.append("Workup for precipitants: UA + culture, CBC + BMP + "
                           "Mg + PO4, oxygen saturation, medication review.")
        if flagged_meds:
            next_steps.append(
                "Deprescribe / hold deliriogenic medications above when feasible."
            )
        if subtype == "hyperactive":
            next_steps.append(
                "Reserve antipsychotics (low-dose haloperidol or quetiapine) ONLY "
                "if patient is a danger to self or staff and non-pharmacologic "
                "measures fail. Avoid in QT-prolongation."
            )
    else:
        next_steps.append(
            "CAM negative -- continue routine cognitive monitoring; if F1+F2 "
            "are present without F3/F4, repeat CAM in 4-8h (delirium is "
            "fluctuating)."
        )

    rationale = _build_rationale(
        feature1_acute_onset_or_fluctuating, feature2_inattention,
        f3, f4,
        cam_positive, subtype, contributors,
    )

    return CAMReport(
        patient_id=patient_id,
        feature1_acute_onset_or_fluctuating=feature1_acute_onset_or_fluctuating,
        feature2_inattention=feature2_inattention,
        feature3_disorganized_thinking=f3,
        feature4_altered_consciousness=f4,
        cam_positive=cam_positive,
        delirium_subtype=subtype,  # type: ignore[arg-type]
        contributing_factors=contributors,
        next_steps=next_steps,
        rationale=rationale,
    )


def _build_rationale(f1: bool, f2: bool, f3: bool, f4: bool,
                       positive: bool, subtype: str,
                       contributors: list[str]) -> str:
    parts = [
        f"CAM features -- F1 (acute onset / fluctuating)={f1}, "
        f"F2 (inattention)={f2}, F3 (disorganized thinking)={f3}, "
        f"F4 (altered consciousness)={f4}.",
        f"CAM = {'POSITIVE' if positive else 'negative'}; subtype={subtype}.",
    ]
    if contributors:
        parts.append("Contributors: " + "; ".join(contributors[:3]))
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_delirium_screening_cam)
