"""healthcare.compute_psychiatric_admission_decision -- disposition tool.

Given a structured suicide-risk + agitation + capacity assessment, produces
a disposition recommendation across the involuntary-vs-voluntary spectrum.

The tool models the legal three-prong test for involuntary hold present in
most U.S. states (with state-specific naming -- Colorado M-1, California
5150, NY MHL §9.39, Florida Baker Act):
  1. Danger to self
  2. Danger to others
  3. Grave disability (unable to provide for basic personal needs because
     of mental illness)

ANY ONE prong + an absence of less-restrictive alternatives can support
an involuntary hold. This tool surfaces which prong(s) are met and what
the legal basis description is, but the actual filing remains a clinician
decision -- the tool always sets `abstain_recommended=True` for any
involuntary pathway.
"""

from __future__ import annotations

from typing import Literal

from shared.schemas import PsychiatricAdmission


# ─────────────────────────────────────────────────────────────────────
# State-specific hold descriptions (informational; not legal advice)
# ─────────────────────────────────────────────────────────────────────

_STATE_HOLD_NAMES: dict[str, str] = {
    "CO": "Colorado M-1 hold (72h emergency mental health hold, C.R.S. 27-65-105)",
    "CA": "California 5150 hold (72h, Welf. & Inst. Code §5150)",
    "NY": "NY MHL §9.39 emergency admission (15-day initial)",
    "FL": "Florida Baker Act (F.S. 394.463; up to 72h examination)",
    "IL": "Illinois Mental Health & Developmental Disabilities Code "
          "(405 ILCS 5/3-600 et seq.)",
}


# ─────────────────────────────────────────────────────────────────────
# Tool implementation
# ─────────────────────────────────────────────────────────────────────

async def compute_psychiatric_admission_decision(
    risk_level: str,
    danger_to_self: bool = False,
    danger_to_others: bool = False,
    grave_disability: bool = False,
    voluntary_capable: bool = True,
    has_engaged_outpatient_treatment: bool = False,
    has_safety_plan_in_place: bool = False,
    state_jurisdiction: str | None = None,
    patient_id: str | None = None,
) -> PsychiatricAdmission:
    """Decide psychiatric disposition.

    Args:
        risk_level: one of low / moderate / high / imminent (output of the
            companion suicide-risk tool).
        danger_to_self / danger_to_others / grave_disability: structured
            findings driving the legal hold pathway.
        voluntary_capable: does the patient have decisional capacity for
            voluntary admission?
        has_engaged_outpatient_treatment: protective factor -- patient is
            already in care.
        has_safety_plan_in_place: protective factor -- collaborative safety
            plan exists.
        state_jurisdiction: optional 2-letter state code; if known, populates
            `legal_basis` with the matching hold statute.
    """
    risk = (risk_level or "low").strip().lower()
    if risk not in ("low", "moderate", "high", "imminent"):
        risk = "moderate"  # be conservative if input is unrecognized

    abstain_reasons: list[str] = []

    # Decide disposition
    disposition: Literal[
        "outpatient_followup",
        "crisis_stabilization_unit",
        "voluntary_inpatient",
        "involuntary_hold_evaluation",
        "emergency_court_petition",
    ]
    follow_up_hours: int

    if risk == "imminent" or danger_to_self or danger_to_others:
        # Inpatient admission required
        if voluntary_capable and not (danger_to_others or risk == "imminent"):
            disposition = "voluntary_inpatient"
            follow_up_hours = 24
        else:
            disposition = "involuntary_hold_evaluation"
            follow_up_hours = 72
            abstain_reasons.append(
                "involuntary_pathway: legal filing requires in-person "
                "physician + qualified mental-health professional sign-off."
            )
    elif risk == "high" or grave_disability:
        if grave_disability and not voluntary_capable:
            disposition = "involuntary_hold_evaluation"
            follow_up_hours = 72
            abstain_reasons.append(
                "involuntary_pathway: grave-disability prong + lack of "
                "decisional capacity warrants formal evaluation."
            )
        elif voluntary_capable:
            disposition = "voluntary_inpatient"
            follow_up_hours = 24
        else:
            disposition = "crisis_stabilization_unit"
            follow_up_hours = 24
    elif risk == "moderate":
        if has_engaged_outpatient_treatment and has_safety_plan_in_place:
            disposition = "outpatient_followup"
            follow_up_hours = 48
        else:
            disposition = "crisis_stabilization_unit"
            follow_up_hours = 24
    else:  # low
        disposition = "outpatient_followup"
        follow_up_hours = 168  # 7 days

    legal_basis: str | None = None
    if disposition in ("involuntary_hold_evaluation", "emergency_court_petition"):
        prongs_met: list[str] = []
        if danger_to_self:
            prongs_met.append("danger to self")
        if danger_to_others:
            prongs_met.append("danger to others")
        if grave_disability:
            prongs_met.append("grave disability")
        prong_text = " AND ".join(prongs_met) if prongs_met else "(no prong explicitly recorded -- re-assess)"
        if state_jurisdiction and state_jurisdiction.upper() in _STATE_HOLD_NAMES:
            legal_basis = (
                f"{_STATE_HOLD_NAMES[state_jurisdiction.upper()]} -- prong(s) "
                f"identified: {prong_text}."
            )
        else:
            legal_basis = (
                f"State-specific emergency mental-health hold (statute varies "
                f"by jurisdiction) -- prong(s) identified: {prong_text}."
            )

    safety_plan_required = disposition not in ("outpatient_followup",) or risk != "low"

    rationale = _build_rationale(risk, danger_to_self, danger_to_others,
                                   grave_disability, voluntary_capable,
                                   has_engaged_outpatient_treatment,
                                   has_safety_plan_in_place,
                                   disposition, follow_up_hours, legal_basis)

    return PsychiatricAdmission(
        patient_id=patient_id,
        risk_level=risk,  # type: ignore[arg-type]
        danger_to_self=danger_to_self,
        danger_to_others=danger_to_others,
        grave_disability=grave_disability,
        voluntary_capable=voluntary_capable,
        disposition=disposition,
        rationale=rationale,
        legal_basis=legal_basis,
        safety_plan_required=safety_plan_required,
        follow_up_within_hours=follow_up_hours,
        abstain_recommended=bool(abstain_reasons),
        abstain_reason="; ".join(abstain_reasons) if abstain_reasons else None,
    )


def _build_rationale(risk: str, dts: bool, dto: bool, gd: bool,
                      capable: bool, in_tx: bool, has_plan: bool,
                      disposition: str, hours: int, legal: str | None) -> str:
    parts = [
        f"Risk level = {risk}; danger-to-self={dts}, danger-to-others={dto}, "
        f"grave-disability={gd}; voluntary-capable={capable}.",
        f"Protective: engaged-outpatient={in_tx}, safety-plan={has_plan}.",
        f"Disposition = {disposition}; follow-up within {hours}h.",
    ]
    if legal:
        parts.append(f"Legal basis: {legal}")
    if disposition.startswith("involuntary"):
        parts.append(
            "INVOLUNTARY pathway involves due-process requirements; the "
            "filing decision and timing rest with the responsible "
            "physician + qualified mental-health professional."
        )
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────
# MCP registration
# ─────────────────────────────────────────────────────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_psychiatric_admission_decision)
