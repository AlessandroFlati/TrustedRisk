"""Phase 16.J - Patient-advocate specialist tests."""

from __future__ import annotations

import pytest

from a2a_agent.patient_advocate import (
    PatientAdvocateInput, SecondOpinionCard,
    evaluate_patient_advocate,
)


def _input(**overrides) -> PatientAdvocateInput:
    base = dict(
        patient_age=68, patient_race="white",
        patient_insurance="medicare", patient_language="english",
        n_chronic_medications=3,
        has_home_caregiver_available=True,
        has_transportation=True,
        fairness_audit_present=True,
        recommended_action="discharge_home",
        risk_point_estimate=0.10,
    )
    base.update(overrides)
    return PatientAdvocateInput(**base)


# ─────────────────────────────────────────────────────────────────────
# Per-axis verdicts
# ─────────────────────────────────────────────────────────────────────

def test_concur_when_all_axes_clean():
    card = evaluate_patient_advocate(_input())
    assert card.overall_verdict == "concur"
    assert all(f.verdict == "ok" for f in card.findings)


def test_fairness_blocker_on_flagged_subgroup_without_audit():
    card = evaluate_patient_advocate(_input(
        patient_race="black",
        fairness_audit_present=False,
    ))
    fairness = next(
        f for f in card.findings if f.axis == "fairness"
    )
    assert fairness.verdict == "blocker"
    assert card.overall_verdict == "escalate"


def test_fairness_concern_on_flagged_subgroup_with_audit():
    card = evaluate_patient_advocate(_input(
        patient_insurance="medicaid",
        fairness_audit_present=True,
    ))
    fairness = next(
        f for f in card.findings if f.axis == "fairness"
    )
    assert fairness.verdict == "concern"
    assert card.overall_verdict == "challenge"


def test_autonomy_concern_on_low_risk_continued_admission():
    card = evaluate_patient_advocate(_input(
        recommended_action="continued_admission",
        risk_point_estimate=0.10,
    ))
    autonomy = next(
        f for f in card.findings if f.axis == "autonomy"
    )
    assert autonomy.verdict == "concern"


def test_accessibility_blocker_when_no_caregiver():
    card = evaluate_patient_advocate(_input(
        has_home_caregiver_available=False,
    ))
    access = next(
        f for f in card.findings if f.axis == "accessibility"
    )
    assert access.verdict == "blocker"
    assert card.overall_verdict == "escalate"


def test_accessibility_ok_for_inpatient_action():
    card = evaluate_patient_advocate(_input(
        recommended_action="continued_admission",
        risk_point_estimate=0.30,
        has_home_caregiver_available=False,
        has_transportation=False,
    ))
    access = next(
        f for f in card.findings if f.axis == "accessibility"
    )
    assert access.verdict == "ok"


def test_financial_concern_on_uninsured_polypharmacy():
    card = evaluate_patient_advocate(_input(
        patient_insurance="uninsured",
        n_chronic_medications=6,
    ))
    fin = next(
        f for f in card.findings if f.axis == "financial_burden"
    )
    assert fin.verdict == "concern"
    assert card.overall_verdict == "challenge"


def test_language_concern_on_unsupported_preferred_language():
    card = evaluate_patient_advocate(_input(
        patient_language="hindi",
    ))
    lang = next(
        f for f in card.findings if f.axis == "language"
    )
    assert lang.verdict == "concern"


def test_language_ok_for_supported():
    card = evaluate_patient_advocate(_input(
        patient_language="spanish",
    ))
    lang = next(
        f for f in card.findings if f.axis == "language"
    )
    assert lang.verdict == "ok"


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────

def test_concern_in_one_axis_yields_challenge():
    card = evaluate_patient_advocate(_input(
        patient_language="hindi",
    ))
    assert card.overall_verdict == "challenge"


def test_blocker_overrides_concur():
    card = evaluate_patient_advocate(_input(
        has_home_caregiver_available=False,
    ))
    assert card.overall_verdict == "escalate"


# ─────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────

def test_card_serialises_round_trip():
    card = evaluate_patient_advocate(_input())
    payload = card.model_dump(mode="json")
    rebuilt = SecondOpinionCard.model_validate(payload)
    assert rebuilt.overall_verdict == card.overall_verdict


def test_card_has_five_findings():
    card = evaluate_patient_advocate(_input())
    assert len(card.findings) == 5
    axes = {f.axis for f in card.findings}
    assert axes == {
        "fairness", "autonomy", "accessibility",
        "financial_burden", "language",
    }


def test_input_rejects_out_of_unit_risk():
    with pytest.raises(Exception):
        PatientAdvocateInput(
            patient_age=70, recommended_action="discharge_home",
            risk_point_estimate=1.5,
        )


def test_evaluate_is_deterministic():
    inp = _input(patient_race="black", fairness_audit_present=False)
    a = evaluate_patient_advocate(inp)
    b = evaluate_patient_advocate(inp)
    assert a.model_dump() == b.model_dump()
