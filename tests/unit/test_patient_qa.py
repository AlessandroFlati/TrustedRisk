"""Unit tests for the three Phase 2.3 patient-agent tools."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.caregiver_handoff import compute_caregiver_handoff
from mcp_server.tools.discharge_qa import compute_discharge_qa
from mcp_server.tools.medication_what_if import compute_medication_what_if


def _run(coro):
    return asyncio.run(coro)


def _decision_card() -> dict:
    return {
        "patient_reference": "Patient/jane",
        "encounter_reference": "Encounter/enc-1",
        "recommendation": {
            "action": "home_with_care",
            "confidence": "preferred",
            "rationale": (
                "Stable on oral diuretics with low readmission risk. "
                "Home-health visits will reinforce medication adherence."
            ),
        },
        "counseling": {
            "medications": [
                {"name": "furosemide 40 mg PO daily"},
                {"name": "lisinopril 10 mg PO daily"},
            ],
            "warning_signs": [
                "Sudden weight gain of 3+ pounds in 1 day",
                "Trouble breathing at rest",
            ],
            "activities": [
                "Walk 10 minutes daily; increase as tolerated",
            ],
            "contact_info": {
                "discharge_line": "555-1234",
                "primary_care": "Dr. Lee 555-5678",
            },
        },
        "followup_plan": [
            "Cardiology in 7-14 days",
            "PCP in 7 days",
        ],
    }


# ─────────────────────── discharge_qa ───────────────────────

def test_discharge_qa_routes_when_can_i_go_home_question():
    out = _run(compute_discharge_qa(
        ["When can I go home?"], decision_card=_decision_card(),
    ))
    assert out.n_items == 1
    item = out.items[0]
    assert item.source_label == "action"
    assert "home" in item.answer.lower()


def test_discharge_qa_routes_followup_question():
    out = _run(compute_discharge_qa(
        ["When is my follow-up appointment?"],
        decision_card=_decision_card(),
    ))
    assert out.items[0].source_label == "followup"
    assert "Cardiology" in out.items[0].answer


def test_discharge_qa_routes_warning_question():
    out = _run(compute_discharge_qa(
        ["When should I call the ER?"], decision_card=_decision_card(),
    ))
    assert out.items[0].source_label == "warnings"
    assert ("911" in out.items[0].answer
                or "ER" in out.items[0].answer
                or "weight gain" in out.items[0].answer.lower())


def test_discharge_qa_routes_meds_question():
    out = _run(compute_discharge_qa(
        ["What medications am I going home on?"],
        decision_card=_decision_card(),
    ))
    assert out.items[0].source_label == "meds"
    assert "furosemide" in out.items[0].answer.lower()


def test_discharge_qa_unknown_question_falls_back_safely():
    out = _run(compute_discharge_qa(
        ["Will my insurance cover this?"],
        decision_card=_decision_card(),
    ))
    assert out.items[0].source_label == "fallback"
    assert "discharge line" in out.items[0].answer.lower()


def test_discharge_qa_carries_multi_turn_context_id():
    out = _run(compute_discharge_qa(
        ["When can I drive again?"],
        decision_card=_decision_card(),
        multi_turn_context_id="ctx-abc-123",
    ))
    assert out.multi_turn_context_id == "ctx-abc-123"


def test_discharge_qa_abstains_on_empty_questions():
    out = _run(compute_discharge_qa([], decision_card=_decision_card()))
    assert out.abstain_recommended is True


def test_discharge_qa_handles_multiple_questions():
    out = _run(compute_discharge_qa(
        [
            "When can I go home?",
            "What medications am I taking?",
            "When is my follow-up?",
        ],
        decision_card=_decision_card(),
    ))
    assert out.n_items == 3


# ─────────────────────── medication_what_if ───────────────────────

def test_what_if_returns_default_scenarios():
    out = _run(compute_medication_what_if(
        medications=["furosemide", "lisinopril"],
    ))
    # 2 meds × 4 default scenarios = 8 items
    assert out.n_items == 8


def test_what_if_warfarin_alcohol_high_severity():
    out = _run(compute_medication_what_if(
        medications=["warfarin 5 mg"],
        scenarios=["take_with_alcohol"],
    ))
    item = out.items[0]
    assert item.safety_severity == "high"
    assert "INR" in item.deterministic_answer or "bleeding" in item.deterministic_answer.lower()


def test_what_if_grapefruit_statin_high():
    out = _run(compute_medication_what_if(
        medications=["atorvastatin 40 mg"],
        scenarios=["take_with_grapefruit"],
    ))
    assert out.items[0].safety_severity == "high"
    assert "statin" in out.items[0].deterministic_answer.lower()


def test_what_if_nsaid_warfarin_high():
    out = _run(compute_medication_what_if(
        medications=["warfarin"],
        scenarios=["take_with_otc_nsaid"],
    ))
    assert out.items[0].safety_severity == "high"
    assert "warfarin" in out.items[0].deterministic_answer.lower()


def test_what_if_stop_abruptly_beta_blocker_high():
    out = _run(compute_medication_what_if(
        medications=["metoprolol 50 mg"],
        scenarios=["stop_abruptly"],
    ))
    assert out.items[0].safety_severity == "high"


def test_what_if_unknown_drug_falls_back_to_defaults():
    """A medication not in the drug-class hint table still gets a
    sensible answer from the default scenario table."""
    out = _run(compute_medication_what_if(
        medications=["zonisamide"],   # not in hint table
        scenarios=["take_with_alcohol"],
    ))
    # default severity for take_with_alcohol is moderate
    assert out.items[0].safety_severity == "moderate"


def test_what_if_abstains_on_empty_medications():
    out = _run(compute_medication_what_if(medications=[]))
    assert out.abstain_recommended is True
    assert out.n_items == 0


def test_what_if_response_carries_references():
    out = _run(compute_medication_what_if(
        medications=["furosemide"], scenarios=["miss_one_dose"],
    ))
    assert out.references


# ─────────────────────── caregiver_handoff ───────────────────────

def test_handoff_summary_mentions_action():
    out = _run(compute_caregiver_handoff(
        decision_card=_decision_card(),
    ))
    assert "home" in out.summary.lower()


def test_handoff_warnings_pulled_from_card():
    out = _run(compute_caregiver_handoff(
        decision_card=_decision_card(),
    ))
    # The card lists "Sudden weight gain..." and "Trouble breathing"
    found_weight = any("weight" in w.lower() for w in out.key_warnings)
    found_breathing = any("breath" in w.lower() for w in out.key_warnings)
    assert found_weight or found_breathing


def test_handoff_contact_info_uses_card_values():
    out = _run(compute_caregiver_handoff(decision_card=_decision_card()))
    assert out.contact_info.get("discharge_line") == "555-1234"


def test_handoff_abstains_without_decision_card():
    out = _run(compute_caregiver_handoff(decision_card=None))
    assert out.abstain_recommended is True


def test_handoff_reading_grade_estimated():
    out = _run(compute_caregiver_handoff(decision_card=_decision_card()))
    # Most caregiver hand-offs land between grade 4 and grade 12 in our
    # rough Flesch-Kincaid implementation. We just assert the estimate
    # is positive and within sanity bounds.
    assert 0.0 < out.reading_level_grade < 30.0


def test_handoff_pulls_patient_name_from_fhir_bundle():
    bundle = {"resourceType": "Bundle", "entry": [
        {"resource": {
            "resourceType": "Patient", "id": "pt-jane",
            "name": [{"given": ["Jane"], "family": "Doe"}],
        }},
    ]}
    out = _run(compute_caregiver_handoff(
        decision_card=_decision_card(), fhir_bundle=bundle,
    ))
    assert "Jane" in out.summary or "Doe" in out.summary
