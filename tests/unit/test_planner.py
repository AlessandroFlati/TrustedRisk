"""Phase 10.8 -- Tool-use planner brain tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.planner import plan_tool_use
from shared.schemas import ToolUsePlanStep


def _run(coro):
    return asyncio.run(coro)


def _tools_in_plan(plan) -> list[str]:
    return [s.tool for s in plan.steps]


def _bundles_in_plan(plan) -> set[str]:
    return {s.bundle for s in plan.steps}


def _specialists_in_plan(plan) -> set[str]:
    return {s.specialist for s in plan.steps}


# ─────────────────────────────────────────────────────────────────────
# Floor -- golden routing per intent
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query, expected_tool, expected_specialist", [
    ("Is this patient safe to discharge home?",
     "compute_readmission_risk", "trustedrisk-discharge"),
    ("Triage this patient at ESI level for the ED.",
     "compute_admission_triage", "trustedrisk-acute"),
    ("Run NEWS2 on the inpatient deterioration vitals.",
     "compute_clinical_deterioration_score", "trustedrisk-acute"),
    ("Score NIHSS for this stroke patient and decide tPA.",
     "compute_stroke_thrombolysis_eligibility", "trustedrisk-acute"),
    ("Compute HEART score for this 58yo with chest pain.",
     "compute_heart_score", "trustedrisk-acute"),
    ("PEWS for a 4-month-old infant with HR 190.",
     "compute_pediatric_early_warning", "trustedrisk-pediatric"),
    ("C-SSRS suicide risk assessment for an adult.",
     "compute_suicide_risk_assessment", "trustedrisk-mental-health"),
    ("Pick empiric antibiotics for sepsis with PCN allergy.",
     "compute_empiric_antibiotic_selection", "trustedrisk-discharge"),
    ("RECIST 1.1 oncology cycle response.",
     "compute_oncology_treatment_response", "trustedrisk-discharge"),
    ("MEOWS for a 32-week pregnant patient.",
     "compute_maternal_early_warning", "trustedrisk-acute"),
    ("Morse falls risk for an 85-year-old on lorazepam.",
     "compute_falls_risk_morse", "trustedrisk-acute"),
    ("ISS trauma severity for polytrauma.",
     "compute_trauma_severity_score", "trustedrisk-acute"),
    ("DKA severity classification with pH 7.18.",
     "compute_dka_severity", "trustedrisk-acute"),
    ("ACR imaging appropriateness for blunt abdominal trauma.",
     "compute_imaging_appropriateness", "trustedrisk-acute"),
    ("AKI KDIGO stage for creatinine 1.0 -> 2.4.",
     "compute_aki_kdigo_stage", "trustedrisk-acute"),
    ("Cost-effectiveness EVOI of running a follow-up troponin.",
     "compute_expected_value_of_intervention", "trustedrisk-population"),
    ("Resolve patient -- the gentleman with chest pain from yesterday.",
     "compute_resolve_patient_from_query", "trustedrisk-evidence"),
    ("Build a differential diagnosis for confusion in an elderly patient.",
     "compute_differential_diagnosis_ranker", "trustedrisk-evidence"),
    ("Translate the discharge counseling to Spanish.",
     "compute_translate_discharge_counseling", "trustedrisk-patient"),
    ("Caregiver hand-off summary for the family.",
     "compute_caregiver_handoff", "trustedrisk-patient"),
    ("Compute Charlson + Elixhauser comorbidity index.",
     "compute_charlson_elixhauser_index", "trustedrisk-discharge"),
    ("Generate the discharge order set bundle.",
     "compute_order_set", "trustedrisk-discharge"),
    ("Recent PubMed papers on apixaban dosing in AKI.",
     "compute_pubmed_search", "trustedrisk-evidence"),
    ("Draft a PA letter for advanced imaging.",
     "compute_pa_letter_draft", "trustedrisk-pa"),
    ("Daily progress note in SOAP format.",
     "compute_progress_note_draft", "trustedrisk-scribe"),
    ("Produce a Joint-Commission discharge summary.",
     "compute_discharge_summary_draft", "trustedrisk-scribe"),
    ("Draft the H&P for tonight's admission.",
     "compute_admission_hnp_draft", "trustedrisk-scribe"),
    ("Suggest ICD-10 codes from the chart.",
     "compute_icd10_suggest", "trustedrisk-coder"),
    ("CYP2C19 PGx dose adjustment for clopidogrel.",
     "compute_pgx_dose_adjustment", "trustedrisk-pgx"),
    ("When should I go to the ER for these symptoms?",
     "compute_when_to_seek_care", "trustedrisk-preadmit"),
    ("HEDIS Stars rating forecast for this MA contract.",
     "compute_quality_measures_aggregate", "trustedrisk-quality"),
    ("Prioritise the care-gap actions for QBP impact.",
     "compute_care_gap_priority_ranking", "trustedrisk-quality"),
    ("Detect syndromic clusters in the ED chief complaints.",
     "compute_syndromic_surveillance", "trustedrisk-pophealth"),
    ("Build vaccine reminder cohorts for the panel.",
     "compute_vaccine_reminder_cohort", "trustedrisk-pophealth"),
    ("Parse this Aetna denial letter.",
     "compute_denial_letter_parse", "trustedrisk-appeals"),
    ("Draft an insurance appeal letter for the denial.",
     "compute_appeal_letter_draft", "trustedrisk-appeals"),
    ("What is the appeal escalation path for this Anthem denial?",
     "compute_appeal_escalation_path", "trustedrisk-appeals"),
])
def test_floor_routes_query_to_expected_tool(
    query: str, expected_tool: str, expected_specialist: str,
):
    plan = _run(plan_tool_use(query))
    tools = _tools_in_plan(plan)
    assert expected_tool in tools, \
        f"Expected {expected_tool!r} in {tools} for query {query!r}"
    specialists = _specialists_in_plan(plan)
    assert expected_specialist in specialists


# ─────────────────────────────────────────────────────────────────────
# Safety floor -- mandatory PHI / fairness injection
# ─────────────────────────────────────────────────────────────────────

def test_phi_floor_injects_detect_phi_for_free_text_phi_input():
    plan = _run(plan_tool_use(
        "Patient John Smith MRN 0001234 needs discharge planning."
    ))
    detect_phi_steps = [s for s in plan.steps if s.tool == "detect_phi"]
    assert len(detect_phi_steps) == 1
    assert detect_phi_steps[0].mandatory is True
    assert plan.safety_floor_engaged is True


def test_fairness_floor_injects_when_demographics_present():
    plan = _run(plan_tool_use(
        "Recommend disposition for a 75-year-old Black female on Medicaid."
    ))
    fa_steps = [s for s in plan.steps if s.tool == "compute_fairness_audit"]
    assert len(fa_steps) == 1
    assert fa_steps[0].mandatory is True
    assert plan.safety_floor_engaged is True


def test_no_safety_floor_for_clean_clinical_query():
    plan = _run(plan_tool_use(
        "Compute HEART score for a typical chest-pain rule-out."
    ))
    assert plan.safety_floor_engaged is False
    assert all(not s.mandatory for s in plan.steps)


# ─────────────────────────────────────────────────────────────────────
# LLM router -- mandatory steps must survive
# ─────────────────────────────────────────────────────────────────────

def test_llm_router_drop_attempt_is_overridden_for_mandatory_steps():
    """An LLM router that tries to drop the mandatory PHI step must NOT
    succeed -- the floor re-injects it."""

    async def _stub_router(query, floor_plan, catalog):
        # Stub returns ZERO steps (worst case for safety)
        return []

    plan = _run(plan_tool_use(
        "Patient John Smith MRN 0001234 needs discharge.",
        llm_router=_stub_router, llm_model_id="stub-v0",
    ))
    tools = _tools_in_plan(plan)
    assert "detect_phi" in tools
    assert plan.llm_router_used is True
    assert plan.llm_model_id == "stub-v0"


def test_llm_router_can_add_steps_freely():
    """Non-mandatory step additions from the LLM router pass through."""

    async def _add_pubmed(query, floor_plan, catalog):
        out = list(floor_plan)
        out.append(ToolUsePlanStep(
            step_id="llm-001",
            specialist="trustedrisk-evidence",
            bundle="external_knowledge",
            tool="compute_pubmed_search",
            args={"query": query},
            rationale="LLM supplemented with literature lookup",
        ))
        return out

    plan = _run(plan_tool_use(
        "Compute HEART score for chest pain.",
        llm_router=_add_pubmed, llm_model_id="stub-v0",
    ))
    assert "compute_pubmed_search" in _tools_in_plan(plan)


def test_llm_router_must_return_list():
    async def _bad(query, floor_plan, catalog):
        return "not a list"      # type: ignore[return-value]
    with pytest.raises(ValueError):
        _run(plan_tool_use("query", llm_router=_bad))


# ─────────────────────────────────────────────────────────────────────
# Plan structure
# ─────────────────────────────────────────────────────────────────────

def test_plan_includes_user_query():
    plan = _run(plan_tool_use("HEDIS Stars projection."))
    assert "HEDIS Stars projection" in plan.user_query


def test_plan_n_steps_equals_steps_length():
    plan = _run(plan_tool_use("Discharge planning + caregiver hand-off."))
    assert plan.n_steps == len(plan.steps)


def test_plan_confidence_in_unit_interval():
    plan = _run(plan_tool_use("Empty intent -- pure noise."))
    assert 0.0 <= plan.confidence <= 1.0


def test_plan_with_no_floor_match_yields_empty_steps():
    plan = _run(plan_tool_use("xxx unintelligible random text yyy"))
    assert plan.n_steps == 0
    assert plan.safety_floor_engaged is False


# ─────────────────────────────────────────────────────────────────────
# Multi-intent -- chained tool use
# ─────────────────────────────────────────────────────────────────────

def test_multi_intent_routes_multiple_steps():
    plan = _run(plan_tool_use(
        "Is this patient safe to discharge, and please also "
        "translate the counseling to Spanish."
    ))
    tools = _tools_in_plan(plan)
    assert "compute_readmission_risk" in tools
    assert "compute_translate_discharge_counseling" in tools


def test_multi_intent_routes_appeals_chain():
    plan = _run(plan_tool_use(
        "Parse this denial letter, draft an appeal letter, "
        "and tell me the escalation path."
    ))
    tools = _tools_in_plan(plan)
    assert "compute_denial_letter_parse" in tools
    assert "compute_appeal_letter_draft" in tools
    assert "compute_appeal_escalation_path" in tools
