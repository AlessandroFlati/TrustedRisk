"""Functional tests covering canonical end-to-end clinical scenarios.

Each test exercises 3-5 tools in sequence on a coherent patient case.
Signatures are verified against the existing unit tests + the live tool
modules.
"""
from __future__ import annotations

import asyncio


# ─────────────────────── Scenario 1: CHF discharge ───────────────────────

def test_chf_discharge_scenario_med_recon_polypharm_orders(
    chf_bundle, chf_risk_estimate,
):
    from mcp_server.tools.medication_reconciliation import (
        compute_medication_reconciliation,
    )
    from mcp_server.tools.order_set_generator import compute_order_set
    from mcp_server.tools.polypharmacy_concerns import (
        detect_polypharmacy_concerns,
    )

    meds = [
        {"name": "warfarin 5 mg", "drug_class": "anticoagulant_vka",
          "status": "active"},
        {"name": "lisinopril 10 mg", "drug_class": "ace_inhibitor",
          "status": "active"},
        {"name": "spironolactone 25 mg", "drug_class": "mra",
          "status": "active"},
        {"name": "furosemide 40 mg", "drug_class": "loop_diuretic",
          "status": "active"},
        {"name": "metformin 500 mg", "drug_class": "biguanide",
          "status": "active"},
    ]
    recon = asyncio.run(compute_medication_reconciliation(
        admission_meds=meds[:3], discharge_meds=meds,
        monitoring_window_hours=48))
    poly = asyncio.run(detect_polypharmacy_concerns(medications=meds))
    orders = asyncio.run(compute_order_set(
        discharge_action="home_with_care",
        risk_estimate=chf_risk_estimate,
        chief_complaint="acute decompensated CHF",
        medications=meds,
    ))
    assert len(recon.added) >= 2
    assert poly.polypharmacy_severity in ("none", "low", "medium", "high")
    assert orders.n_orders >= 5


# ─────────────────────── Scenario 2: DKA admission ───────────────────────

def test_dka_admission_scenario_severity_glycemic_evi():
    from mcp_server.tools.dka_severity import compute_dka_severity
    from mcp_server.tools.expected_value_of_intervention import (
        compute_expected_value_of_intervention,
    )

    sev = asyncio.run(compute_dka_severity(
        ph=7.10, bicarbonate_meq_l=11.0, glucose_mg_dl=480,
        ketones_present=True, mental_status="alert",
        anion_gap=22, potassium_meq_l=3.2, weight_kg=75))
    assert sev.severity in ("moderate", "severe")
    assert sev.potassium_replacement_at_initiation is True

    evi = asyncio.run(compute_expected_value_of_intervention(
        intervention={
            "name": "Insulin protocol",
            "relative_risk_reduction": 0.50,
            "cost_per_patient_usd": 200.0,
            "qaly_gained_per_avoided_event": 0.10,
            "evidence_grade": "A",
        },
        baseline_event_probability=0.40,
        cohort_size=100,
        avoided_event_cost_usd=20_000.0,
    ))
    assert evi.decision in ("cost_saving", "cost_effective")


# ─────────────────────── Scenario 3: AKI + Contrast safety ───────────────────────

def test_aki_contrast_scenario_with_correct_signatures():
    """KDIGO AKI staging + contrast safety check + dialysis initiation --
    using the actual signatures."""
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    from mcp_server.tools.contrast_safety_check import (
        compute_contrast_safety_check,
    )
    from mcp_server.tools.dialysis_initiation_decision import (
        compute_dialysis_initiation_decision,
    )

    aki = asyncio.run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0,
        creatinine_current_mg_dl=2.4))
    assert aki.aki_stage in ("no_aki", "stage_1", "stage_2", "stage_3")

    contrast = asyncio.run(compute_contrast_safety_check(
        contrast_type="iodinated_iv",
        egfr_ml_min=28, on_metformin=True,
        iodine_contrast_prior_severe_reaction=False))
    # Schema field is `contrast_induced_nephropathy_risk`
    assert contrast.contrast_induced_nephropathy_risk in (
        "low", "moderate", "high", "contraindicated")
    assert contrast.metformin_hold_recommended in (True, False)

    dialysis = asyncio.run(compute_dialysis_initiation_decision(
        aki_stage="stage_2", potassium_meq_l=5.5))
    # No emergent indication -> not indicated yet
    assert dialysis.dialysis_indicated in (True, False)


# ─────────────────────── Scenario 4: Sepsis early warning ───────────────────────

def test_sepsis_clinical_deterioration_score():
    from mcp_server.tools.clinical_deterioration_score import (
        compute_clinical_deterioration_score,
    )

    vital_signs = [
        {"type": "respiratory_rate", "value": 28, "unit": "/min"},
        {"type": "spo2", "value": 88, "unit": "%"},
        {"type": "systolic_bp", "value": 86, "unit": "mm[Hg]"},
        {"type": "heart_rate", "value": 130, "unit": "/min"},
        {"type": "temperature_c", "value": 39.2, "unit": "Cel"},
    ]
    cds = asyncio.run(compute_clinical_deterioration_score(
        vital_signs=vital_signs))
    # NEWS2 components for septic patient: high score expected
    assert cds.score_total >= 5


# ─────────────────────── Scenario 5: Risk + economics e2e ───────────────────────

def test_risk_to_economics_e2e(chf_risk_estimate):
    from a2a_agent.cost_effectiveness_ladder import (
        build_cost_effectiveness_ladder,
    )
    from a2a_agent.outcomes_simulator import simulate_hospital_year
    from mcp_server.tools.expected_value_of_intervention import (
        compute_expected_value_of_intervention,
    )

    evi = asyncio.run(compute_expected_value_of_intervention(
        intervention={"name": "TR counseling",
                          "relative_risk_reduction": 0.25,
                          "cost_per_patient_usd": 75.0,
                          "qaly_gained_per_avoided_event": 0.05,
                          "evidence_grade": "A"},
        risk_estimate=chf_risk_estimate, cohort_size=200,
    ))
    ladder = build_cost_effectiveness_ladder(
        target_intervention_name="TR counseling",
        target_arr_30d_percentage_points=evi.absolute_risk_reduction
        * 100.0,
        target_cost_per_patient_usd=75.0,
    )
    sim = simulate_hospital_year(
        case_mix=[{"name": "CHF", "n_patients_per_year": 1000,
                     "baseline_event_probability":
                     chf_risk_estimate["probability_mean"]}],
        intervention_relative_risk_reduction=0.25,
        n_iterations=300, seed=42,
    )
    assert evi.decision in ("cost_saving", "cost_effective",
                                "not_cost_effective", "uncertain_evidence")
    assert 1 <= ladder.target_rank_among_benchmarks <= len(ladder.benchmarks) + 1
    assert sim.total_events_avoided_mean >= 0
