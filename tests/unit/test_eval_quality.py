"""Unit tests for EVAL-1/2/3/4 -- quality benchmarks + comparative-effectiveness."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from a2a_agent.eval_quality import (
    compute_comparative_effectiveness,
    compute_hedis_score,
    compute_hospital_compare_benchmark,
    compute_schwartz_quality_score,
)


# ───────────────────────────────────────────────────────────────────
# EVAL-1 -- HEDIS measure scorer
# ───────────────────────────────────────────────────────────────────

def _bundle(*, age: int, sex: str = "female",
              icd10_codes: list[str] | None = None,
              observations: list[dict] | None = None,
              procedures: list[dict] | None = None,
              med_requests: list[dict] | None = None) -> dict:
    today = datetime.now(timezone.utc).date()
    dob = today.replace(year=today.year - age)
    bundle: dict = {"resourceType": "Bundle", "type": "collection",
                       "entry": [{"resource": {
                           "resourceType": "Patient", "id": "pt-1",
                           "gender": sex,
                           "birthDate": dob.isoformat()}}]}
    for c in icd10_codes or []:
        bundle["entry"].append({"resource": {
            "resourceType": "Condition",
            "code": {"coding": [{
                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                "code": c}]}}})
    for o in observations or []:
        bundle["entry"].append({"resource": o})
    for p in procedures or []:
        bundle["entry"].append({"resource": p})
    for m in med_requests or []:
        bundle["entry"].append({"resource": m})
    return bundle


def test_hedis_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        compute_hedis_score("nope")  # type: ignore[arg-type]


def test_hedis_diabetic_with_a1c_in_range_compliant():
    obs = [{
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org",
                                "code": "4548-4"}]},
        "valueQuantity": {"value": 7.2, "unit": "%"},
        "effectiveDateTime": "2024-04-01T00:00:00Z",
    }]
    r = compute_hedis_score(_bundle(age=60, icd10_codes=["E11.9"],
                                            observations=obs))
    cdc = next(m for m in r.measures if m.measure_id == "CDC-HM2")
    assert cdc.eligible is True
    assert cdc.in_compliance is True


def test_hedis_diabetic_with_a1c_high_non_compliant():
    obs = [{
        "resourceType": "Observation",
        "code": {"coding": [{"system": "http://loinc.org",
                                "code": "4548-4"}]},
        "valueQuantity": {"value": 9.8, "unit": "%"},
        "effectiveDateTime": "2024-04-01T00:00:00Z",
    }]
    r = compute_hedis_score(_bundle(age=60, icd10_codes=["E11.9"],
                                            observations=obs))
    cdc = next(m for m in r.measures if m.measure_id == "CDC-HM2")
    assert cdc.in_compliance is False


def test_hedis_non_diabetic_not_eligible_for_a1c():
    r = compute_hedis_score(_bundle(age=60))
    cdc = next(m for m in r.measures if m.measure_id == "CDC-HM2")
    assert cdc.eligible is False


def test_hedis_cbp_compliant_with_in_range_bp():
    obs = [
        {"resourceType": "Observation",
         "code": {"coding": [{"system": "http://loinc.org",
                                  "code": "8480-6"}]},
         "valueQuantity": {"value": 128, "unit": "mm[Hg]"},
         "effectiveDateTime": "2024-04-01T00:00:00Z"},
        {"resourceType": "Observation",
         "code": {"coding": [{"system": "http://loinc.org",
                                  "code": "8462-4"}]},
         "valueQuantity": {"value": 78, "unit": "mm[Hg]"},
         "effectiveDateTime": "2024-04-01T00:00:00Z"},
    ]
    r = compute_hedis_score(_bundle(age=60, icd10_codes=["I10"],
                                            observations=obs))
    cbp = next(m for m in r.measures if m.measure_id == "CBP")
    assert cbp.eligible is True
    assert cbp.in_compliance is True


def test_hedis_cbp_non_compliant_with_high_bp():
    obs = [
        {"resourceType": "Observation",
         "code": {"coding": [{"system": "http://loinc.org",
                                  "code": "8480-6"}]},
         "valueQuantity": {"value": 156},
         "effectiveDateTime": "2024-04-01T00:00:00Z"},
        {"resourceType": "Observation",
         "code": {"coding": [{"system": "http://loinc.org",
                                  "code": "8462-4"}]},
         "valueQuantity": {"value": 96},
         "effectiveDateTime": "2024-04-01T00:00:00Z"},
    ]
    r = compute_hedis_score(_bundle(age=60, icd10_codes=["I10"],
                                            observations=obs))
    cbp = next(m for m in r.measures if m.measure_id == "CBP")
    assert cbp.in_compliance is False


def test_hedis_col_compliant_with_recent_colonoscopy():
    procs = [{
        "resourceType": "Procedure",
        "code": {"text": "Colonoscopy"},
        "performedDateTime": (datetime.now(timezone.utc)
                                  - timedelta(days=180)).isoformat(),
    }]
    r = compute_hedis_score(_bundle(age=60, procedures=procs))
    col = next(m for m in r.measures if m.measure_id == "COL")
    assert col.in_compliance is True


def test_hedis_col_non_compliant_no_screening():
    r = compute_hedis_score(_bundle(age=60))
    col = next(m for m in r.measures if m.measure_id == "COL")
    assert col.eligible is True
    assert col.in_compliance is False


def test_hedis_bcs_only_for_women_50_to_74():
    male = compute_hedis_score(_bundle(age=60, sex="male"))
    bcs_male = next(m for m in male.measures if m.measure_id == "BCS")
    assert bcs_male.eligible is False

    young = compute_hedis_score(_bundle(age=40, sex="female"))
    bcs_young = next(m for m in young.measures if m.measure_id == "BCS")
    assert bcs_young.eligible is False


def test_hedis_aab_compliant_no_antibiotic():
    r = compute_hedis_score(_bundle(age=40, icd10_codes=["J20.9"]))
    aab = next(m for m in r.measures if m.measure_id == "AAB")
    assert aab.eligible is True
    assert aab.in_compliance is True


def test_hedis_aab_non_compliant_with_amoxicillin():
    meds = [{
        "resourceType": "MedicationRequest",
        "medicationCodeableConcept": {"text": "amoxicillin 500 mg"},
    }]
    r = compute_hedis_score(_bundle(age=40, icd10_codes=["J20.9"],
                                            med_requests=meds))
    aab = next(m for m in r.measures if m.measure_id == "AAB")
    assert aab.in_compliance is False


def test_hedis_composite_score_in_unit_range():
    r = compute_hedis_score(_bundle(age=60, icd10_codes=["E11.9", "I10"]))
    assert 0.0 <= r.composite_score_pct <= 100.0


def test_hedis_n_eligible_consistent_with_measures():
    r = compute_hedis_score(_bundle(age=60, sex="female",
                                            icd10_codes=["E11.9", "I10"]))
    n_eligible = sum(1 for m in r.measures if m.eligible)
    assert r.n_measures_evaluated == n_eligible


# ───────────────────────────────────────────────────────────────────
# EVAL-2 -- Hospital Compare benchmark
# ───────────────────────────────────────────────────────────────────

def test_hospital_compare_invalid_tier_raises():
    with pytest.raises(ValueError, match="peer_tier"):
        compute_hospital_compare_benchmark(
            "Test", {"readmission_rate_30d_pct": 14.0},
            peer_tier="random")


def test_hospital_compare_empty_kpis_raises():
    with pytest.raises(ValueError, match="non-empty"):
        compute_hospital_compare_benchmark("Test", {},
                                                  peer_tier="academic_major")


def test_hospital_compare_better_kpis_high_percentile():
    """A hospital with lower-than-median readmission scores higher."""
    r = compute_hospital_compare_benchmark(
        institution_label="Hospital A",
        institution_kpis={"readmission_rate_30d_pct": 12.0},   # < 16.5 median
        peer_tier="academic_major",
    )
    assert r.percentile_rank["readmission_rate_30d_pct"] > 50


def test_hospital_compare_worse_kpis_low_percentile():
    r = compute_hospital_compare_benchmark(
        institution_label="Hospital B",
        institution_kpis={"readmission_rate_30d_pct": 22.0},   # > 16.5 median
        peer_tier="academic_major",
    )
    assert r.percentile_rank["readmission_rate_30d_pct"] < 50


def test_hospital_compare_higher_is_better_for_hcahps():
    """HCAHPS -- higher = better. Above-median should percentile-rank high."""
    r = compute_hospital_compare_benchmark(
        institution_label="Hospital C",
        institution_kpis={"hcahps_overall_pct": 85},   # > 72 median
        peer_tier="academic_major",
    )
    assert r.percentile_rank["hcahps_overall_pct"] > 50


def test_hospital_compare_unknown_kpi_skipped():
    """KPIs not in the peer-median table are skipped silently."""
    r = compute_hospital_compare_benchmark(
        institution_label="Hospital D",
        institution_kpis={"made_up_metric": 99},
        peer_tier="rural",
    )
    assert "made_up_metric" not in r.deltas_vs_median


def test_hospital_compare_composite_score_in_range():
    r = compute_hospital_compare_benchmark(
        institution_label="X",
        institution_kpis={"readmission_rate_30d_pct": 15.0,
                              "hcahps_overall_pct": 75},
        peer_tier="community_large",
    )
    assert 0.0 <= r.composite_quality_score <= 100.0


# ───────────────────────────────────────────────────────────────────
# EVAL-3 -- Schwartz quality framework
# ───────────────────────────────────────────────────────────────────

def _decision_card_full() -> dict:
    return {
        "recommendation": {"action": "home_with_care",
                              "confidence": "medium"},
        "reasoning": {
            "risk_estimate": {"model_version": "0.7.0",
                                "probability_mean": 0.30,
                                "lace_raw_score": 10,
                                "contributing_factors": [{"name": "L"}]},
            "utility_analysis": {"dominant_action": "home_with_care"},
        },
        "validation": {
            "grounding": {"overall_verdict": "supported"},
            "phi_check": {"risk_level": "none"},
        },
        "audit": {"request_id": "req-001"},
        "abstain": [],
        "self_critique": {"verdict": "approved"},
        "counseling": {"sections": [{"section_id": "your_medications"}]},
    }


def test_schwartz_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        compute_schwartz_quality_score(42)  # type: ignore[arg-type]


def test_schwartz_full_card_grades_high():
    r = compute_schwartz_quality_score(_decision_card_full())
    assert r.grade in ("A", "B")
    assert r.composite_score >= 0.7


def test_schwartz_minimal_card_grades_low():
    minimal = {"recommendation": None, "reasoning": {}, "validation": {},
                  "audit": {}, "abstain": []}
    r = compute_schwartz_quality_score(minimal)
    assert r.grade in ("D", "F")


def test_schwartz_each_pillar_in_unit_range():
    r = compute_schwartz_quality_score(_decision_card_full())
    for s in (r.trustworthy_score, r.relevant_score,
                  r.actionable_score, r.usable_score, r.composite_score):
        assert 0.0 <= s <= 1.0


def test_schwartz_weak_dimensions_listed_when_low():
    minimal = {"recommendation": {"action": "discharge_home"},
                  "reasoning": {}, "validation": {},
                  "audit": {}, "abstain": []}
    r = compute_schwartz_quality_score(minimal)
    assert "trustworthy" in r.weak_dimensions or \
        "actionable" in r.weak_dimensions


def test_schwartz_force_abstain_lowers_trustworthy():
    card = _decision_card_full()
    card["self_critique"] = {"verdict": "force_abstain"}
    r_abst = compute_schwartz_quality_score(card)
    card["self_critique"] = {"verdict": "approved"}
    r_appr = compute_schwartz_quality_score(card)
    assert r_appr.trustworthy_score >= r_abst.trustworthy_score


def test_schwartz_request_id_propagated():
    r = compute_schwartz_quality_score(_decision_card_full())
    assert r.request_id == "req-001"


def test_schwartz_supported_grounding_boosts_trustworthy():
    card = _decision_card_full()
    card["validation"]["grounding"]["overall_verdict"] = "unsupported"
    r_unsup = compute_schwartz_quality_score(card)
    card["validation"]["grounding"]["overall_verdict"] = "supported"
    r_sup = compute_schwartz_quality_score(card)
    assert r_sup.trustworthy_score > r_unsup.trustworthy_score


# ───────────────────────────────────────────────────────────────────
# EVAL-4 -- Comparative-effectiveness simulator
# ───────────────────────────────────────────────────────────────────

def test_comparative_invalid_n_raises():
    with pytest.raises(ValueError, match="n_patients"):
        compute_comparative_effectiveness(n_patients=5)


def test_comparative_invalid_baseline_raises():
    with pytest.raises(ValueError, match="baseline"):
        compute_comparative_effectiveness(n_patients=100,
                                                  baseline_event_rate=2.0)


def test_comparative_invalid_rrr_raises():
    with pytest.raises(ValueError, match="relative_risk_reduction"):
        compute_comparative_effectiveness(
            n_patients=100,
            intervention_relative_risk_reduction=1.5)


def test_comparative_seed_reproducible():
    a = compute_comparative_effectiveness(n_patients=200, seed=7)
    b = compute_comparative_effectiveness(n_patients=200, seed=7)
    assert a.arr_30d == b.arr_30d
    assert a.intervention_arm_event_rate == b.intervention_arm_event_rate


def test_comparative_high_rrr_gives_positive_arr():
    r = compute_comparative_effectiveness(
        n_patients=2000, baseline_event_rate=0.30,
        intervention_relative_risk_reduction=0.50, seed=1,
    )
    assert r.arr_30d > 0
    assert r.nnt is not None and r.nnt > 0


def test_comparative_zero_rrr_yields_arr_near_zero():
    r = compute_comparative_effectiveness(
        n_patients=5000, baseline_event_rate=0.20,
        intervention_relative_risk_reduction=0.0, seed=42,
    )
    assert abs(r.arr_30d) < 0.05   # noise band


def test_comparative_p_value_valid():
    r = compute_comparative_effectiveness(n_patients=1000)
    assert 0.0 <= r.p_value <= 1.0


def test_comparative_ci95_brackets_arr():
    r = compute_comparative_effectiveness(n_patients=5000,
                                                  seed=7)
    assert r.ci95_arr_low <= r.arr_30d <= r.ci95_arr_high


def test_comparative_event_rates_in_unit_interval():
    r = compute_comparative_effectiveness(n_patients=1000,
                                                  baseline_event_rate=0.40,
                                                  seed=42)
    assert 0.0 <= r.intervention_arm_event_rate <= 1.0
    assert 0.0 <= r.control_arm_event_rate <= 1.0
