"""Unit tests for LIB-1 / LIB-2 / LIB-3 / LIB-4 clinical-workflow tools."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest


def _run(coro):
    return asyncio.run(coro)


# ───────────────────────────────────────────────────────────────────
# LIB-1 -- compute_order_set
# ───────────────────────────────────────────────────────────────────

from mcp_server.tools.order_set_generator import compute_order_set


def test_order_set_invalid_action_raises():
    with pytest.raises(ValueError, match="discharge_action"):
        _run(compute_order_set("nope"))


def test_order_set_high_risk_short_window():
    r = _run(compute_order_set("home_with_care",
                                      risk_estimate={"probability_mean": 0.45}))
    assert r.risk_tier == "high"
    assert r.next_visit_window_days == (3, 7)


def test_order_set_low_risk_long_window():
    r = _run(compute_order_set("discharge_home",
                                      risk_estimate={"probability_mean": 0.05}))
    assert r.risk_tier == "low"
    assert r.next_visit_window_days == (14, 30)


def test_order_set_home_with_care_includes_home_health():
    r = _run(compute_order_set("home_with_care",
                                      risk_estimate={"probability_mean": 0.30}))
    consult_orders = [o for o in r.orders if o.category == "consult"]
    assert any("home health" in o.order_text.lower()
                  for o in consult_orders)


def test_order_set_warfarin_adds_inr_monitoring():
    r = _run(compute_order_set(
        "discharge_home", risk_estimate={"probability_mean": 0.20},
        medications=[{"name": "warfarin 5 mg",
                          "drug_class": "anticoagulant_vka",
                          "frequency": "daily"}],
    ))
    lab_orders = [o for o in r.orders if o.category == "lab_monitoring"]
    assert any("INR" in o.order_text for o in lab_orders)


def test_order_set_chf_chief_complaint_adds_cardiology():
    r = _run(compute_order_set(
        "home_with_care", risk_estimate={"probability_mean": 0.30},
        chief_complaint="acute decompensated CHF",
    ))
    consult_orders = [o for o in r.orders if o.category == "consult"]
    assert any("cardiology" in o.order_text.lower()
                  for o in consult_orders)


def test_order_set_aki_chief_complaint_adds_nephrology():
    r = _run(compute_order_set("home_with_care",
                                      risk_estimate={"probability_mean": 0.20},
                                      chief_complaint="AKI stage 2 resolved"))
    assert any("nephrology" in o.order_text.lower() for o in r.orders)


def test_order_set_continued_admission_adds_q4_vitals():
    r = _run(compute_order_set("continued_admission",
                                      risk_estimate={"probability_mean": 0.50}))
    vital_orders = [o for o in r.orders if o.category == "vital_monitoring"]
    assert any("4 hours" in o.order_text for o in vital_orders)


def test_order_set_high_risk_includes_teach_back():
    r = _run(compute_order_set("home_with_care",
                                      risk_estimate={"probability_mean": 0.40}))
    edu_orders = [o for o in r.orders if o.category == "patient_education"]
    assert any("teach-back" in o.order_text.lower() for o in edu_orders)


# ───────────────────────────────────────────────────────────────────
# LIB-2 -- compute_medication_adherence_predictor
# ───────────────────────────────────────────────────────────────────

from mcp_server.tools.medication_adherence import (
    compute_medication_adherence_predictor,
)


def test_adherence_non_list_raises():
    with pytest.raises(ValueError, match="must be a list"):
        _run(compute_medication_adherence_predictor("warfarin"))  # type: ignore


def test_adherence_empty_list_low_risk():
    r = _run(compute_medication_adherence_predictor([]))
    assert r.n_medications == 0
    assert r.risk_tier == "low"


def test_adherence_polypharmacy_lowers_probability():
    """≥10 meds -> polypharmacy_severe factor + lower probability."""
    meds = [{"name": f"drug{i}", "drug_class": "other",
                "frequency": "daily"} for i in range(12)]
    r = _run(compute_medication_adherence_predictor(meds))
    assert r.adherence_30d_probability < 0.6
    assert any("polypharmacy_severe" in f.name for f in r.contributing_factors)


def test_adherence_caregiver_boost_increases_probability():
    meds = [{"name": "warfarin", "drug_class": "anticoagulant_vka",
                "frequency": "daily"}]
    no_caregiver = _run(compute_medication_adherence_predictor(
        meds, has_caregiver=False))
    with_caregiver = _run(compute_medication_adherence_predictor(
        meds, has_caregiver=True))
    assert (with_caregiver.adherence_30d_probability
                > no_caregiver.adherence_30d_probability)


def test_adherence_qid_dosing_flagged():
    meds = [{"name": "drug1", "frequency": "QID"}]
    r = _run(compute_medication_adherence_predictor(meds))
    assert r.distinct_dose_times == 4
    assert any("dose_schedule" in f.name for f in r.contributing_factors)


def test_adherence_prior_non_adherence_dominant():
    meds = [{"name": "warfarin", "drug_class": "anticoagulant_vka",
                "frequency": "daily"}]
    r = _run(compute_medication_adherence_predictor(
        meds, prior_adherence_known=False))
    assert any("prior_non_adherence" in f.name
                  for f in r.contributing_factors)
    assert r.adherence_30d_probability < 0.5


def test_adherence_interventions_targeted_to_factors():
    meds = [{"name": "warfarin", "drug_class": "anticoagulant_vka",
                "frequency": "QID"} for _ in range(7)]
    r = _run(compute_medication_adherence_predictor(
        meds, patient_age=80, insurance_type="medicaid"))
    blob = " ".join(r.interventions_recommended).lower()
    assert "blister-pack" in blob or "pill organizer" in blob
    assert "geriatric" in blob or "brown-bag" in blob
    assert "generic" in blob or "patient-assistance" in blob


def test_adherence_complexity_index_increases_with_dose_freq():
    """A 4-drugs-QID regimen has higher complexity than 4-drugs-daily."""
    daily = [{"name": f"d{i}", "frequency": "daily"} for i in range(4)]
    qid = [{"name": f"d{i}", "frequency": "QID"} for i in range(4)]
    r1 = _run(compute_medication_adherence_predictor(daily))
    r2 = _run(compute_medication_adherence_predictor(qid))
    assert r2.regimen_complexity_index > r1.regimen_complexity_index


# ───────────────────────────────────────────────────────────────────
# LIB-3 -- compute_care_gap_detector
# ───────────────────────────────────────────────────────────────────

from mcp_server.tools.care_gap_detector import compute_care_gap_detector


def _patient_bundle(*, age: int = 60, sex: str = "female",
                       conditions: list[str] | None = None,
                       observations: list[dict] | None = None,
                       immunizations: list[dict] | None = None,
                       procedures: list[dict] | None = None) -> dict:
    today = datetime.now(timezone.utc).date()
    dob = today.replace(year=today.year - age)
    bundle = {"resourceType": "Bundle", "type": "collection",
                "entry": [
                    {"resource": {"resourceType": "Patient",
                                     "id": "pt-1", "gender": sex,
                                     "birthDate": dob.isoformat()}}
                ]}
    for c in conditions or []:
        bundle["entry"].append({"resource": {
            "resourceType": "Condition",
            "code": {"coding": [{
                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                "code": c,
            }]},
        }})
    for o in observations or []:
        bundle["entry"].append({"resource": o})
    for i in immunizations or []:
        bundle["entry"].append({"resource": i})
    for p in procedures or []:
        bundle["entry"].append({"resource": p})
    return bundle


def test_care_gap_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        _run(compute_care_gap_detector("not a dict"))  # type: ignore[arg-type]


def test_care_gap_woman_60_no_history_finds_gaps():
    bundle = _patient_bundle(age=60, sex="female")
    r = _run(compute_care_gap_detector(bundle))
    titles = " ".join(g.title for g in r.high_priority_gaps
                          + r.moderate_priority_gaps).lower()
    assert "mammography" in titles
    assert "colorectal" in titles
    assert "influenza" in titles


def test_care_gap_man_70_smoker_finds_aaa():
    bundle = _patient_bundle(age=70, sex="male")
    r = _run(compute_care_gap_detector(bundle))
    aaa_gaps = [g for g in r.moderate_priority_gaps
                  if "aortic" in g.title.lower()]
    assert len(aaa_gaps) == 1


def test_care_gap_a1c_only_for_diabetic_patients():
    """A1c gap should NOT fire on a non-diabetic; SHOULD fire on diabetic."""
    non_dm = _run(compute_care_gap_detector(
        _patient_bundle(age=55, sex="female")))
    a1c_titles = " ".join(g.title for g in non_dm.high_priority_gaps).lower()
    assert "a1c" not in a1c_titles

    dm = _run(compute_care_gap_detector(
        _patient_bundle(age=55, sex="female", conditions=["E11.9"])))
    a1c_titles_dm = " ".join(
        g.title for g in dm.high_priority_gaps).lower()
    assert "a1c" in a1c_titles_dm


def test_care_gap_recent_screening_does_not_fire():
    """A mammogram done 6 months ago should NOT trigger the gap."""
    six_months_ago = (datetime.now(timezone.utc)
                          - timedelta(days=180)).isoformat()
    bundle = _patient_bundle(
        age=60, sex="female",
        procedures=[{
            "resourceType": "Procedure",
            "code": {"text": "Mammography"},
            "performedDateTime": six_months_ago,
        }],
    )
    r = _run(compute_care_gap_detector(bundle))
    titles = " ".join(g.title for g in r.high_priority_gaps).lower()
    assert "mammography" not in titles


def test_care_gap_old_flu_shot_overdue():
    """A flu shot from 2 years ago should be overdue."""
    two_years_ago = (datetime.now(timezone.utc)
                         - timedelta(days=730)).isoformat()
    bundle = _patient_bundle(
        age=50,
        immunizations=[{
            "resourceType": "Immunization",
            "vaccineCode": {"coding": [{
                "system": "http://hl7.org/fhir/sid/cvx",
                "code": "150",
            }]},
            "occurrenceDateTime": two_years_ago,
        }],
    )
    r = _run(compute_care_gap_detector(bundle))
    flu = next((g for g in r.high_priority_gaps
                  if "influenza" in g.title.lower()), None)
    assert flu is not None
    assert flu.overdue_days is not None and flu.overdue_days > 0


def test_care_gap_priority_buckets_distinct():
    """High / moderate / low gaps don't appear in multiple buckets."""
    bundle = _patient_bundle(age=70, sex="female")
    r = _run(compute_care_gap_detector(bundle))
    high_ids = {g.gap_id for g in r.high_priority_gaps}
    mod_ids = {g.gap_id for g in r.moderate_priority_gaps}
    low_ids = {g.gap_id for g in r.low_priority_gaps}
    assert high_ids.isdisjoint(mod_ids)
    assert high_ids.isdisjoint(low_ids)
    assert mod_ids.isdisjoint(low_ids)


def test_care_gap_n_gaps_matches_bucket_sums():
    bundle = _patient_bundle(age=60, sex="female")
    r = _run(compute_care_gap_detector(bundle))
    assert r.n_gaps_found == (len(r.high_priority_gaps)
                                  + len(r.moderate_priority_gaps)
                                  + len(r.low_priority_gaps))


# ───────────────────────────────────────────────────────────────────
# LIB-4 -- compute_prom_influence
# ───────────────────────────────────────────────────────────────────

from mcp_server.tools.prom_influence import compute_prom_influence


def test_prom_eq5d_score_drives_qaly():
    r = _run(compute_prom_influence({
        "instrument": "EQ-5D-5L", "eq5d_index_score": 0.85,
    }))
    expected = 0.85 * (30.0 / 365.0)
    assert abs(r.qaly_delta_30d - expected) < 1e-3


def test_prom_high_burden_shifts_away_from_discharge_home():
    r = _run(compute_prom_influence({
        "instrument": "PROMIS-29", "pain_intensity": 9.0,
        "fatigue": 80, "depression": 75, "anxiety": 70,
        "sleep_disturbance": 65, "physical_function": 30,
    }))
    assert r.distress_burden_score >= 70
    assert r.action_dominance_shift["discharge_home"] < 0
    assert r.action_dominance_shift["home_with_care"] > 0 or \
           r.action_dominance_shift["snf"] > 0


def test_prom_low_burden_no_shift():
    r = _run(compute_prom_influence({
        "instrument": "PROMIS-29", "pain_intensity": 1.0,
        "fatigue": 40, "depression": 35, "anxiety": 35,
        "sleep_disturbance": 40, "physical_function": 70,
    }))
    assert r.distress_burden_score < 50
    assert r.action_dominance_shift["discharge_home"] >= 0


def test_prom_high_burden_dimensions_listed():
    r = _run(compute_prom_influence({
        "instrument": "PROMIS-29",
        "pain_intensity": 8.0, "fatigue": 70, "depression": 65,
    }))
    assert "pain_intensity" in r.high_burden_dimensions
    assert "fatigue" in r.high_burden_dimensions
    assert "depression" in r.high_burden_dimensions


def test_prom_recommendation_severity_tiered():
    r_severe = _run(compute_prom_influence({
        "instrument": "PROMIS-29", "pain_intensity": 9.0,
        "fatigue": 80, "depression": 80, "anxiety": 75,
        "sleep_disturbance": 70,
    }))
    r_moderate = _run(compute_prom_influence({
        "instrument": "PROMIS-29", "pain_intensity": 5.0,
        "fatigue": 55, "depression": 55,
    }))
    r_low = _run(compute_prom_influence({
        "instrument": "PROMIS-29", "pain_intensity": 1.0,
        "fatigue": 40,
    }))
    assert "Severe" in r_severe.recommendation or "escalate" in \
        r_severe.recommendation.lower()
    assert "Moderate" in r_moderate.recommendation or \
        "review" in r_moderate.recommendation.lower()
    assert "norms" in r_low.recommendation.lower() or \
        "proceed" in r_low.recommendation.lower()


def test_prom_dict_input_coerced():
    payload = {"instrument": "EQ-5D-5L", "eq5d_index_score": 0.5}
    r = _run(compute_prom_influence(payload))
    assert r.instrument == "EQ-5D-5L"


def test_prom_low_physical_function_shifts_to_snf():
    r = _run(compute_prom_influence({
        "instrument": "PROMIS-29", "physical_function": 25,
        "pain_intensity": 5.0, "fatigue": 60,
    }))
    assert r.action_dominance_shift["snf"] > 0


# ───────────────────────────────────────────────────────────────────
# Bundle registration
# ───────────────────────────────────────────────────────────────────

def test_clinical_workflow_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "clinical_workflow" in BUNDLES
    expected = {
        "compute_order_set",
        "compute_medication_adherence_predictor",
        "compute_care_gap_detector",
        "compute_prom_influence",
    }
    assert expected <= set(BUNDLES["clinical_workflow"])
