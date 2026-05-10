"""Synthetic patient cohort generator for each of the 12 bundles (DATA-1).

For each bundle, generates 50 patients with realistic distributions over
the bundle's tool input domain. The cases are written as JSON
collections in `tests/golden/cohorts/{bundle_id}.json` and consumed by
the parametrized batch regression tests in `tests/golden/test_bundle_cohorts.py`.

Each bundle file has structure:
  {
    "bundle_id": "...",
    "tool": "compute_...",
    "cases": [
      {"name": "...", "input": {...},
       "expected_subset": {...} | "schema_only": true}
    ]
  }

The default expectation is "schema_only" -- meaning the case must produce
a Pydantic-valid output without crashing. Specific clinical assertions
are layered on top for known patterns (e.g., infant + lethargy +
fever -> severity_tier in (medium, high)).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT_DIR = ROOT / "tests" / "golden" / "cohorts"
N_PER_BUNDLE = 50
SEED = 4242


# ─────────────────────── Per-bundle generators ───────────────────────

def gen_pews_cohort(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for i in range(N_PER_BUNDLE):
        age_months = rng.randint(0, 216)
        # Skew distribution toward worrisome cases (sepsis-like)
        is_sick = rng.random() < 0.3
        cases.append({
            "name": f"pews_case_{i:03d}",
            "input": {
                "age_months": age_months,
                "behavior": rng.choice(
                    ["lethargic", "irritable", "appropriate", "sleeping"]
                    if is_sick else ["appropriate", "appropriate", "sleeping"]
                ),
                "heart_rate": rng.choice(
                    [180, 190, 200, 210] if is_sick
                    else [80, 100, 120, 140]
                ),
                "respiratory_rate": rng.choice(
                    [50, 60, 70] if is_sick
                    else [16, 20, 28, 35]
                ),
                "spo2": rng.choice([88, 90, 92] if is_sick
                                     else [96, 97, 98, 99]),
                "parental_or_nurse_concern": is_sick and rng.random() < 0.7,
            },
            "schema_only": True,
        })
    return cases


def gen_news2_cohort(rng: random.Random) -> list[dict[str, Any]]:
    from datetime import datetime, timezone
    cases = []
    iso = datetime.now(timezone.utc).isoformat()
    for i in range(N_PER_BUNDLE):
        is_septic = rng.random() < 0.25
        cases.append({
            "name": f"news2_case_{i:03d}",
            "input": {
                "vital_signs": [
                    {"type": "respiratory_rate",
                     "value": rng.randint(28, 35) if is_septic else rng.randint(12, 22),
                     "observed_at": iso},
                    {"type": "heart_rate",
                     "value": rng.randint(120, 140) if is_septic else rng.randint(60, 95),
                     "observed_at": iso},
                    {"type": "spo2",
                     "value": rng.randint(88, 92) if is_septic else rng.randint(95, 99),
                     "observed_at": iso},
                    {"type": "systolic_bp",
                     "value": rng.randint(80, 95) if is_septic else rng.randint(110, 145),
                     "observed_at": iso},
                    {"type": "temperature",
                     "value": rng.uniform(38.5, 39.5) if is_septic else rng.uniform(36.3, 37.5),
                     "observed_at": iso},
                    {"type": "consciousness",
                     "value": rng.choice(["V", "P"]) if is_septic else "A",
                     "observed_at": iso},
                ],
            },
            "schema_only": True,
        })
    return cases


def gen_heart_cohort(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for i in range(N_PER_BUNDLE):
        is_high_risk = rng.random() < 0.3
        cases.append({
            "name": f"heart_case_{i:03d}",
            "input": {
                "history_descriptor": rng.choice(
                    ["highly_suspicious", "moderately_suspicious"]
                    if is_high_risk
                    else ["non_suspicious", "slightly_suspicious",
                            "moderately_suspicious"]
                ),
                "ecg_descriptor": rng.choice(
                    ["significant_st_depression", "stemi"]
                    if is_high_risk
                    else ["normal", "non_specific_repolarization"]
                ),
                "age": rng.randint(45, 80) if is_high_risk else rng.randint(30, 70),
                "risk_factors_count": rng.randint(2, 5) if is_high_risk
                                         else rng.randint(0, 3),
                "known_atherosclerotic_disease": rng.random() < 0.4 if is_high_risk
                                                       else rng.random() < 0.1,
                "troponin_times_uln": rng.uniform(2.0, 8.0) if is_high_risk
                                        else rng.uniform(0.0, 1.0),
            },
            "schema_only": True,
        })
    return cases


def gen_nihss_cohort(rng: random.Random) -> list[dict[str, Any]]:
    items_max = {
        "loc_responsiveness": 3, "loc_questions": 2, "loc_commands": 2,
        "best_gaze": 2, "visual_fields": 3, "facial_palsy": 3,
        "motor_arm_left": 4, "motor_arm_right": 4,
        "motor_leg_left": 4, "motor_leg_right": 4,
        "limb_ataxia": 2, "sensory": 2, "best_language": 3,
        "dysarthria": 2, "extinction_inattention": 2,
    }
    cases = []
    for i in range(N_PER_BUNDLE):
        is_severe = rng.random() < 0.4
        scores = {
            it: rng.randint(0, max_pts) if (is_severe and rng.random() < 0.6)
                else rng.choice([0, 0, 0, 1, max_pts])
            for it, max_pts in items_max.items()
        }
        cases.append({
            "name": f"nihss_case_{i:03d}",
            "input": {
                "item_scores": scores,
                "last_known_well_minutes_ago": rng.randint(30, 1500),
            },
            "schema_only": True,
        })
    return cases


def gen_falls_cohort(rng: random.Random) -> list[dict[str, Any]]:
    falls_meds = [
        "lorazepam 1 mg", "diazepam 5 mg", "oxycodone 5 mg",
        "diphenhydramine 25 mg", "zolpidem 10 mg", "tramadol 50 mg",
        "amitriptyline 25 mg", "haloperidol 1 mg",
    ]
    safe_meds = [
        "lisinopril 10 mg", "metformin 1000 mg", "atorvastatin 40 mg",
        "vitamin d", "calcium 600 mg",
    ]
    cases = []
    for i in range(N_PER_BUNDLE):
        is_high_risk = rng.random() < 0.35
        n_meds = rng.randint(0, 4)
        meds = (rng.sample(falls_meds, k=min(n_meds, len(falls_meds)))
                if is_high_risk else
                rng.sample(safe_meds, k=min(n_meds, len(safe_meds))))
        cases.append({
            "name": f"falls_case_{i:03d}",
            "input": {
                "history_of_falling_3mo": is_high_risk and rng.random() < 0.7,
                "secondary_diagnosis_present": rng.random() < 0.6,
                "ambulatory_aid": rng.choice(
                    ["walker", "furniture"] if is_high_risk
                    else ["none", "cane"]
                ),
                "has_iv_or_heparin_lock": rng.random() < 0.5,
                "gait": rng.choice(
                    ["weak", "impaired"] if is_high_risk
                    else ["normal", "weak"]
                ),
                "mental_status": rng.choice(
                    ["forgets_limitations"] if is_high_risk
                    else ["oriented", "oriented", "forgets_limitations"]
                ),
                "current_medications": meds,
            },
            "schema_only": True,
        })
    return cases


def gen_dka_cohort(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for i in range(N_PER_BUNDLE):
        severity_class = rng.choice(["mild", "moderate", "severe", "not_dka"])
        if severity_class == "severe":
            ph, hco3 = rng.uniform(6.85, 6.99), rng.uniform(3, 9)
            ms = rng.choice(["stupor", "drowsy"])
        elif severity_class == "moderate":
            ph, hco3 = rng.uniform(7.00, 7.24), rng.uniform(10, 14)
            ms = rng.choice(["alert", "drowsy"])
        elif severity_class == "mild":
            ph, hco3 = rng.uniform(7.25, 7.30), rng.uniform(15, 18)
            ms = "alert"
        else:
            ph, hco3 = rng.uniform(7.32, 7.42), rng.uniform(22, 26)
            ms = "alert"
        cases.append({
            "name": f"dka_case_{i:03d}_{severity_class}",
            "input": {
                "ph": round(ph, 2), "bicarbonate_meq_l": round(hco3, 1),
                "glucose_mg_dl": rng.randint(280, 700) if severity_class != "not_dka"
                                    else rng.randint(120, 200),
                "ketones_present": severity_class != "not_dka",
                "mental_status": ms,
                "potassium_meq_l": round(rng.uniform(2.8, 5.5), 1),
                "weight_kg": rng.randint(50, 100),
            },
            "schema_only": True,
        })
    return cases


def gen_aki_cohort(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for i in range(N_PER_BUNDLE):
        baseline = round(rng.uniform(0.7, 1.5), 2)
        is_aki = rng.random() < 0.5
        current = (round(baseline * rng.uniform(1.5, 4.0), 2) if is_aki
                    else round(baseline * rng.uniform(0.95, 1.4), 2))
        cases.append({
            "name": f"aki_case_{i:03d}",
            "input": {
                "creatinine_baseline_mg_dl": baseline,
                "creatinine_current_mg_dl": current,
                "urine_output_ml_per_kg_per_hour": (
                    round(rng.uniform(0.1, 0.5), 2) if is_aki
                    else round(rng.uniform(0.6, 1.5), 2)
                ),
                "urine_output_window_hours": rng.choice([6, 12, 24]),
                "nephrotoxic_medications_present": rng.random() < 0.4,
            },
            "schema_only": True,
        })
    return cases


def gen_imaging_cohort(rng: random.Random) -> list[dict[str, Any]]:
    scenarios = [
        "chest_pain_acs_workup", "suspected_pe_low_risk",
        "suspected_pe_high_risk", "acute_abdominal_pain",
        "acute_lbp_no_red_flag", "acute_headache_thunderclap",
        "acute_stroke", "blunt_abdominal_trauma",
        "pediatric_appendicitis", "chest_pain_low_risk",
    ]
    cases = []
    for i in range(N_PER_BUNDLE):
        scenario = rng.choice(scenarios)
        cases.append({
            "name": f"imaging_case_{i:03d}_{scenario}",
            "input": {
                "clinical_scenario": scenario,
                "patient_age": rng.randint(3, 90),
                "patient_pregnant": rng.random() < 0.05,
                "cumulative_radiation_msv_last_12mo": (
                    rng.uniform(0, 70) if rng.random() < 0.2 else 0
                ),
            },
            "schema_only": True,
        })
    return cases


def gen_contrast_cohort(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for i in range(N_PER_BUNDLE):
        cases.append({
            "name": f"contrast_case_{i:03d}",
            "input": {
                "contrast_type": rng.choice(
                    ["iodinated_iv", "gadolinium_iv", "no_contrast"]
                ),
                "egfr_ml_min": rng.choice(
                    [None, rng.randint(15, 45), rng.randint(60, 120)]
                ),
                "on_metformin": rng.random() < 0.3,
                "iodine_contrast_prior_severe_reaction": rng.random() < 0.05,
                "iodine_contrast_prior_mild_reaction": rng.random() < 0.10,
                "pregnant": rng.random() < 0.05,
            },
            "schema_only": True,
        })
    return cases


def gen_polypharmacy_cohort(rng: random.Random) -> list[dict[str, Any]]:
    drug_pool = [
        ("warfarin 5mg", "anticoagulant_vka"),
        ("apixaban 5mg", "anticoagulant_doac"),
        ("aspirin 81mg", "antiplatelet"),
        ("ibuprofen 400mg", "nsaid"),
        ("lisinopril 10mg", "ace_inhibitor"),
        ("losartan 50mg", "arb"),
        ("spironolactone 25mg", "mra"),
        ("furosemide 40mg", "loop_diuretic"),
        ("metoprolol 25mg", "beta_blocker"),
        ("atorvastatin 40mg", "statin"),
        ("metformin 1000mg", "biguanide"),
        ("insulin glargine", "insulin"),
        ("amoxicillin 500mg", "antibiotic"),
        ("oxycodone 5mg", "opioid"),
    ]
    cases = []
    for i in range(N_PER_BUNDLE):
        n = rng.randint(2, 12)
        meds = rng.sample(drug_pool, k=min(n, len(drug_pool)))
        cases.append({
            "name": f"polypharmacy_case_{i:03d}",
            "input": {
                "medications": [
                    {"name": name, "drug_class": dc, "status": "active"}
                    for name, dc in meds
                ],
            },
            "schema_only": True,
        })
    return cases


def gen_suicide_cohort(rng: random.Random) -> list[dict[str, Any]]:
    cases = []
    for i in range(N_PER_BUNDLE):
        is_imminent = rng.random() < 0.20
        ideation_recent = (rng.choice([4, 5]) if is_imminent
                            else rng.randint(0, 3))
        cases.append({
            "name": f"suicide_case_{i:03d}",
            "input": {
                "ideation_lifetime_level": rng.randint(0, 5),
                "ideation_past_30d_level": ideation_recent,
                "behavior_lifetime_attempts": (
                    rng.randint(1, 3) if is_imminent and rng.random() < 0.5
                    else 0
                ),
                "behavior_past_30d_any": is_imminent and rng.random() < 0.4,
                "warning_factors_count": rng.randint(0, 5),
                "protective_factors_count": rng.randint(0, 5),
                "patient_demographics": rng.choice([
                    None, {"race": "white"}, {"race": "asian"},
                    {"race": "black"}, {"sexual_orientation": "lgbtq"},
                ]),
            },
            "schema_only": True,
        })
    return cases


def gen_trauma_cohort(rng: random.Random) -> list[dict[str, Any]]:
    regions = ["head_neck", "face", "chest", "abdomen_pelvis",
                "extremities_pelvic_girdle", "external"]
    cases = []
    for i in range(N_PER_BUNDLE):
        n_inj = rng.randint(0, 4)
        cases.append({
            "name": f"trauma_case_{i:03d}",
            "input": {
                "injuries": [
                    {"body_region": rng.choice(regions),
                     "ais_severity": rng.randint(1, 5)}
                    for _ in range(n_inj)
                ],
                "glasgow_coma_score": rng.randint(3, 15),
                "systolic_bp": rng.randint(60, 200),
                "respiratory_rate": rng.randint(8, 40),
                "has_active_hemorrhage": rng.random() < 0.3,
            },
            "schema_only": True,
        })
    return cases


# ─────────────────────── Bundle -> generator + tool mapping ───────────────────────

_BUNDLE_GENERATORS = [
    ("pediatric", "compute_pediatric_early_warning", gen_pews_cohort),
    ("ed_acute", "compute_clinical_deterioration_score", gen_news2_cohort),
    ("stroke_acs", "compute_stroke_severity", gen_nihss_cohort),
    ("stroke_acs_heart", "compute_heart_score", gen_heart_cohort),  # 2nd cohort
    ("obstetric_geriatric", "compute_falls_risk_morse", gen_falls_cohort),
    ("endocrine_acute", "compute_dka_severity", gen_dka_cohort),
    ("nephrology", "compute_aki_kdigo_stage", gen_aki_cohort),
    ("imaging_appropriateness", "compute_imaging_appropriateness",
     gen_imaging_cohort),
    ("imaging_contrast", "compute_contrast_safety_check",
     gen_contrast_cohort),
    ("core_discharge_polypharmacy", "detect_polypharmacy_concerns",
     gen_polypharmacy_cohort),
    ("mental_health", "compute_suicide_risk_assessment",
     gen_suicide_cohort),
    ("trauma_critical", "compute_trauma_severity_score", gen_trauma_cohort),
]


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    n_total = 0
    for bundle_id, tool_name, gen_fn in _BUNDLE_GENERATORS:
        cases = gen_fn(rng)
        out_path = OUTPUT_DIR / f"{bundle_id}.json"
        out_path.write_text(json.dumps({
            "bundle_id": bundle_id, "tool": tool_name,
            "n_cases": len(cases), "cases": cases,
        }, indent=2), encoding="utf-8")
        print(f"  {bundle_id}: {len(cases)} cases -> {out_path.relative_to(ROOT)}")
        n_total += len(cases)
    print(f"\nGenerated {n_total} synthetic cases across "
            f"{len(_BUNDLE_GENERATORS)} bundle cohorts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
