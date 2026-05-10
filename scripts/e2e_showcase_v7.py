"""Phase 15.A -- E2E showcase v7.

Exercises the Phase-13/14 surface that the v5 + v6 showcases didn't
cover: critical_care, cardiology_depth, heme_onc_depth, rheumatology,
peri_op_risk, infectious_disease, gi_hepatology_depth, neurology_depth,
ob_peds_advanced, sleep_pain, endocrinology_advanced, transplant,
specialty_clinics, model_research, legacy_ehr_parsers.

Eight cross-bundle scenarios end-to-end, each captures (input, output,
latency_ms) and is rendered into a single self-contained HTML report.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/e2e_showcase_v7.py

Output:
    docs/e2e/v7/index.html
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mcp_server.tools.cardiology_depth import (
    compute_cha2ds2_vasc, compute_grace_acs_score,
    compute_has_bled, compute_timi_acs_score,
)
from mcp_server.tools.critical_care import (
    compute_apache_ii_score, compute_meld_score,
    compute_rifle_aki_classification, compute_sofa_score,
)
from mcp_server.tools.endocrinology_advanced import (
    compute_adrenal_insufficiency_workup,
    compute_hypocalcemia_severity, compute_thyroid_management,
)
from mcp_server.tools.gi_hepatology_depth import (
    compute_fib4_liver_fibrosis,
    compute_glasgow_blatchford_ugib,
    compute_maddrey_alcoholic_hepatitis,
    compute_rome_iv_ibs,
)
from mcp_server.tools.heart_score import compute_heart_score
from mcp_server.tools.heme_onc_depth import (
    compute_ecog_performance_status,
    compute_ipss_r_mds_score,
    compute_iss_myeloma_staging,
    compute_karnofsky_performance,
)
from mcp_server.tools.infectious_disease import (
    compute_hiv_management_tier, compute_lactate_clearance,
    compute_qsofa_score, compute_tb_risk_screen,
)
from mcp_server.tools.legacy_ehr_parsers import (
    compute_ccda_document_parse,
    compute_hl7v2_message_parse,
)
from mcp_server.tools.model_research import (
    compute_cox_proportional_hazards,
    compute_ensemble_stacking,
    compute_shap_attribution,
)
from mcp_server.tools.neurology_depth import (
    compute_hauser_ambulation_index, compute_hunt_hess_sah,
    compute_ich_score, compute_modified_rankin,
)
from mcp_server.tools.ob_peds_advanced import (
    compute_apgar_score, compute_bell_nec_stage,
    compute_bilirubin_nomogram, compute_bishop_induction_score,
)
from mcp_server.tools.peri_op_risk import (
    compute_ariscat_pulmonary_risk, compute_caprini_vte_risk,
    compute_rcri_cardiac_risk,
)
from mcp_server.tools.rheumatology import (
    compute_acr_eular_ra_classification,
    compute_asdas_axspa,
    compute_das28_rheumatoid_arthritis,
)
from mcp_server.tools.sleep_pain import (
    compute_dn4_neuropathic_pain,
    compute_epworth_sleepiness_scale,
    compute_stop_bang_osa_screen,
)
from mcp_server.tools.specialty_clinics import (
    compute_diabetic_retinopathy_severity,
    compute_lesion_triage, compute_pft_interpretation,
)
from mcp_server.tools.stroke_severity import compute_stroke_severity
from mcp_server.tools.transplant import (
    compute_epts_recipient_score,
    compute_immunosuppression_dose_check, compute_kdpi_kidney_donor,
)


OUT_DIR = ROOT / "docs" / "e2e" / "v7"


@dataclass
class ToolCall:
    tool: str
    bundle: str
    inputs: dict[str, Any]
    outputs: Any
    latency_ms: float


@dataclass
class Scenario:
    title: str
    summary: str
    bundles_touched: set[str] = field(default_factory=set)
    calls: list[ToolCall] = field(default_factory=list)


async def _exec(
    sc: Scenario, bundle: str, fn: Callable[..., Awaitable[Any]],
    **kwargs: Any,
) -> Any:
    t0 = time.perf_counter()
    out = await fn(**kwargs)
    latency = (time.perf_counter() - t0) * 1000
    sc.bundles_touched.add(bundle)
    sc.calls.append(ToolCall(
        tool=fn.__name__, bundle=bundle,
        inputs=dict(kwargs),
        outputs=out.model_dump(mode="json")
            if hasattr(out, "model_dump") else out,
        latency_ms=round(latency, 3),
    ))
    return out


# ─────────────────────────────────────────────────────────────────────
# S1. ICU Sepsis cascade
# ─────────────────────────────────────────────────────────────────────


async def s1_sepsis_cascade() -> Scenario:
    sc = Scenario(
        title="S1 - ICU Sepsis cascade",
        summary=(
            "62y M presents with altered mentation, RR 28, SBP 86. "
            "Lactate 4.2 mmol/L on arrival, 3.1 mmol/L at 2h. "
            "Pipeline: qSOFA -> lactate clearance -> SOFA -> APACHE II "
            "-> MELD (liver) -> RIFLE (kidney)."
        ),
    )
    await _exec(sc, "infectious_disease", compute_qsofa_score,
                altered_mentation_gcs_lt_15=True,
                respiratory_rate_ge_22=True,
                systolic_bp_le_100=True)
    await _exec(sc, "infectious_disease", compute_lactate_clearance,
                initial_lactate_mmol_l=4.2,
                repeat_lactate_mmol_l=3.1)
    await _exec(sc, "critical_care", compute_sofa_score,
                pao2_fio2_ratio=210, mechanical_ventilation=True,
                platelets_thousands_per_uL=110, bilirubin_mg_dl=1.5,
                mean_arterial_pressure_mmHg=62,
                pressors_doses={"norepinephrine": 0.08},
                glasgow_coma_scale=12, creatinine_mg_dl=2.1,
                suspected_infection=True)
    await _exec(sc, "critical_care", compute_apache_ii_score,
                age=62, temperature_c=39.1,
                mean_arterial_pressure_mmHg=62, heart_rate=128,
                respiratory_rate=28, fio2=0.5, pao2=72,
                arterial_ph=7.28, serum_sodium_mmol_l=135,
                serum_potassium_mmol_l=4.6,
                serum_creatinine_mg_dl=2.1,
                hematocrit_pct=33, wbc_thousands_per_uL=18,
                glasgow_coma_scale=12,
                acute_renal_failure=True, immunocompromised=False)
    await _exec(sc, "critical_care", compute_meld_score,
                bilirubin_mg_dl=1.5, creatinine_mg_dl=2.1,
                inr=1.5, sodium_mmol_l=135)
    await _exec(sc, "critical_care",
                compute_rifle_aki_classification,
                serum_creatinine_baseline_mg_dl=1.0,
                serum_creatinine_current_mg_dl=2.1,
                urine_output_ml_per_kg_per_hour=0.4)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S2. Cardiology AFib stroke + ACS
# ─────────────────────────────────────────────────────────────────────


async def s2_cardiology_acs() -> Scenario:
    sc = Scenario(
        title="S2 - Cardiology AFib + ACS rule-out",
        summary=(
            "74y F with paroxysmal AFib + chest pain. "
            "CHA2DS2-VASc + HAS-BLED for AC decision; HEART + TIMI + "
            "GRACE for the ACS workup."
        ),
    )
    await _exec(sc, "cardiology_depth", compute_cha2ds2_vasc,
                age=74, sex_female=True,
                congestive_heart_failure=False, hypertension=True,
                diabetes_mellitus=True,
                stroke_tia_thromboembolism_history=False,
                vascular_disease=True)
    await _exec(sc, "cardiology_depth", compute_has_bled,
                hypertension_uncontrolled_sbp_gt_160=True,
                abnormal_renal_function=False,
                age_gt_65=True,
                drugs_concomitant_antiplatelet_or_nsaid=False)
    await _exec(sc, "stroke_acs", compute_heart_score,
                history_descriptor="moderately_suspicious",
                ecg_descriptor="non_specific_repolarisation",
                age=74, risk_factors_count=3,
                known_atherosclerotic_disease=True,
                troponin_times_uln=0.0)
    await _exec(sc, "cardiology_depth", compute_timi_acs_score,
                age_ge_65=True,
                three_or_more_cad_risk_factors=True,
                known_cad_50_pct_stenosis=False,
                aspirin_use_in_last_7_days=True,
                severe_anginal_episodes_in_last_24h=True,
                st_deviation_ge_0_5_mm=False,
                elevated_cardiac_markers=False)
    await _exec(sc, "cardiology_depth", compute_grace_acs_score,
                age=74, heart_rate=98, systolic_bp=128,
                creatinine_mg_dl=1.2, killip_class=1,
                cardiac_arrest_at_admission=False,
                st_segment_deviation=False,
                elevated_cardiac_enzymes=False)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S3. Heme/Onco staging + chemo
# ─────────────────────────────────────────────────────────────────────


async def s3_heme_onc() -> Scenario:
    sc = Scenario(
        title="S3 - Heme/Onc staging",
        summary=(
            "67y M, IgG kappa myeloma + intermediate-risk MDS. "
            "ISS, IPSS-R, ECOG, Karnofsky."
        ),
    )
    await _exec(sc, "heme_onc_depth", compute_iss_myeloma_staging,
                serum_beta2_microglobulin_mg_l=4.8,
                serum_albumin_g_dl=3.2)
    await _exec(sc, "heme_onc_depth", compute_ipss_r_mds_score,
                cytogenetic_category="intermediate",
                bm_blast_pct=4.5, hemoglobin_g_dl=9.5,
                platelets_thousands_per_uL=85, anc_thousands_per_uL=1.1)
    await _exec(sc, "heme_onc_depth",
                compute_ecog_performance_status, grade=2)
    await _exec(sc, "heme_onc_depth",
                compute_karnofsky_performance, score=70)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S4. Peri-op rheumatology pre-arthroplasty
# ─────────────────────────────────────────────────────────────────────


async def s4_periop_rheum() -> Scenario:
    sc = Scenario(
        title="S4 - Peri-op rheumatology pre-arthroplasty",
        summary=(
            "62y F seropositive RA on methotrexate + prednisone. "
            "Pre-op work-up before elective right total knee "
            "arthroplasty: DAS28, ASDAS, ACR/EULAR + RCRI + ARISCAT "
            "+ Caprini."
        ),
    )
    await _exec(sc, "rheumatology",
                compute_das28_rheumatoid_arthritis,
                tender_joint_count_28=8, swollen_joint_count_28=6,
                esr_mm_per_hour=42,
                patient_global_assessment_vas_100=55)
    await _exec(sc, "rheumatology", compute_asdas_axspa,
                back_pain_vas_0_10=4.5,
                duration_morning_stiffness_vas_0_10=5.0,
                patient_global_vas_0_10=5.5,
                peripheral_pain_swelling_vas_0_10=4.0,
                crp_mg_l=12.0)
    await _exec(sc, "rheumatology",
                compute_acr_eular_ra_classification,
                n_small_joints_involved=8,
                n_large_joints_involved=2,
                rf_or_acpa_positive=True,
                rf_or_acpa_high_titre=True,
                elevated_crp_or_esr=True,
                symptom_duration_weeks_ge_6=True)
    await _exec(sc, "peri_op_risk", compute_rcri_cardiac_risk,
                high_risk_surgery=False,
                history_ischemic_heart_disease=False,
                history_congestive_heart_failure=False,
                history_cerebrovascular_disease=False,
                insulin_dependent_diabetes=False,
                creatinine_gt_2_mg_dl=False)
    await _exec(sc, "peri_op_risk", compute_ariscat_pulmonary_risk,
                age=62, preop_spo2_pct=96.0,
                surgical_incision_site="lower_extremity",
                surgical_duration_hours=1.5,
                respiratory_infection_last_month=False)
    await _exec(sc, "peri_op_risk", compute_caprini_vte_risk,
                age=62, bmi_gt_25=True,
                major_surgery_planned=True,
                history_vte=False, active_malignancy=False)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S5. Neuro emergency (SAH + ICH + NIHSS)
# ─────────────────────────────────────────────────────────────────────


async def s5_neuro_emergency() -> Scenario:
    sc = Scenario(
        title="S5 - Neuro emergency triage",
        summary=(
            "Two patients: SAH grade-III on imaging; spontaneous ICH "
            "with GCS 11 and 35 mL volume. NIHSS for the SAH patient "
            "+ Hunt-Hess + ICH score + modified Rankin + Hauser."
        ),
    )
    await _exec(sc, "stroke_acs", compute_stroke_severity,
                item_scores={
                    "facial_palsy": 2,
                    "motor_arm_left": 3, "motor_arm_right": 0,
                    "motor_leg_left": 2, "motor_leg_right": 0,
                    "language": 2, "dysarthria": 1,
                    "extinction_inattention": 1,
                    "level_of_consciousness": 1,
                    "loc_questions": 1, "loc_commands": 1,
                    "gaze": 1, "visual_fields": 1,
                    "ataxia": 1, "sensory": 1,
                },
                last_known_well_minutes_ago=120)
    await _exec(sc, "neurology_depth", compute_hunt_hess_sah, grade=3)
    await _exec(sc, "neurology_depth", compute_ich_score,
                glasgow_coma_scale=11, ich_volume_ml=35,
                intraventricular_hemorrhage=True,
                infratentorial_origin=False, age_ge_80=False)
    await _exec(sc, "neurology_depth", compute_modified_rankin, grade=3)
    await _exec(sc, "neurology_depth",
                compute_hauser_ambulation_index, grade=4)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S6. OB/Peds advanced
# ─────────────────────────────────────────────────────────────────────


async def s6_ob_peds() -> Scenario:
    sc = Scenario(
        title="S6 - OB/Peds advanced",
        summary=(
            "G1P0 at 39w admitted for elective induction; preterm "
            "neonate at 28w with suspected NEC; term neonate at 60h "
            "of life with TSB 13.5."
        ),
    )
    await _exec(sc, "ob_peds_advanced", compute_bishop_induction_score,
                cervical_dilation_cm=2.0,
                cervical_effacement_pct=50,
                fetal_station=-2,
                cervical_consistency="medium",
                cervical_position="anterior")
    await _exec(sc, "ob_peds_advanced", compute_apgar_score,
                one_min_appearance=1, one_min_pulse=2,
                one_min_grimace=1, one_min_activity=1,
                one_min_respiration=1,
                five_min_appearance=2, five_min_pulse=2,
                five_min_grimace=2, five_min_activity=2,
                five_min_respiration=2)
    await _exec(sc, "ob_peds_advanced", compute_bell_nec_stage,
                abdominal_distension=True,
                occult_blood_in_stool=True,
                radiographic_pneumatosis_intestinalis=True,
                persistent_metabolic_acidosis=False,
                portal_venous_gas=False,
                pneumoperitoneum=False,
                septic_shock_or_dic=False)
    await _exec(sc, "ob_peds_advanced", compute_bilirubin_nomogram,
                age_hours=60, total_bilirubin_mg_dl=13.5,
                gestational_age_weeks=39,
                has_neurologic_risk_factors=False)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S7. GI/Hepatology + ID screen
# ─────────────────────────────────────────────────────────────────────


async def s7_gi_hep_id() -> Scenario:
    sc = Scenario(
        title="S7 - GI/Hepatology + ID screen",
        summary=(
            "55y M with chronic alcohol use admitted for jaundice + "
            "melena; HIV co-infected on ART. Maddrey + FIB-4 + "
            "Glasgow-Blatchford + Rome IV (chronic IBS-mixed) + HIV "
            "tier + TB risk screen."
        ),
    )
    await _exec(sc, "gi_hepatology_depth",
                compute_maddrey_alcoholic_hepatitis,
                patient_pt_seconds=22.5, control_pt_seconds=12.5,
                serum_bilirubin_mg_dl=18.5)
    await _exec(sc, "gi_hepatology_depth",
                compute_fib4_liver_fibrosis,
                age=55, ast_iu_l=145, alt_iu_l=85,
                platelets_thousands_per_uL=95)
    await _exec(sc, "gi_hepatology_depth",
                compute_glasgow_blatchford_ugib,
                blood_urea_mmol_l=8.5, hemoglobin_g_dl=10.2,
                sex="male", systolic_bp_mmHg=104,
                pulse_ge_100=True, melena=True,
                hepatic_disease=True)
    await _exec(sc, "gi_hepatology_depth", compute_rome_iv_ibs,
                abdominal_pain_days_per_week=2,
                related_to_defecation=True,
                associated_with_change_in_stool_frequency=True,
                associated_with_change_in_stool_form=True,
                symptom_duration_months=8,
                bowel_pattern_predominant="mixed")
    await _exec(sc, "infectious_disease", compute_hiv_management_tier,
                cd4_count=380, viral_load_copies_ml=120, on_art=True)
    await _exec(sc, "infectious_disease", compute_tb_risk_screen,
                high_burden_country_residence_or_travel=True,
                immunocompromised_HIV_or_TNF_inhibitor=True,
                cough_gt_3_weeks_with_constitutional_symptoms=True)
    return sc


# ─────────────────────────────────────────────────────────────────────
# S8. Outpatient panel -- long-tail bundles
# ─────────────────────────────────────────────────────────────────────


async def s8_outpatient_panel() -> Scenario:
    sc = Scenario(
        title="S8 - Outpatient panel + research models + EHR ingest",
        summary=(
            "Outpatient panel of 4 patients: sleep / pain / endocrine "
            "screen, kidney transplant donor + recipient pairing, "
            "specialty clinic referrals (derm + ophtho + pulmo). "
            "Research models on cohort data + HL7 v2 ingestion + "
            "C-CDA chart parse."
        ),
    )
    # Sleep + pain
    await _exec(sc, "sleep_pain", compute_epworth_sleepiness_scale,
                sitting_reading=2, watching_tv=3,
                sitting_inactive_in_public=2,
                passenger_in_car_one_hour=1,
                lying_down_to_rest_afternoon=3,
                sitting_and_talking_to_someone=1,
                sitting_quietly_after_lunch=2,
                in_car_stopped_in_traffic=1)
    await _exec(sc, "sleep_pain", compute_stop_bang_osa_screen,
                snoring_loudly=True, tired_during_day=True,
                observed_apnea=False, high_blood_pressure=True,
                bmi_gt_35=True, age_gt_50=True,
                neck_circumference_gt_40_cm=True, male_sex=True)
    await _exec(sc, "sleep_pain", compute_dn4_neuropathic_pain,
                burning=True, electric_shocks=True, tingling=True,
                pins_and_needles=True, numbness=True,
                hypoesthesia_to_touch=True,
                hypoesthesia_to_pinprick=True)
    # Endocrine
    await _exec(sc, "endocrinology_advanced",
                compute_thyroid_management,
                tsh_mU_L=8.5, free_t4_ng_dl=0.8,
                current_levothyroxine_dose_mcg=75,
                on_levothyroxine=True)
    await _exec(sc, "endocrinology_advanced",
                compute_adrenal_insufficiency_workup,
                morning_cortisol_ug_dl=4.2,
                cortisol_after_acth_stim_ug_dl=12.0,
                acth_pg_ml=85)
    await _exec(sc, "endocrinology_advanced",
                compute_hypocalcemia_severity,
                serum_calcium_mg_dl=7.4, serum_albumin_g_dl=3.4,
                qtc_ms=470, symptomatic_tetany=True)
    # Transplant
    await _exec(sc, "transplant", compute_kdpi_kidney_donor,
                age=42, height_cm=178, weight_kg=82,
                ethnicity_african_american=False,
                history_of_hypertension=True,
                history_of_diabetes=False,
                cause_of_death_cva=False,
                serum_creatinine_mg_dl=1.1)
    await _exec(sc, "transplant", compute_epts_recipient_score,
                age=58, time_on_dialysis_years=3.5,
                prior_solid_organ_transplant=False, diabetes=True)
    await _exec(sc, "transplant",
                compute_immunosuppression_dose_check,
                drug="tacrolimus", weight_kg=82,
                cyp3a5_phenotype="poor_metabolizer",
                egfr_ml_min=68, on_strong_cyp3a4_inhibitor=False)
    # Specialty clinics
    await _exec(sc, "specialty_clinics", compute_lesion_triage,
                asymmetry=True, border_irregularity=True,
                color_variegated=True, diameter_mm=8.5,
                evolving_or_changing=True,
                seven_point_atypical_features=3,
                history_uv_exposure=True,
                family_history_melanoma=True)
    await _exec(sc, "specialty_clinics",
                compute_diabetic_retinopathy_severity,
                microaneurysms=True, retinal_hemorrhages_count=4,
                venous_beading=True,
                intraretinal_microvascular_abnormalities=True,
                neovascularization=False, vitreous_hemorrhage=False,
                macular_thickening=False)
    await _exec(sc, "specialty_clinics", compute_pft_interpretation,
                fev1_l=2.05, fvc_l=3.1, fev1_pct_predicted=62,
                asthma_control_test_score=18)
    # Model research
    await _exec(sc, "model_research",
                compute_cox_proportional_hazards,
                durations=[5.0, 8.0, 12.0, 3.0, 14.0, 7.0,
                            9.0, 11.0, 4.0, 13.0],
                events=[1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
                covariates=[
                    [0.1, 1.0], [0.4, 0.0], [0.2, 1.0],
                    [0.7, 0.0], [0.1, 1.0], [0.5, 0.0],
                    [0.3, 1.0], [0.6, 0.0], [0.8, 0.0],
                    [0.2, 1.0],
                ],
                feature_names=["lace_norm", "is_male"])
    await _exec(sc, "model_research", compute_shap_attribution,
                feature_values={
                    "L_los": 4, "A_acuity": 3,
                    "C_charlson": 2, "E_ed": 1,
                },
                coefficients={
                    "L_los": 0.06, "A_acuity": 0.10,
                    "C_charlson": 0.08, "E_ed": 0.05,
                },
                intercept=-1.5)
    await _exec(sc, "model_research", compute_ensemble_stacking,
                base_predictions={
                    "betabin": [0.1, 0.2, 0.3, 0.6, 0.8] * 4,
                    "gbm": [0.15, 0.18, 0.32, 0.55, 0.78] * 4,
                },
                labels=[0, 0, 1, 1, 1] * 4,
                train_split_pct=0.6)
    # Legacy EHR parsers
    hl7 = (
        "MSH|^~\\&|EPIC|HOSP|RECV|FAC|20260430090000||ADT^A01|"
        "MID01|P|2.5\r"
        "PID|1||MRN-V7^^^HOSP^MR||DOE^JOHN||19601115|M\r"
        "PV1|1|I|3W^301^A||||||MED\r"
        "OBX|1|NM|718-7^Hemoglobin^LN||10.1|g/dL|13.0-17.0|L|||F\r"
    )
    await _exec(sc, "legacy_ehr_parsers",
                compute_hl7v2_message_parse, message=hl7)
    ccd = """<?xml version="1.0"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <recordTarget><patientRole><id extension="MRN-V7"/>
    <patient><name><family>DOE</family><given>JOHN</given></name>
    <administrativeGenderCode code="M"/>
    <birthTime value="19601115"/></patient>
  </patientRole></recordTarget>
  <component><structuredBody><component>
    <section>
      <templateId root="2.16.840.1.113883.10.20.22.2.5.1"/>
      <entry><act><code code="I50.9" displayName="Heart failure"/></act></entry>
    </section>
  </component></structuredBody></component>
</ClinicalDocument>"""
    await _exec(sc, "legacy_ehr_parsers",
                compute_ccda_document_parse, xml_text=ccd)
    return sc


# ─────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TrustedRisk - E2E Showcase v7</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
            Helvetica, Arial, sans-serif; margin: 24px;
            background:#0d1117; color:#c9d1d9; max-width: 1200px; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  h2 {{ border-bottom: 1px solid #30363d; padding-bottom: 4px; }}
  details {{ margin: 6px 0; padding: 6px 12px;
              background:#161b22; border:1px solid #30363d;
              border-radius:4px; }}
  summary {{ cursor: pointer; color:#79c0ff; font-weight:600; }}
  table {{ border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th, td {{ border: 1px solid #30363d; padding: 4px 8px;
            text-align: left; vertical-align: top; }}
  th {{ background: #161b22; }}
  pre {{ background:#0a0f15; color:#e6edf3; padding: 6px;
          border-radius:4px; overflow-x:auto; max-width: 100%;
          font-size: 12px; }}
  .pill {{ display:inline-block; background:#1f6feb; color:#fff;
            padding:1px 8px; border-radius:12px; font-size:11px;
            margin-right:4px; }}
  .latency {{ color:#8b949e; font-size:11px; }}
</style></head>
<body>
<h1>TrustedRisk - E2E Showcase v7</h1>
<p class=latency>Generated {timestamp} - {n_scenarios} scenarios, {n_calls} tool calls,
{n_bundles} distinct bundles touched.</p>
{summary_block}
{scenarios_html}
</body></html>
"""


def _render(scenarios: list[Scenario]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_calls = sum(len(s.calls) for s in scenarios)
    bundles_total: set[str] = set()
    for s in scenarios:
        bundles_total.update(s.bundles_touched)

    summary_rows = "".join(
        "<tr>"
        f"<td>{html_lib.escape(s.title)}</td>"
        f"<td>{len(s.calls)}</td>"
        f"<td>{', '.join(sorted(s.bundles_touched))}</td>"
        "</tr>"
        for s in scenarios
    )
    summary_block = (
        "<h2>Scenario index</h2>"
        f"<table><thead><tr>"
        f"<th>Scenario</th><th>Tool calls</th><th>Bundles</th>"
        f"</tr></thead>"
        f"<tbody>{summary_rows}</tbody></table>"
        f"<p>Bundles covered ({len(bundles_total)}): "
        f"{', '.join(sorted(bundles_total))}.</p>"
    )

    parts: list[str] = []
    for s in scenarios:
        bundles = " ".join(
            f"<span class=pill>{html_lib.escape(b)}</span>"
            for b in sorted(s.bundles_touched)
        )
        rows = ""
        for c in s.calls:
            inputs_pretty = json.dumps(
                _to_jsonable(c.inputs), indent=2, default=str)
            outputs_pretty = json.dumps(
                _to_jsonable(c.outputs), indent=2, default=str)
            rows += (
                "<details>"
                f"<summary>{html_lib.escape(c.tool)} "
                f"<span class=pill>{html_lib.escape(c.bundle)}</span> "
                f"<span class=latency>{c.latency_ms} ms</span>"
                f"</summary>"
                f"<h3>Inputs</h3>"
                f"<pre>{html_lib.escape(inputs_pretty)}</pre>"
                f"<h3>Outputs</h3>"
                f"<pre>{html_lib.escape(outputs_pretty)}</pre>"
                f"</details>"
            )
        parts.append(
            f"<h2>{html_lib.escape(s.title)}</h2>"
            f"<p>{html_lib.escape(s.summary)}</p>"
            f"<p>{bundles}</p>"
            f"{rows}"
        )

    return _HTML_TEMPLATE.format(
        timestamp=timestamp,
        n_scenarios=len(scenarios), n_calls=n_calls,
        n_bundles=len(bundles_total),
        summary_block=summary_block,
        scenarios_html="\n".join(parts),
    )


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    return x


async def _amain() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = [
        await s1_sepsis_cascade(),
        await s2_cardiology_acs(),
        await s3_heme_onc(),
        await s4_periop_rheum(),
        await s5_neuro_emergency(),
        await s6_ob_peds(),
        await s7_gi_hep_id(),
        await s8_outpatient_panel(),
    ]
    html = _render(scenarios)
    out = OUT_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    n_calls = sum(len(s.calls) for s in scenarios)
    bundles: set[str] = set()
    for s in scenarios:
        bundles.update(s.bundles_touched)
    print(f"Wrote {out}")
    print(
        f"{len(scenarios)} scenarios, {n_calls} tool calls, "
        f"{len(bundles)} bundles touched."
    )
    return 0


def main() -> int:
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())
