"""Phase 11.6 -- 5 end-to-end clinical scenario runners.

Each scenario chains 6+ deterministic-floor tools end-to-end and
returns a `ScenarioRun` summary that the test suite + the tutorial
notebooks both consume. No live FHIR -- every scenario provides its
inputs inline, so the runs are reproducible in CI without network.

Scenarios:
  1. Acute stroke + LVO + reperfusion
  2. Sepsis bundle + antibiogram-driven empiric pick
  3. Polytrauma + MTP activation
  4. Geriatric polypharmacy med review
  5. Mental-health crisis + admission decision

Public API:
    `run_scenario(scenario_id) -> ScenarioRun`
    `run_all_scenarios() -> list[ScenarioRun]`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class ScenarioStep:
    tool: str
    summary: str
    output: Any = None


@dataclass
class ScenarioRun:
    scenario_id: str
    title: str
    steps: list[ScenarioStep] = field(default_factory=list)
    final_summary: str = ""

    @property
    def n_tools_invoked(self) -> int:
        return len(self.steps)


# ─────────────────────────────────────────────────────────────────────
# Scenario 1 -- Acute stroke + LVO
# ─────────────────────────────────────────────────────────────────────


async def scenario_acute_stroke() -> ScenarioRun:
    from mcp_server.tools.stroke_severity import compute_stroke_severity
    from mcp_server.tools.stroke_thrombolysis_eligibility import (
        compute_stroke_thrombolysis_eligibility,
    )
    from mcp_server.tools.imaging_appropriateness import (
        compute_imaging_appropriateness,
    )
    from mcp_server.tools.contrast_safety_check import (
        compute_contrast_safety_check,
    )
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    from mcp_server.tools.detect_phi import detect_phi

    run = ScenarioRun(
        scenario_id="acute_stroke_lvo",
        title="Acute stroke + LVO + reperfusion eligibility",
    )

    nihss = await compute_stroke_severity(
        item_scores={
            "loc": 0, "loc_questions": 1, "loc_commands": 0,
            "gaze": 2, "visual_fields": 2, "facial_palsy": 2,
            "motor_arm_left": 0, "motor_arm_right": 4,
            "motor_leg_left": 0, "motor_leg_right": 2,
            "limb_ataxia": 0, "sensory": 1, "language": 2,
            "dysarthria": 1, "extinction": 2,
        },
        last_known_well_minutes_ago=110,
    )
    run.steps.append(ScenarioStep(
        "compute_stroke_severity",
        f"NIHSS {nihss.score_total} ({nihss.severity_tier}, "
        f"LVO suspected={nihss.lvo_suspected})",
        nihss,
    ))

    elig = await compute_stroke_thrombolysis_eligibility(
        last_known_well_minutes_ago=110,
        nihss_total=nihss.score_total,
        clinical_factors={
            "bp_systolic": 158, "bp_diastolic": 92,
            "glucose_mg_dl": 132,
            "on_anticoagulant": False,
            "platelets_per_uL": 240_000,
            "inr": 1.0,
            "aspects_score": 8,
            "lvo_confirmed": True,
        },
    )
    run.steps.append(ScenarioStep(
        "compute_stroke_thrombolysis_eligibility",
        f"Decision: {elig.decision}",
        elig,
    ))

    img = await compute_imaging_appropriateness(
        clinical_scenario="acute_stroke_workup",
        patient_age=72,
    )
    run.steps.append(ScenarioStep(
        "compute_imaging_appropriateness",
        f"Top modality: {img.top_pick_modality}",
        img,
    ))

    contrast = await compute_contrast_safety_check(
        contrast_type="iodinated_iv", egfr_ml_min=68.0,
    )
    run.steps.append(ScenarioStep(
        "compute_contrast_safety_check",
        f"Proceed: {contrast.proceed_with_contrast}",
        contrast,
    ))

    aki = await compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0,
        creatinine_current_mg_dl=1.05,
    )
    run.steps.append(ScenarioStep(
        "compute_aki_kdigo_stage",
        f"AKI stage {aki.aki_stage}",
        aki,
    ))

    phi = await detect_phi(
        text="Acute stroke alert at 09:14, LKW 07:30, NIHSS 17.",
    )
    run.steps.append(ScenarioStep(
        "detect_phi", f"PHI risk: {phi.risk_level}", phi,
    ))

    run.final_summary = (
        f"NIHSS {nihss.score_total} -> {elig.decision}; "
        f"top imaging modality {img.top_pick_modality}; "
        f"AKI stage {aki.aki_stage}."
    )
    return run


# ─────────────────────────────────────────────────────────────────────
# Scenario 2 -- Sepsis bundle + antibiogram empiric pick
# ─────────────────────────────────────────────────────────────────────


async def scenario_sepsis_bundle() -> ScenarioRun:
    from mcp_server.tools.clinical_deterioration_score import (
        compute_clinical_deterioration_score,
    )
    from mcp_server.tools.empiric_antibiotic_selection import (
        compute_empiric_antibiotic_selection,
    )
    from mcp_server.tools.lab_trend_analysis import compute_lab_trend_analysis
    from mcp_server.tools.discharge_counseling import (
        compute_discharge_counseling,
    )
    from mcp_server.tools.detect_phi import detect_phi

    run = ScenarioRun(
        scenario_id="sepsis_bundle",
        title="Sepsis bundle + antibiogram-driven empiric pick",
    )

    news2 = await compute_clinical_deterioration_score(
        vital_signs=[{
            "respiration_rate": 26,
            "oxygen_saturation_pct": 91,
            "on_supplemental_oxygen": True,
            "temperature_c": 38.4,
            "systolic_bp_mmHg": 96,
            "heart_rate": 122,
            "consciousness_avpu": "V",
        }],
    )
    run.steps.append(ScenarioStep(
        "compute_clinical_deterioration_score",
        f"NEWS2 {news2.score_total} ({news2.severity_tier}); "
        f"response = {news2.recommended_response}",
        news2,
    ))

    abx = await compute_empiric_antibiotic_selection(
        infection_source="urinary",
        severity="septic_shock",
        local_antibiogram={"esbl_prevalence_pct": 22.0,
                                "mrsa_prevalence_pct": 5.0},
        patient_factors={
            "patient_allergies": ["penicillin"],
            "egfr_ml_min": 58.0,
            "risk_factors": ["recent_hospitalisation_90d"],
        },
    )
    run.steps.append(ScenarioStep(
        "compute_empiric_antibiotic_selection",
        f"Top empiric: {abx.top_pick_id}",
        abx,
    ))

    lab = await compute_lab_trend_analysis(
        observations=[
            {"loinc_code": "2160-0", "value": 0.9,
             "datetime": "2026-04-25T10:00:00Z"},
            {"loinc_code": "2160-0", "value": 1.4,
             "datetime": "2026-04-26T10:00:00Z"},
            {"loinc_code": "2160-0", "value": 2.1,
             "datetime": "2026-04-27T10:00:00Z"},
        ],
    )
    run.steps.append(ScenarioStep(
        "compute_lab_trend_analysis",
        f"Trends: {len(lab.trends)} lab(s)",
        lab,
    ))

    counseling = await compute_discharge_counseling(
        medications=[
            {"name": "ceftriaxone", "dose_mg": 2000, "status": "active"},
            {"name": "amlodipine", "dose_mg": 5, "status": "active"},
        ],
        lace_score=11,
        recommendation_action="continued_admission",
    )
    run.steps.append(ScenarioStep(
        "compute_discharge_counseling",
        f"{len(counseling.sections)} sections, "
        f"{counseling.n_red_flags} red flags",
        counseling,
    ))

    phi = await detect_phi(text="Septic UTI, started ceftriaxone 2g IV.")
    run.steps.append(ScenarioStep(
        "detect_phi", f"PHI risk: {phi.risk_level}", phi,
    ))

    run.final_summary = (
        f"NEWS2 {news2.score_total} ({news2.severity_tier}); "
        f"empiric pick honours local ESBL prevalence; "
        f"{len(lab.trends)} lab trend(s) tracked."
    )
    return run


# ─────────────────────────────────────────────────────────────────────
# Scenario 3 -- Polytrauma + MTP
# ─────────────────────────────────────────────────────────────────────


async def scenario_polytrauma_mtp() -> ScenarioRun:
    from mcp_server.tools.trauma_severity_score import (
        compute_trauma_severity_score,
    )
    from mcp_server.tools.massive_transfusion_protocol import (
        compute_massive_transfusion_protocol,
    )
    from mcp_server.tools.imaging_appropriateness import (
        compute_imaging_appropriateness,
    )
    from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
    from mcp_server.tools.detect_phi import detect_phi

    run = ScenarioRun(
        scenario_id="polytrauma_mtp",
        title="Polytrauma + MTP activation",
    )

    iss = await compute_trauma_severity_score(
        injuries=[
            {"body_region": "head", "ais_severity": 2},
            {"body_region": "chest", "ais_severity": 4},
            {"body_region": "abdomen", "ais_severity": 3},
            {"body_region": "extremities", "ais_severity": 2},
        ],
        glasgow_coma_score=11,
        systolic_bp=82,
        respiratory_rate=22,
        has_active_hemorrhage=True,
    )
    run.steps.append(ScenarioStep(
        "compute_trauma_severity_score",
        f"ISS {iss.iss}, RTS {iss.rts:.2f}, "
        f"triage {iss.triage_priority}",
        iss,
    ))

    mtp = await compute_massive_transfusion_protocol(
        penetrating_mechanism=False,
        field_or_arrival_sbp_le_90=True,
        heart_rate_ge_120=True,
        positive_fast_exam=True,
        minutes_since_injury=20,
    )
    run.steps.append(ScenarioStep(
        "compute_massive_transfusion_protocol",
        f"ABC {mtp.abc_score}, MTP = {mtp.mtp_activated}, "
        f"TXA = {mtp.txa_indicated}",
        mtp,
    ))

    img = await compute_imaging_appropriateness(
        clinical_scenario="blunt_abdominal_trauma",
        patient_age=35,
    )
    run.steps.append(ScenarioStep(
        "compute_imaging_appropriateness",
        f"Top modality: {img.top_pick_modality}",
        img,
    ))

    aki = await compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0,
        creatinine_current_mg_dl=1.6,
        urine_output_ml_per_kg_per_hour=0.4,
    )
    run.steps.append(ScenarioStep(
        "compute_aki_kdigo_stage",
        f"AKI stage {aki.aki_stage}",
        aki,
    ))

    phi = await detect_phi(
        text="35yo M MVA polytrauma, ISS 27, MTP activated.",
    )
    run.steps.append(ScenarioStep(
        "detect_phi", f"PHI risk: {phi.risk_level}", phi,
    ))

    run.final_summary = (
        f"ISS {iss.iss}; MTP = {mtp.mtp_activated}; "
        f"AKI stage {aki.aki_stage}."
    )
    return run


# ─────────────────────────────────────────────────────────────────────
# Scenario 4 -- Geriatric polypharmacy med review
# ─────────────────────────────────────────────────────────────────────


async def scenario_geriatric_polypharmacy() -> ScenarioRun:
    from mcp_server.tools.polypharmacy_concerns import (
        detect_polypharmacy_concerns,
    )
    from mcp_server.tools.medication_reconciliation import (
        compute_medication_reconciliation,
    )
    from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse
    from mcp_server.tools.delirium_screening_cam import (
        compute_delirium_screening_cam,
    )
    from mcp_server.tools.discharge_counseling import (
        compute_discharge_counseling,
    )
    from mcp_server.tools.medication_what_if import (
        compute_medication_what_if,
    )
    from mcp_server.tools.detect_phi import detect_phi

    home_meds = [
        {"name": "warfarin", "dose_mg": 5, "status": "active"},
        {"name": "lisinopril", "dose_mg": 10, "status": "active"},
        {"name": "lorazepam", "dose_mg": 0.5, "status": "active"},
        {"name": "oxycodone", "dose_mg": 5, "status": "active"},
        {"name": "diphenhydramine", "dose_mg": 25, "status": "active"},
        {"name": "metformin", "dose_mg": 500, "status": "active"},
    ]
    discharge_meds = [
        {"name": "warfarin", "dose_mg": 5, "status": "active"},
        {"name": "lisinopril", "dose_mg": 10, "status": "active"},
        {"name": "metformin", "dose_mg": 500, "status": "active"},
        {"name": "atorvastatin", "dose_mg": 20, "status": "active"},
    ]

    run = ScenarioRun(
        scenario_id="geriatric_polypharmacy",
        title="Geriatric polypharmacy med review",
    )

    poly = await detect_polypharmacy_concerns(medications=home_meds)
    run.steps.append(ScenarioStep(
        "detect_polypharmacy_concerns",
        f"Severity {poly.polypharmacy_severity}, "
        f"{len(poly.interactions)} DDIs",
        poly,
    ))

    recon = await compute_medication_reconciliation(
        admission_meds=home_meds,
        discharge_meds=discharge_meds,
    )
    run.steps.append(ScenarioStep(
        "compute_medication_reconciliation",
        f"+{len(recon.added)}/-{len(recon.removed)}/Δ"
        f"{len(recon.dose_changed)}",
        recon,
    ))

    morse = await compute_falls_risk_morse(
        history_of_falling_3mo=True,
        secondary_diagnosis_present=True,
        ambulatory_aid="walker",
        has_iv_or_heparin_lock=False,
        gait="weak",
        mental_status="oriented",
        current_medications=[m["name"] for m in home_meds],
    )
    run.steps.append(ScenarioStep(
        "compute_falls_risk_morse",
        f"Morse {morse.score_total} ({morse.risk_tier})",
        morse,
    ))

    cam = await compute_delirium_screening_cam(
        feature1_acute_onset_or_fluctuating=True,
        feature2_inattention=True,
        feature3_disorganized_thinking=False,
        feature4_altered_consciousness=False,
        current_medications=[m["name"] for m in home_meds],
    )
    run.steps.append(ScenarioStep(
        "compute_delirium_screening_cam",
        f"CAM positive = {cam.cam_positive}, "
        f"subtype {cam.delirium_subtype}",
        cam,
    ))

    counseling = await compute_discharge_counseling(
        medications=discharge_meds, lace_score=9,
        recommendation_action="home_with_care",
    )
    run.steps.append(ScenarioStep(
        "compute_discharge_counseling",
        f"{counseling.n_medications_explained} meds explained",
        counseling,
    ))

    what_if = await compute_medication_what_if(
        medications=[m["name"] for m in discharge_meds],
        scenarios=["miss_one_dose", "take_with_alcohol", "stop_abruptly"],
    )
    run.steps.append(ScenarioStep(
        "compute_medication_what_if",
        f"{what_if.n_items} items analysed",
        what_if,
    ))

    phi = await detect_phi(text="85F admitted with falls + AMS.")
    run.steps.append(ScenarioStep(
        "detect_phi", f"PHI risk: {phi.risk_level}", phi,
    ))

    run.final_summary = (
        f"{len(home_meds)}->{len(discharge_meds)} meds; "
        f"Morse {morse.score_total} ({morse.risk_tier}); "
        f"CAM positive = {cam.cam_positive}; "
        f"deprescribed deliriogenic meds."
    )
    return run


# ─────────────────────────────────────────────────────────────────────
# Scenario 5 -- Mental-health crisis + admission decision
# ─────────────────────────────────────────────────────────────────────


async def scenario_mental_health_crisis() -> ScenarioRun:
    from mcp_server.tools.suicide_risk_assessment import (
        compute_suicide_risk_assessment,
    )
    from mcp_server.tools.psychiatric_admission_decision import (
        compute_psychiatric_admission_decision,
    )
    from mcp_server.tools.discharge_counseling import (
        compute_discharge_counseling,
    )
    from mcp_server.tools.translate_discharge_counseling import (
        compute_translate_discharge_counseling,
    )
    from mcp_server.tools.caregiver_summary import (
        compute_caregiver_summary,
    )
    from mcp_server.tools.detect_phi import detect_phi

    run = ScenarioRun(
        scenario_id="mental_health_crisis",
        title="Mental-health crisis + admission decision",
    )

    ssrs = await compute_suicide_risk_assessment(
        ideation_lifetime_level=4,
        ideation_past_30d_level=4,
        behavior_lifetime_attempts=1,
        behavior_past_30d_any=False,
        warning_factors_count=2,
        protective_factors_count=1,
        patient_demographics={
            "race": "white", "sex": "male", "age": 28,
        },
    )
    run.steps.append(ScenarioStep(
        "compute_suicide_risk_assessment",
        f"Risk level {ssrs.risk_level}; "
        f"abstain = {ssrs.abstain_recommended}",
        ssrs,
    ))

    psych = await compute_psychiatric_admission_decision(
        risk_level=ssrs.risk_level,
        danger_to_self=True,
        danger_to_others=False,
        grave_disability=False,
        voluntary_capable=False,
        state_jurisdiction="CA",
    )
    run.steps.append(ScenarioStep(
        "compute_psychiatric_admission_decision",
        f"Disposition: {psych.disposition}",
        psych,
    ))

    counseling = await compute_discharge_counseling(
        medications=[
            {"name": "sertraline", "dose_mg": 50, "status": "active"},
        ],
        lace_score=4,
        recommendation_action="home_with_care",
    )
    run.steps.append(ScenarioStep(
        "compute_discharge_counseling",
        f"{len(counseling.sections)} sections",
        counseling,
    ))

    es = await compute_translate_discharge_counseling(
        counseling=counseling, target_language="es",
    )
    run.steps.append(ScenarioStep(
        "compute_translate_discharge_counseling",
        f"Translated to {es.target_locale}",
        es,
    ))

    cg = await compute_caregiver_summary(counseling=counseling)
    run.steps.append(ScenarioStep(
        "compute_caregiver_summary",
        f"Caregiver summary; {len(cg.red_flag_actions)} red-flag actions",
        cg,
    ))

    phi = await detect_phi(
        text="28yo M C-SSRS tier-4 with plan + intent.",
    )
    run.steps.append(ScenarioStep(
        "detect_phi", f"PHI risk: {phi.risk_level}", phi,
    ))

    run.final_summary = (
        f"C-SSRS risk {ssrs.risk_level}; disposition = "
        f"{psych.disposition}; counseling translated to ES."
    )
    return run


# ─────────────────────────────────────────────────────────────────────
# Public registry + dispatcher
# ─────────────────────────────────────────────────────────────────────


_SCENARIOS: dict[str, Callable[[], Awaitable[ScenarioRun]]] = {
    "acute_stroke_lvo":       scenario_acute_stroke,
    "sepsis_bundle":          scenario_sepsis_bundle,
    "polytrauma_mtp":         scenario_polytrauma_mtp,
    "geriatric_polypharmacy": scenario_geriatric_polypharmacy,
    "mental_health_crisis":   scenario_mental_health_crisis,
}


def list_scenarios() -> list[str]:
    return list(_SCENARIOS.keys())


async def run_scenario(scenario_id: str) -> ScenarioRun:
    if scenario_id not in _SCENARIOS:
        raise KeyError(
            f"Unknown scenario {scenario_id!r}. "
            f"Available: {list(_SCENARIOS)}"
        )
    return await _SCENARIOS[scenario_id]()


async def run_all_scenarios() -> list[ScenarioRun]:
    return [await fn() for fn in _SCENARIOS.values()]
