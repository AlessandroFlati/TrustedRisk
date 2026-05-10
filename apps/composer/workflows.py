"""TrustedRisk Care Engine — workflow registry.

The Care Engine is a three-layer composition surface that turns a
free-form clinical prompt into a chained execution across the
17-specialist federation:

    Layer 1: BASE workflows (58, hand-crafted)
        Tool-level recipes drawn from clinical protocol. Each step
        names a specialist + tool; `${input.x}` flows fields from the
        master input, `${steps.<id>.output.<field>}` chains downstream
        steps on upstream outputs. Mode is per-workflow:

          - parallel-by-protocol (sepsis bundle, intake battery...)
            — concurrent fan-out, no data dependency between steps.
          - data-dependent chain (PA pipeline, AKI staging -> dialysis
            decision, RECIST -> dose adjustment...) — every chain ref
            sits where the protocol calls for one.
          - mixed (intake parallel + chained downstream consult letter).

    Layer 2: MACRO workflows (50, hand-crafted)
        Care arcs that compose 2-4 base workflows in sequence. Each
        hop is a `sub_workflow_id` reference, executed recursively by
        the orchestrator. Macro chains are implicit: hop_2 sees the
        same patient hop_1 just operated on. Examples:
        complete_chf_admission, peri-op arc, surgical_full,
        pa_with_preemptive_appeal.

    Layer 3: PARAMETRIC variants (149, factory-generated)
        A clone of a base workflow with id-suffix + an input overlay
        targeting one or more axes (payer, age band, severity,
        specialty, disposition, outreach channel, ...). Each variant
        represents a clinically-distinct decision path the workflow
        actually takes — same engine, different context.

Total: 257 workflows; 146 chained (56%). The orchestrator's
free-form-prompt dispatch picks the right layer based on the user's
phrasing, then runs the chosen workflow against the in-process
backend OR the A2A HTTP backend (real specialist sub-agent calls)
depending on `TRUSTEDRISK_COMPOSER_BACKEND`.
"""

from __future__ import annotations

from .orchestrator import Workflow, WorkflowStep

# Tool imports — kept inline so each workflow's needs are visible
from mcp_server.tools.acs_disposition_decision import (
    compute_acs_disposition_decision,
)
from mcp_server.tools.admission_hnp_draft import compute_admission_hnp_draft
from mcp_server.tools.admission_triage import compute_admission_triage
from mcp_server.tools.care_gap_detector import compute_care_gap_detector
from mcp_server.tools.caregiver_handoff import compute_caregiver_handoff
from mcp_server.tools.clinical_deterioration_score import (
    compute_clinical_deterioration_score,
)
from mcp_server.tools.consult_letter_draft import compute_consult_letter_draft
from mcp_server.tools.contrast_safety_check import compute_contrast_safety_check
from mcp_server.tools.differential_diagnosis_ranker import (
    compute_differential_diagnosis_ranker,
)
from mcp_server.tools.discharge_counseling import compute_discharge_counseling
from mcp_server.tools.discharge_qa import compute_discharge_qa
from mcp_server.tools.discharge_summary_draft import (
    compute_discharge_summary_draft,
)
from mcp_server.tools.empiric_antibiotic_selection import (
    compute_empiric_antibiotic_selection,
)
from mcp_server.tools.heart_score import compute_heart_score
from mcp_server.tools.medication_reconciliation import (
    compute_medication_reconciliation,
)
from mcp_server.tools.medication_what_if import compute_medication_what_if
from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
from mcp_server.tools.fairness_audit import compute_fairness_audit
from mcp_server.tools.readmission_risk import compute_readmission_risk
from mcp_server.tools.stroke_severity import compute_stroke_severity
from mcp_server.tools.stroke_thrombolysis_eligibility import (
    compute_stroke_thrombolysis_eligibility,
)
from mcp_server.tools.trauma_severity_score import compute_trauma_severity_score
from mcp_server.tools.massive_transfusion_protocol import (
    compute_massive_transfusion_protocol,
)
from mcp_server.tools.dka_severity import compute_dka_severity
from mcp_server.tools.inpatient_glycemic_control import (
    compute_inpatient_glycemic_control,
)
from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
from mcp_server.tools.dialysis_initiation_decision import (
    compute_dialysis_initiation_decision,
)
from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse
from mcp_server.tools.delirium_screening_cam import (
    compute_delirium_screening_cam,
)
from mcp_server.tools.peri_op_risk import compute_rcri_cardiac_risk
from mcp_server.tools.progress_note_draft import compute_progress_note_draft
from mcp_server.tools.patient_faq import compute_patient_faq
from mcp_server.tools.caregiver_summary import compute_caregiver_summary
from mcp_server.tools.expected_value_of_intervention import (
    compute_expected_value_of_intervention,
)
from mcp_server.tools.pgx import (
    compute_pgx_dose_adjustment,
    compute_pgx_drug_alternatives,
    compute_pgx_eligibility_check,
)
from mcp_server.tools.antibiotic_de_escalation import (
    compute_antibiotic_de_escalation,
)
from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing
from mcp_server.tools.peri_op_risk import (
    compute_ariscat_pulmonary_risk,
    compute_caprini_vte_risk,
)
from mcp_server.tools.chart_intelligence import compute_clinical_ner
from mcp_server.tools.auto_coding import (
    compute_coding_audit,
    compute_cpt_suggest,
    compute_icd10_suggest,
)
from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
from mcp_server.tools.pa_payer_rules_match import (
    compute_pa_payer_rules_match,
)
from mcp_server.tools.pa_letter_draft import compute_pa_letter_draft
from mcp_server.tools.pa_appeal_likelihood import (
    compute_pa_appeal_likelihood,
)
from mcp_server.tools.insurance_appeals import (
    compute_appeal_escalation_path,
    compute_appeal_letter_draft,
    compute_denial_letter_parse,
)
from mcp_server.tools.quality_stars import (
    compute_care_gap_priority_ranking,
    compute_quality_measures_aggregate,
    compute_stars_rating_forecast,
)
from mcp_server.tools.population_health import (
    compute_outbreak_heatmap,
    compute_syndromic_surveillance,
    compute_vaccine_reminder_cohort,
)
from mcp_server.tools.preadmit_triage import (
    compute_symptom_red_flag_check,
    compute_when_to_seek_care,
)
from mcp_server.tools.suicide_risk_assessment import (
    compute_suicide_risk_assessment,
)
from mcp_server.tools.psychiatric_admission_decision import (
    compute_psychiatric_admission_decision,
)
from mcp_server.tools.pediatric_early_warning import (
    compute_pediatric_early_warning,
)
from mcp_server.tools.preeclampsia_assessment import (
    compute_preeclampsia_assessment,
)
from mcp_server.tools.maternal_early_warning import (
    compute_maternal_early_warning,
)
from mcp_server.tools.chemo_dose_adjustment import compute_chemo_dose_adjustment
from mcp_server.tools.oncology_treatment_response import (
    compute_oncology_treatment_response,
)
from mcp_server.tools.treatment_selection import compute_treatment_selection


# ─────────────────────────────────────────────────────────────────────
# 1. CHF admission — `acute` → `evidence` → `discharge` → `scribe`
# ─────────────────────────────────────────────────────────────────────

CHF_ADMISSION = Workflow(
    id="chf_admission",
    title="CHF inpatient admission workflow",
    description=(
        "Adult presents with worsening shortness of breath. "
        "Composes ED triage + deterioration score + DDx + readmission "
        "risk + polypharmacy check + admission H&P drafting across 4 "
        "specialists (acute, evidence, discharge, scribe)."
    ),
    required_inputs=[
        "chief_complaint", "vital_signs", "age", "patient_id",
        "fhir_bundle", "medications",
    ],
    steps=[
        WorkflowStep(
            id="triage",
            specialist="trustedrisk-acute",
            tool_name="compute_admission_triage",
            callable=compute_admission_triage,
            description="ED triage — ESI level + disposition",
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "vital_signs": "${input.vital_signs}",
                "age": "${input.age}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="deterioration",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            description="NEWS2 + trend",
            inputs={
                "vital_signs": "${input.vital_signs}",
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="ddx",
            specialist="trustedrisk-evidence",
            tool_name="compute_differential_diagnosis_ranker",
            callable=compute_differential_diagnosis_ranker,
            description="Grounded DDx ranker",
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "structured_features": "${input.vital_signs}",
            },
        ),
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            description="DDI scan on home meds",
            inputs={"medications": "${input.medications}"},
        ),
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            description="Admission History and Physical draft",
            inputs={
                "patient_reference": "${input.patient_id}",
                "chief_complaint": "${input.chief_complaint}",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# 2. Sepsis workup — `acute` → `evidence` → `antimicrobial` → `scribe`
# ─────────────────────────────────────────────────────────────────────

SEPSIS_WORKUP = Workflow(
    id="sepsis_workup",
    title="Suspected sepsis ED workup",
    description=(
        "Febrile patient with hypotension; ED triage + qSOFA-like "
        "deterioration check + DDx + empiric antibiotic selection "
        "(with local antibiogram) + consult letter to ID. Mixed mode: "
        "triage / deterioration / ddx / antibiotic run in parallel "
        "(protocol-independent intake), then `consult` chains on "
        "antibiotic.rationale so the ID specialist receives the empiric "
        "choice that was made and not just the chief complaint."
    ),
    required_inputs=[
        "chief_complaint", "vital_signs", "age", "patient_id",
        "infection_source", "patient_factors", "fhir_bundle",
    ],
    steps=[
        WorkflowStep(
            id="triage",
            specialist="trustedrisk-acute",
            tool_name="compute_admission_triage",
            callable=compute_admission_triage,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "vital_signs": "${input.vital_signs}",
                "age": "${input.age}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="deterioration",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": "${input.vital_signs}",
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="ddx",
            specialist="trustedrisk-evidence",
            tool_name="compute_differential_diagnosis_ranker",
            callable=compute_differential_diagnosis_ranker,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "structured_features": "${input.vital_signs}",
            },
        ),
        WorkflowStep(
            id="antibiotic",
            specialist="trustedrisk-discharge",   # antimicrobial bundle in discharge specialist
            tool_name="compute_empiric_antibiotic_selection",
            callable=compute_empiric_antibiotic_selection,
            description="Empiric antibiotic with local antibiogram",
            inputs={
                "infection_source": "${input.infection_source}",
                "severity": "septic",
                "patient_factors": "${input.patient_factors}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            description="ID consult letter (chains on antibiotic decision)",
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "ED hospitalist",
                # Chain: pull the empiric antibiotic rationale into the
                # consultation reason so the ID specialist sees the
                # decision that was made, not a generic restatement.
                "consultation_reason":
                    "${steps.antibiotic.output.rationale}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "infectious_diseases",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# 3. Discharge planning — `discharge` → `scribe` → `patient`
# ─────────────────────────────────────────────────────────────────────

DISCHARGE_PLANNING = Workflow(
    id="discharge_planning",
    title="End-to-end discharge planning",
    description=(
        "Compose readmission risk + medication reconciliation + "
        "discharge counseling + discharge-summary drafting + caregiver "
        "hand-off across 3 specialists."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "admission_meds", "discharge_meds",
        "discharge_action", "lace_score", "patient_demographics",
        "hospital_course_text", "followup_plan", "patient_instructions",
    ],
    steps=[
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,   # requires bound SHARP FHIR context
        ),
        WorkflowStep(
            id="fairness",
            specialist="trustedrisk-acute",
            tool_name="compute_fairness_audit",
            callable=compute_fairness_audit,
            inputs={
                "risk": "${steps.risk.output}",
                "patient_demographics": "${input.patient_demographics}",
            },
            optional=True,   # depends on risk; skips cleanly when risk skipped
        ),
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,   # tolerates absent FHIR context
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.discharge_meds}",
                "lace_score": "${input.lace_score}",
                "recommendation_action": "${input.discharge_action}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="summary",
            specialist="trustedrisk-scribe",
            tool_name="compute_discharge_summary_draft",
            callable=compute_discharge_summary_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "discharge_disposition": "${input.discharge_action}",
                "hospital_course_text": "${input.hospital_course_text}",
                "followup_plan": "${input.followup_plan}",
                "patient_instructions": "${input.patient_instructions}",
            },
            # The scribe tool fail-fasts when clinician-supplied narrative
            # inputs (hospital_course / followup / patient_instructions) are
            # absent. Keep this step optional so a chat-only invocation that
            # cannot supply those still produces the upstream risk +
            # fairness + reconciliation steps; the summary itself returns
            # abstain_recommended=True with an explicit list of missing
            # inputs the caller must add to a follow-up call.
            optional=True,
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            description="Caregiver hand-off package",
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {
                        "action": "${input.discharge_action}",
                    },
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
            optional=True,
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# 4. Outpatient med review — `discharge` → `patient`
# ─────────────────────────────────────────────────────────────────────

OUTPATIENT_MED_REVIEW = Workflow(
    id="outpatient_med_review",
    title="Outpatient medication review",
    description=(
        "Geriatric outpatient med-review (LACE = 0). Polypharmacy check "
        "+ care-gap detection + patient what-if scenarios."
    ),
    required_inputs=[
        "patient_id", "medications", "fhir_bundle", "patient_age",
        "patient_sex",
    ],
    steps=[
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.medications}"},
        ),
        WorkflowStep(
            id="gaps",
            specialist="trustedrisk-discharge",
            tool_name="compute_care_gap_detector",
            callable=compute_care_gap_detector,
            inputs={
                "fhir_bundle": "${input.fhir_bundle}",
                "patient_age": "${input.patient_age}",
                "patient_sex": "${input.patient_sex}",
            },
        ),
        WorkflowStep(
            id="what_if",
            specialist="trustedrisk-patient",
            tool_name="compute_medication_what_if",
            callable=compute_medication_what_if,
            inputs={
                "medications": "${input.medications}",
                "scenarios": [
                    "miss_one_dose", "take_with_alcohol",
                    "stop_abruptly", "take_with_otc_nsaid",
                ],
                "patient_reference": "${input.patient_id}",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# 5. Stroke alert — `acute` → `imaging` → `scribe`
# ─────────────────────────────────────────────────────────────────────

STROKE_ALERT = Workflow(
    id="stroke_alert",
    title="Acute stroke reperfusion alert",
    description=(
        "70 y M with focal deficit. NIHSS scoring + tPA / EVT "
        "eligibility + contrast safety pre-CT + admission consult letter."
    ),
    required_inputs=[
        "patient_id", "nihss_item_scores", "last_known_well_minutes_ago",
        "egfr_ml_min", "fhir_bundle",
    ],
    steps=[
        WorkflowStep(
            id="severity",
            specialist="trustedrisk-acute",
            tool_name="compute_stroke_severity",
            callable=compute_stroke_severity,
            inputs={
                "item_scores": "${input.nihss_item_scores}",
                "last_known_well_minutes_ago":
                    "${input.last_known_well_minutes_ago}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="thrombolysis",
            specialist="trustedrisk-acute",
            tool_name="compute_stroke_thrombolysis_eligibility",
            callable=compute_stroke_thrombolysis_eligibility,
            inputs={
                "last_known_well_minutes_ago":
                    "${input.last_known_well_minutes_ago}",
                "nihss_total": "${steps.severity.output.score_total}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="contrast",
            specialist="trustedrisk-acute",
            tool_name="compute_contrast_safety_check",
            callable=compute_contrast_safety_check,
            inputs={
                "contrast_type": "iodinated_iv",
                "egfr_ml_min": "${input.egfr_ml_min}",
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "ED hospitalist",
                "consultation_reason":
                    "Acute stroke alert — tPA/EVT eligibility decision",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "neurology",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical A — ED acute care (additions A4..A8)
# ─────────────────────────────────────────────────────────────────────

TRAUMA_RESUS_DECISION = Workflow(
    id="trauma_resus_decision",
    title="Polytrauma resuscitation decision",
    description=(
        "Polytrauma after high-energy mechanism. Composes ISS+T-RTS "
        "severity, ABC-score MTP gating, IV-contrast safety for the "
        "trauma CT, and a trauma-surgery consult letter."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "injuries",
        "glasgow_coma_score", "systolic_bp", "respiratory_rate",
    ],
    steps=[
        WorkflowStep(
            id="severity",
            specialist="trustedrisk-acute",
            tool_name="compute_trauma_severity_score",
            callable=compute_trauma_severity_score,
            inputs={
                "injuries": "${input.injuries}",
                "glasgow_coma_score": "${input.glasgow_coma_score}",
                "systolic_bp": "${input.systolic_bp}",
                "respiratory_rate": "${input.respiratory_rate}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="mtp",
            specialist="trustedrisk-acute",
            tool_name="compute_massive_transfusion_protocol",
            callable=compute_massive_transfusion_protocol,
            inputs={
                "field_or_arrival_sbp_le_90": True,
                "heart_rate_ge_120": True,
                "positive_fast_exam": True,
                "estimated_blood_loss_ml": 1500,
                "minutes_since_injury": 60,
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="contrast",
            specialist="trustedrisk-acute",
            tool_name="compute_contrast_safety_check",
            callable=compute_contrast_safety_check,
            inputs={
                "contrast_type": "iodinated_iv",
                "egfr_ml_min": 65.0,
            },
            optional=True,
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            description="Trauma-surgery consult (chains on severity.rationale)",
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "ED trauma team",
                # Chain: forward the ISS+RTS rationale to trauma surgery
                # so the consult sees the same severity stratification
                # the activation was based on.
                "consultation_reason":
                    "${steps.severity.output.rationale}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "trauma_surgery",
            },
        ),
    ],
)


ACUTE_CHEST_PAIN = Workflow(
    id="acute_chest_pain",
    title="Acute chest pain workup",
    description=(
        "Adult with acute chest pain. Composes HEART risk band, "
        "ACS disposition decision (cath / observation / discharge), "
        "and an admission HnP draft."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "age",
        "history_descriptor", "ecg_descriptor",
        "risk_factors_count", "troponin_times_uln",
    ],
    steps=[
        WorkflowStep(
            id="heart",
            specialist="trustedrisk-acute",
            tool_name="compute_heart_score",
            callable=compute_heart_score,
            inputs={
                "history_descriptor": "${input.history_descriptor}",
                "ecg_descriptor": "${input.ecg_descriptor}",
                "age": "${input.age}",
                "risk_factors_count": "${input.risk_factors_count}",
                "troponin_times_uln": "${input.troponin_times_uln}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="disposition",
            specialist="trustedrisk-acute",
            tool_name="compute_acs_disposition_decision",
            callable=compute_acs_disposition_decision,
            inputs={
                "heart_score_total": "${steps.heart.output.total_score}",
                "has_stemi": False,
                "has_dynamic_troponin": False,
                "ongoing_chest_pain": True,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "chief_complaint": "acute chest pain",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


DKA_MANAGEMENT = Workflow(
    id="dka_management",
    title="Diabetic ketoacidosis management",
    description=(
        "Adult with DKA presentation. Composes ADA/ISPAD severity "
        "tier, inpatient glycemic-control transition plan, and an "
        "admission HnP draft."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "ph",
        "bicarbonate_meq_l", "glucose_mg_dl", "weight_kg",
    ],
    steps=[
        WorkflowStep(
            id="severity",
            specialist="trustedrisk-acute",
            tool_name="compute_dka_severity",
            callable=compute_dka_severity,
            inputs={
                "ph": "${input.ph}",
                "bicarbonate_meq_l": "${input.bicarbonate_meq_l}",
                "glucose_mg_dl": "${input.glucose_mg_dl}",
                "ketones_present": True,
                "weight_kg": "${input.weight_kg}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="glycemic",
            specialist="trustedrisk-acute",
            tool_name="compute_inpatient_glycemic_control",
            callable=compute_inpatient_glycemic_control,
            description="Inpatient glycemic control (chains ICU flag from severity)",
            inputs={
                # Chain: ICU disposition is determined by DKA severity tier;
                # severe DKA mandates ICU per ADA, mild can be floor.
                "is_icu": "${steps.severity.output.icu_admission_indicated}",
                "average_glucose_24h": 320.0,
                "n_severe_hyperglycemic_episodes_24h": 3,
                "current_regimen": "iv_insulin_infusion",
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "chief_complaint": "diabetic ketoacidosis",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


AKI_WORKUP = Workflow(
    id="aki_workup",
    title="Acute kidney injury workup",
    description=(
        "Inpatient with rising creatinine. Composes KDIGO AKI stage, "
        "dialysis-initiation decision, contrast safety re-check for "
        "any planned imaging, and a nephrology consult letter."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "creatinine_baseline_mg_dl", "creatinine_current_mg_dl",
    ],
    steps=[
        WorkflowStep(
            id="staging",
            specialist="trustedrisk-acute",
            tool_name="compute_aki_kdigo_stage",
            callable=compute_aki_kdigo_stage,
            inputs={
                "creatinine_baseline_mg_dl":
                    "${input.creatinine_baseline_mg_dl}",
                "creatinine_current_mg_dl":
                    "${input.creatinine_current_mg_dl}",
                "urine_output_ml_per_kg_per_hour": 0.4,
                "nephrotoxic_medications_present": True,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="dialysis",
            specialist="trustedrisk-acute",
            tool_name="compute_dialysis_initiation_decision",
            callable=compute_dialysis_initiation_decision,
            inputs={
                "aki_stage": "${steps.staging.output.aki_stage}",
                "ph": 7.21, "bicarbonate_meq_l": 14.0,
                "potassium_meq_l": 6.2,
                "refractory_hyperkalemia": False,
                "volume_overload_refractory": True,
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="contrast",
            specialist="trustedrisk-acute",
            tool_name="compute_contrast_safety_check",
            callable=compute_contrast_safety_check,
            inputs={
                "contrast_type": "iodinated_iv",
                "egfr_ml_min": 24.0,
                "on_metformin": True,
            },
            optional=True,
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Hospitalist",
                "consultation_reason":
                    "Stage-3 AKI with refractory volume overload",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "nephrology",
            },
        ),
    ],
)


RESPIRATORY_FAILURE_WORKUP = Workflow(
    id="respiratory_failure_workup",
    title="Acute respiratory failure workup",
    description=(
        "Adult with hypoxemic / hypercapnic respiratory failure. "
        "Composes ED triage + NEWS2 deterioration trend + grounded "
        "DDx + admission HnP draft. Chained: the H&P's chief-complaint "
        "section inherits the DDx top suspect so the documented "
        "assessment matches the diagnostic reasoning."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "chief_complaint", "vital_signs", "age",
    ],
    steps=[
        WorkflowStep(
            id="triage",
            specialist="trustedrisk-acute",
            tool_name="compute_admission_triage",
            callable=compute_admission_triage,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "vital_signs": "${input.vital_signs}",
                "age": "${input.age}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="deterioration",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": [
                    {"type": "respiratory_rate", "value": 32},
                    {"type": "spo2", "value": 86},
                    {"type": "heart_rate", "value": 130},
                    {"type": "supplemental_oxygen", "value": True},
                    {"type": "consciousness", "value": "alert"},
                ],
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="ddx",
            specialist="trustedrisk-evidence",
            tool_name="compute_differential_diagnosis_ranker",
            callable=compute_differential_diagnosis_ranker,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "structured_features": "${input.vital_signs}",
            },
        ),
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            description="Admission H&P (chains on DDx free-text summary)",
            inputs={
                "patient_reference": "${input.patient_id}",
                # Chain: prefer the DDx tool's summarised chief complaint
                # over the raw user input -- it's the assessment the
                # clinician saw and reduces "chart says X, note says Y"
                # divergence.
                "chief_complaint": "${steps.ddx.output.free_text_summary}",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical B — hospital course (B9..B13)
# ─────────────────────────────────────────────────────────────────────

INPATIENT_DETERIORATION_RESPONSE = Workflow(
    id="inpatient_deterioration_response",
    title="Inpatient deterioration response",
    description=(
        "Floor patient triggering NEWS2 on the bedside monitor. "
        "Composes deterioration score + grounded DDx + consult letter "
        "to the rapid-response team."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "vital_signs_obs",
        "chief_complaint",
    ],
    steps=[
        WorkflowStep(
            id="news2",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": "${input.vital_signs_obs}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="ddx",
            specialist="trustedrisk-evidence",
            tool_name="compute_differential_diagnosis_ranker",
            callable=compute_differential_diagnosis_ranker,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "structured_features": {},
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Floor RN",
                "consultation_reason":
                    "Bedside NEWS2 escalation, request for rapid-response evaluation",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "internal_medicine",
            },
        ),
    ],
)


ICU_STEP_DOWN = Workflow(
    id="icu_step_down",
    title="ICU step-down readiness review",
    description=(
        "Patient has stabilised in ICU and is candidate for floor "
        "transfer. Composes deterioration score, post-transfer "
        "readmission risk, and a caregiver hand-off package."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "vital_signs_obs",
    ],
    steps=[
        WorkflowStep(
            id="news2",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": "${input.vital_signs_obs}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {"action": "step_down_to_floor"},
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


INPATIENT_GLYCEMIC_CONTROL_WORKFLOW = Workflow(
    id="inpatient_glycemic_control_workflow",
    title="Inpatient glycemic control review",
    description=(
        "Inpatient with sustained hyperglycemia. Composes ADA-2024 "
        "glycemic-control adjustment plan and a progress-note draft "
        "summarising the change."
    ),
    required_inputs=["patient_id", "fhir_bundle"],
    steps=[
        WorkflowStep(
            id="glycemic",
            specialist="trustedrisk-acute",
            tool_name="compute_inpatient_glycemic_control",
            callable=compute_inpatient_glycemic_control,
            inputs={
                "is_icu": False,
                "average_glucose_24h": 245.0,
                "n_severe_hyperglycemic_episodes_24h": 2,
                "current_regimen": "basal_bolus",
                "current_basal_total_units": 28.0,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="note",
            specialist="trustedrisk-scribe",
            tool_name="compute_progress_note_draft",
            callable=compute_progress_note_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "subjective_text":
                    "Hyperglycemia trend on basal-bolus regimen. "
                    "Adjusting per ADA 2024 inpatient targets.",
            },
        ),
    ],
)


PERIOP_COMPLICATIONS_RESPONSE = Workflow(
    id="periop_complications_response",
    title="Peri-operative complications response",
    description=(
        "Post-op patient with developing complication. Composes "
        "Lee Revised Cardiac Risk Index, contrast safety re-check "
        "for any planned imaging, and a progress-note draft."
    ),
    required_inputs=["patient_id", "fhir_bundle"],
    steps=[
        WorkflowStep(
            id="cardiac_risk",
            specialist="trustedrisk-acute",
            tool_name="compute_rcri_cardiac_risk",
            callable=compute_rcri_cardiac_risk,
            inputs={
                "high_risk_surgery": True,
                "history_ischemic_heart_disease": True,
                "history_congestive_heart_failure": True,
                "creatinine_gt_2_mg_dl": False,
            },
        ),
        WorkflowStep(
            id="contrast",
            specialist="trustedrisk-acute",
            tool_name="compute_contrast_safety_check",
            callable=compute_contrast_safety_check,
            inputs={
                "contrast_type": "iodinated_iv",
                "egfr_ml_min": 48.0,
            },
        ),
        WorkflowStep(
            id="note",
            specialist="trustedrisk-scribe",
            tool_name="compute_progress_note_draft",
            callable=compute_progress_note_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "subjective_text":
                    "Post-op day 1: developing chest discomfort and "
                    "dyspnea, ECG shows new ST changes. Workup in progress.",
            },
        ),
    ],
)


INPATIENT_FALLS_INTERVENTION = Workflow(
    id="inpatient_falls_intervention",
    title="Inpatient falls intervention",
    description=(
        "Inpatient with concerning gait or cognition. Composes "
        "Morse Falls Scale, CAM delirium screen, polypharmacy "
        "scan, and a counseling handout for the family."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="falls",
            specialist="trustedrisk-acute",
            tool_name="compute_falls_risk_morse",
            callable=compute_falls_risk_morse,
            inputs={
                "history_of_falling_3mo": True,
                "secondary_diagnosis_present": True,
                "ambulatory_aid": "walker",
                "has_iv_or_heparin_lock": True,
                "gait": "weak",
                "mental_status": "forgets_limitations",
                "current_medications": "${input.current_medications}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="delirium",
            specialist="trustedrisk-acute",
            tool_name="compute_delirium_screening_cam",
            callable=compute_delirium_screening_cam,
            inputs={
                "feature1_acute_onset_or_fluctuating": True,
                "feature2_inattention": True,
                "feature3_disorganized_thinking": True,
                "feature4_altered_consciousness": False,
                "motor_subtype": "hypoactive",
                "current_medications": "${input.current_medications}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 11,
                "recommendation_action": "continued_admission",
            },
            optional=True,
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical C — discharge & continuity (C15..C19)
# ─────────────────────────────────────────────────────────────────────

POST_DISCHARGE_FOLLOWUP = Workflow(
    id="post_discharge_followup",
    title="Post-discharge phone followup",
    description=(
        "Day 3-7 post-discharge call. Composes readmission-risk "
        "re-check, medication reconciliation against the patient's "
        "actual home regimen, and a structured patient FAQ for the "
        "questions the patient asked."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "admission_meds",
        "discharge_meds", "patient_question",
    ],
    steps=[
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,
        ),
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            inputs={"question": "${input.patient_question}"},
            # FAQ requires a clinician- or patient-authored question
            # (real-time), not a value derivable from the FHIR bundle.
            # Marked optional so the workflow can complete without it
            # when no question is supplied (Q3 + Q4).
            optional=True,
        ),
    ],
)


CAREGIVER_HANDOFF_PREP = Workflow(
    id="caregiver_handoff_prep",
    title="Caregiver hand-off package",
    description=(
        "Discharge with primary caregiver. Composes readmission "
        "risk, structured caregiver hand-off, and patient-language "
        "counseling."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "discharge_meds",
        "discharge_action",
    ],
    steps=[
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {
                        "action": "${input.discharge_action}",
                    },
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.discharge_meds}",
                "lace_score": 9,
                "recommendation_action": "${input.discharge_action}",
            },
            optional=True,
        ),
    ],
)


TRANSITIONAL_CARE_MANAGEMENT = Workflow(
    id="transitional_care_management",
    title="Transitional care management visit",
    description=(
        "TCM 7- or 14-day post-discharge visit (CMS 99495/99496). "
        "Composes care-gap detection, medication reconciliation, "
        "and patient-language counseling."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "patient_age", "patient_sex",
        "admission_meds", "discharge_meds",
    ],
    steps=[
        WorkflowStep(
            id="gaps",
            specialist="trustedrisk-discharge",
            tool_name="compute_care_gap_detector",
            callable=compute_care_gap_detector,
            inputs={
                "fhir_bundle": "${input.fhir_bundle}",
                "patient_age": "${input.patient_age}",
                "patient_sex": "${input.patient_sex}",
            },
        ),
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.discharge_meds}",
                "lace_score": 8,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


CHRONIC_DISEASE_FOLLOWUP = Workflow(
    id="chronic_disease_followup",
    title="Chronic disease tail followup",
    description=(
        "Routine outpatient followup on a chronic disease cohort "
        "patient (CHF / DM / COPD / CKD / HTN). Composes readmission "
        "risk re-check, care-gap scan, and counseling."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "patient_age", "patient_sex",
        "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,
        ),
        WorkflowStep(
            id="gaps",
            specialist="trustedrisk-discharge",
            tool_name="compute_care_gap_detector",
            callable=compute_care_gap_detector,
            inputs={
                "fhir_bundle": "${input.fhir_bundle}",
                "patient_age": "${input.patient_age}",
                "patient_sex": "${input.patient_sex}",
            },
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 7,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


PALLIATIVE_TRANSITION = Workflow(
    id="palliative_transition",
    title="Palliative care transition",
    description=(
        "Patient transitioning to palliative care. Composes patient "
        "counseling, caregiver hand-off, and a cost-effectiveness "
        "review of the proposed regimen so the family can make an "
        "informed shared-decision."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 12,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {"action": "home_with_care"},
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="cost",
            specialist="trustedrisk-population",
            tool_name="compute_expected_value_of_intervention",
            callable=compute_expected_value_of_intervention,
            inputs={
                "intervention": {
                    "name": "Palliative home-care bundle",
                    "absolute_risk_reduction": 0.0,
                    "cost_per_patient_usd": 4500.0,
                    "qaly_gain_per_patient": 0.08,
                },
                "baseline_event_probability": 0.65,
                "cohort_size": 100,
                "wtp_threshold_per_qaly_usd": 100000.0,
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical D — medication management (D21..D26)
# ─────────────────────────────────────────────────────────────────────

PGX_PRESCRIBING_CHECK = Workflow(
    id="pgx_prescribing_check",
    title="Pharmacogenomic prescribing check",
    description=(
        "Pre-prescription PGx safety check. Composes eligibility "
        "(should we even genotype?), drug-alternatives lookup, and "
        "dose-adjustment recommendation when the genotype is known."
    ),
    required_inputs=[
        "patient_id", "requested_drug", "genotypes",
    ],
    steps=[
        WorkflowStep(
            id="eligibility",
            specialist="trustedrisk-pgx",
            tool_name="compute_pgx_eligibility_check",
            callable=compute_pgx_eligibility_check,
            inputs={
                "requested_test": "${input.requested_drug}",
                "medications_in_consideration": [
                    "${input.requested_drug}",
                ],
            },
        ),
        WorkflowStep(
            id="alternatives",
            specialist="trustedrisk-pgx",
            tool_name="compute_pgx_drug_alternatives",
            callable=compute_pgx_drug_alternatives,
            inputs={
                "requested_drug": "${input.requested_drug}",
                "genotypes": "${input.genotypes}",
                "patient_reference": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="dose",
            specialist="trustedrisk-pgx",
            tool_name="compute_pgx_dose_adjustment",
            callable=compute_pgx_dose_adjustment,
            inputs={
                "medications": ["${input.requested_drug}"],
                "genotypes": "${input.genotypes}",
                "patient_reference": "${input.patient_id}",
            },
        ),
    ],
)


DDI_AUDIT = Workflow(
    id="ddi_audit",
    title="Drug-drug interaction audit",
    description=(
        "Standalone DDI audit on the active medication list. "
        "Composes polypharmacy concerns, medication reconciliation "
        "against the chart, and a counseling note flagging the "
        "highest-severity interactions."
    ),
    required_inputs=[
        "patient_id", "current_medications",
        "admission_meds", "discharge_meds",
    ],
    steps=[
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 6,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


ANTICOAGULANT_REVIEW = Workflow(
    id="anticoagulant_review",
    title="Anticoagulant safety review",
    description=(
        "Patient on warfarin / DOAC with concerning recent labs "
        "(INR swing, bleeding signs, missed dose). Composes "
        "polypharmacy DDI scan focused on anticoagulant interactions, "
        "medication reconciliation, and patient counseling. Chained: "
        "the patient counseling references the discrepancy concerns "
        "surfaced by med-recon so the patient hears about the same "
        "drug change the chart records."
    ),
    required_inputs=[
        "patient_id", "current_medications",
        "admission_meds", "discharge_meds",
    ],
    steps=[
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            description="Patient counseling (chains on discharge meds)",
            inputs={
                "patient_id": "${input.patient_id}",
                # Chain: counsel against the exact discharge med list,
                # not the on-hand current_medications. If med-recon adds
                # or changes a drug the patient should hear that, not
                # the pre-admission list.
                "medications": "${input.discharge_meds}",
                "lace_score": 8,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


ANTIBIOTIC_STEWARDSHIP = Workflow(
    id="antibiotic_stewardship",
    title="Antibiotic stewardship round",
    description=(
        "Day-3 antibiotic review on an inpatient: composes empiric "
        "regimen recommendation, de-escalation plan based on "
        "culture sensitivities, and a consult letter to ID."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "infection_source",
        "patient_factors", "current_regimen",
    ],
    steps=[
        WorkflowStep(
            id="empiric",
            specialist="trustedrisk-discharge",
            tool_name="compute_empiric_antibiotic_selection",
            callable=compute_empiric_antibiotic_selection,
            inputs={
                "infection_source": "${input.infection_source}",
                "severity": "moderate",
                "patient_factors": "${input.patient_factors}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="deescalation",
            specialist="trustedrisk-discharge",
            tool_name="compute_antibiotic_de_escalation",
            callable=compute_antibiotic_de_escalation,
            inputs={
                "current_regimen": "${input.current_regimen}",
                "pathogen": "Escherichia coli",
                "susceptibility": {
                    "ceftriaxone": "S", "ciprofloxacin": "R",
                    "meropenem": "S",
                },
                "days_on_therapy": 3,
                "total_planned_duration_days": 7,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            description="ID consult letter (chains on de-escalation rationale)",
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Inpatient hospitalist",
                # Chain: forward the de-escalation rationale (which
                # narrow regimen to use, why, and the IV-to-PO eligibility)
                # so the ID consult sees the actual stewardship decision.
                "consultation_reason":
                    "${steps.deescalation.output.rationale}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "infectious_diseases",
            },
        ),
    ],
)


OPIOID_SAFETY_REVIEW = Workflow(
    id="opioid_safety_review",
    title="Opioid safety review",
    description=(
        "Patient on chronic opioid therapy. Composes polypharmacy "
        "DDI scan (benzo + opioid + sedative interactions), DDI "
        "details, and a patient-language counseling handout focused "
        "on overdose / naloxone education."
    ),
    required_inputs=["patient_id", "current_medications"],
    steps=[
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 5,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
        WorkflowStep(
            id="faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            inputs={
                "question":
                    "I take morphine and lorazepam. What signs of overdose "
                    "should my family watch for?",
            },
        ),
    ],
)


PEDIATRIC_DOSING_REVIEW = Workflow(
    id="pediatric_dosing_review",
    title="Pediatric prescribing dose review",
    description=(
        "Outpatient pediatric prescribing: composes weight-based "
        "dose computation, polypharmacy DDI scan against existing "
        "regimen, and a parent-language counseling note."
    ),
    required_inputs=[
        "patient_id", "drug", "weight_kg", "age_months",
        "indication", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="dose",
            specialist="trustedrisk-pediatric",
            tool_name="compute_weight_based_dosing",
            callable=compute_weight_based_dosing,
            inputs={
                "drug": "${input.drug}",
                "weight_kg": "${input.weight_kg}",
                "age_months": "${input.age_months}",
                "indication": "${input.indication}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 4,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical E — specialty risk frameworks (E27..E34)
# ─────────────────────────────────────────────────────────────────────

PEDIATRIC_ACUTE_WORKUP = Workflow(
    id="pediatric_acute_workup",
    title="Pediatric acute workup",
    description=(
        "Acutely ill child. Composes PEWS early-warning score, "
        "weight-based dose for the empiric antibiotic, and a "
        "caregiver hand-off package for the parents."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "age_months", "weight_kg",
        "drug", "indication",
    ],
    steps=[
        WorkflowStep(
            id="pews",
            specialist="trustedrisk-pediatric",
            tool_name="compute_pediatric_early_warning",
            callable=compute_pediatric_early_warning,
            inputs={
                "age_months": "${input.age_months}",
                "behavior": "irritable",
                "heart_rate": 150.0,
                "respiratory_rate": 38.0,
                "spo2": 93.0,
                "accessory_muscle_use": True,
                "parental_or_nurse_concern": True,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="dose",
            specialist="trustedrisk-pediatric",
            tool_name="compute_weight_based_dosing",
            callable=compute_weight_based_dosing,
            inputs={
                "drug": "${input.drug}",
                "weight_kg": "${input.weight_kg}",
                "age_months": "${input.age_months}",
                "indication": "${input.indication}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {"action": "admit_observation"},
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


MENTAL_HEALTH_CRISIS = Workflow(
    id="mental_health_crisis",
    title="Mental health crisis assessment",
    description=(
        "Patient presenting with suicidal ideation or psychiatric "
        "emergency. Composes C-SSRS suicide-risk score, psychiatric "
        "admission disposition (voluntary vs involuntary), and a "
        "patient-language counseling note about the safety plan."
    ),
    required_inputs=[
        "patient_id", "ideation_lifetime_level",
        "ideation_past_30d_level", "behavior_lifetime_attempts",
        "behavior_past_30d_any",
    ],
    steps=[
        WorkflowStep(
            id="cssrs",
            specialist="trustedrisk-mental-health",
            tool_name="compute_suicide_risk_assessment",
            callable=compute_suicide_risk_assessment,
            inputs={
                "ideation_lifetime_level":
                    "${input.ideation_lifetime_level}",
                "ideation_past_30d_level":
                    "${input.ideation_past_30d_level}",
                "behavior_lifetime_attempts":
                    "${input.behavior_lifetime_attempts}",
                "behavior_past_30d_any":
                    "${input.behavior_past_30d_any}",
                "warning_factors_count": 3,
                "protective_factors_count": 1,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="admission",
            specialist="trustedrisk-mental-health",
            tool_name="compute_psychiatric_admission_decision",
            callable=compute_psychiatric_admission_decision,
            inputs={
                "risk_level":
                    "${steps.cssrs.output.risk_level}",
                "danger_to_self": True,
                "voluntary_capable": False,
                "has_safety_plan_in_place": False,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": [],
                "lace_score": 8,
                "recommendation_action": "continued_admission",
                "extra_red_flags": [
                    "If you have any thoughts of harming yourself, "
                    "call 988 or go to the nearest emergency room.",
                ],
            },
            optional=True,
        ),
    ],
)


MATERNAL_OBSTETRIC_EMERGENCY = Workflow(
    id="maternal_obstetric_emergency",
    title="Maternal obstetric emergency",
    description=(
        "Pregnant patient with concerning vital signs. Composes "
        "MEOWS deterioration screen, ACOG preeclampsia assessment, "
        "and an admission HnP draft."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "gestational_age_weeks", "systolic_bp", "diastolic_bp",
        "vital_signs_obs",
    ],
    steps=[
        WorkflowStep(
            id="meows",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": "${input.vital_signs_obs}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="preeclampsia",
            specialist="trustedrisk-acute",
            tool_name="compute_preeclampsia_assessment",
            callable=compute_preeclampsia_assessment,
            inputs={
                "gestational_age_weeks":
                    "${input.gestational_age_weeks}",
                "systolic_bp": "${input.systolic_bp}",
                "diastolic_bp": "${input.diastolic_bp}",
                "proteinuria_present": True,
                "clinical_factors": {
                    "headache": True, "visual_disturbance": True,
                },
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "chief_complaint": "obstetric emergency, suspected preeclampsia",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


ONCOLOGY_CYCLE_REVIEW = Workflow(
    id="oncology_cycle_review",
    title="Oncology cycle review",
    description=(
        "Patient mid-treatment for an oncologic regimen. Composes "
        "RECIST response evaluation on the recent imaging, "
        "chemo dose adjustment for the next cycle, treatment "
        "selection re-assessment, and patient counseling. Chains: "
        "RECIST verdict steers the dose-adjustment decision (PD calls "
        "for hold + line switch; CR/PR proceeds), and the treatment-"
        "selection step is gated on RECIST progression so we only "
        "re-evaluate line of therapy when the disease is actually "
        "progressing."
    ),
    required_inputs=[
        "patient_id", "regimen", "cycle_number",
        "target_lesions", "egfr_ml_min", "anc_per_ul",
        "condition", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="recist",
            specialist="trustedrisk-discharge",
            tool_name="compute_oncology_treatment_response",
            callable=compute_oncology_treatment_response,
            inputs={
                "target_lesions": "${input.target_lesions}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="dose",
            specialist="trustedrisk-discharge",
            tool_name="compute_chemo_dose_adjustment",
            callable=compute_chemo_dose_adjustment,
            inputs={
                "regimen": "${input.regimen}",
                "cycle_number": "${input.cycle_number}",
                "egfr_ml_min": "${input.egfr_ml_min}",
                "anc_per_ul": "${input.anc_per_ul}",
                "ecog_performance_status": 1,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="selection",
            specialist="trustedrisk-evidence",
            tool_name="compute_treatment_selection",
            callable=compute_treatment_selection,
            description="Line-of-therapy reassessment (chains RECIST verdict)",
            inputs={
                "condition": "${input.condition}",
                # Chain: when RECIST shows progression, escalate the
                # line; on response, stay on current line. The factor
                # below carries the RECIST overall_response into the
                # treatment_selection's protocol-aware logic.
                "patient_factors": {
                    "line_of_therapy": "second",
                    "recist_response":
                        "${steps.recist.output.overall_response}",
                },
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            description="Patient FAQ (chains dose-adjustment rationale)",
            inputs={
                # Chain: the FAQ answer must reference the same dose
                # decision the clinician just made; otherwise we tell
                # the patient something the chart contradicts.
                "question":
                    "${steps.dose.output.rationale}",
            },
        ),
    ],
)


GERIATRIC_ASSESSMENT = Workflow(
    id="geriatric_assessment",
    title="Comprehensive geriatric assessment",
    description=(
        "Outpatient or inpatient geriatric review. Composes Morse "
        "Falls + CAM delirium + polypharmacy scan, with a caregiver "
        "hand-off package for the family."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="falls",
            specialist="trustedrisk-acute",
            tool_name="compute_falls_risk_morse",
            callable=compute_falls_risk_morse,
            inputs={
                "history_of_falling_3mo": True,
                "secondary_diagnosis_present": True,
                "ambulatory_aid": "cane",
                "gait": "weak",
                "mental_status": "forgets_limitations",
                "current_medications": "${input.current_medications}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="delirium",
            specialist="trustedrisk-acute",
            tool_name="compute_delirium_screening_cam",
            callable=compute_delirium_screening_cam,
            inputs={
                "feature1_acute_onset_or_fluctuating": True,
                "feature2_inattention": True,
                "current_medications": "${input.current_medications}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="polypharmacy",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {"action": "home_with_care"},
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


RHEUMATOLOGY_FOLLOWUP = Workflow(
    id="rheumatology_followup",
    title="Rheumatology followup",
    description=(
        "Patient with rheumatoid arthritis or autoimmune disease "
        "on biologic / DMARD. Composes guideline-grounded treatment "
        "selection, polypharmacy / DDI scan, and patient-language "
        "counseling about monitoring labs."
    ),
    required_inputs=[
        "patient_id", "condition", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="selection",
            specialist="trustedrisk-evidence",
            tool_name="compute_treatment_selection",
            callable=compute_treatment_selection,
            inputs={
                "condition": "${input.condition}",
                "patient_factors": {
                    "biologic_naive": False,
                    "anti_tnf_failures": 2,
                },
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 5,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


TRANSPLANT_IMMUNOSUPPRESSION_REVIEW = Workflow(
    id="transplant_immunosuppression_review",
    title="Transplant immunosuppression review",
    description=(
        "Solid-organ transplant recipient on tacrolimus / "
        "cyclosporine / sirolimus. Composes PGx alternatives + "
        "dose adjustment based on pharmacogenomic profile, plus "
        "DDI scan against the entire regimen."
    ),
    required_inputs=[
        "patient_id", "requested_drug", "genotypes",
        "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="alternatives",
            specialist="trustedrisk-pgx",
            tool_name="compute_pgx_drug_alternatives",
            callable=compute_pgx_drug_alternatives,
            inputs={
                "requested_drug": "${input.requested_drug}",
                "genotypes": "${input.genotypes}",
                "patient_reference": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="dose",
            specialist="trustedrisk-pgx",
            tool_name="compute_pgx_dose_adjustment",
            callable=compute_pgx_dose_adjustment,
            inputs={
                "medications": ["${input.requested_drug}"],
                "genotypes": "${input.genotypes}",
                "patient_reference": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
    ],
)


INFECTIOUS_DISEASE_CONSULT = Workflow(
    id="infectious_disease_consult",
    title="Infectious disease consult",
    description=(
        "Inpatient with complex infection. Composes empiric "
        "antibiotic recommendation, de-escalation plan based on "
        "culture data, and a structured consult letter to ID."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "infection_source",
        "patient_factors", "current_regimen",
    ],
    steps=[
        WorkflowStep(
            id="empiric",
            specialist="trustedrisk-discharge",
            tool_name="compute_empiric_antibiotic_selection",
            callable=compute_empiric_antibiotic_selection,
            inputs={
                "infection_source": "${input.infection_source}",
                "severity": "severe",
                "patient_factors": "${input.patient_factors}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="deescalation",
            specialist="trustedrisk-discharge",
            tool_name="compute_antibiotic_de_escalation",
            callable=compute_antibiotic_de_escalation,
            inputs={
                "current_regimen": "${input.current_regimen}",
                "pathogen": "Pseudomonas aeruginosa",
                "susceptibility": {
                    "piperacillin_tazobactam": "S",
                    "meropenem": "S", "cefepime": "I",
                },
                "days_on_therapy": 5,
                "total_planned_duration_days": 10,
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Inpatient hospitalist",
                "consultation_reason":
                    "Complex infection, antimicrobial stewardship review",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "infectious_diseases",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical F — peri-operative (F35..F38)
# ─────────────────────────────────────────────────────────────────────

PRE_OP_OPTIMIZATION = Workflow(
    id="pre_op_optimization",
    title="Pre-operative optimization",
    description=(
        "Pre-op clinic visit ahead of elective surgery. Composes "
        "symptom red-flag screen, IV-contrast safety re-check (for "
        "any planned imaging), an admission HnP draft, and a "
        "consent letter to the patient."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "raw_input",
    ],
    steps=[
        WorkflowStep(
            id="redflag",
            specialist="trustedrisk-preadmit",
            tool_name="compute_symptom_red_flag_check",
            callable=compute_symptom_red_flag_check,
            inputs={"raw_input": "${input.raw_input}"},
        ),
        WorkflowStep(
            id="contrast",
            specialist="trustedrisk-acute",
            tool_name="compute_contrast_safety_check",
            callable=compute_contrast_safety_check,
            inputs={
                "contrast_type": "iodinated_iv",
                "egfr_ml_min": 58.0,
            },
        ),
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "chief_complaint": "Pre-operative evaluation",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="consent",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": [],
                "lace_score": 5,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


PERIOP_RISK_STRATIFICATION = Workflow(
    id="periop_risk_stratification",
    title="Peri-operative risk stratification",
    description=(
        "Anesthesia / pre-op risk evaluation. Composes Lee Revised "
        "Cardiac Risk Index, ARISCAT pulmonary risk, Caprini VTE "
        "risk, and a structured progress-note draft summarising "
        "the recommendations."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "age", "preop_spo2_pct",
        "surgical_incision_site", "surgical_duration_hours",
    ],
    steps=[
        WorkflowStep(
            id="cardiac_risk",
            specialist="trustedrisk-acute",
            tool_name="compute_rcri_cardiac_risk",
            callable=compute_rcri_cardiac_risk,
            inputs={
                "high_risk_surgery": True,
                "history_ischemic_heart_disease": True,
                "history_congestive_heart_failure": False,
                "creatinine_gt_2_mg_dl": False,
                "history_cerebrovascular_disease": False,
                "insulin_dependent_diabetes": True,
            },
        ),
        WorkflowStep(
            id="pulm_risk",
            specialist="trustedrisk-acute",
            tool_name="compute_ariscat_pulmonary_risk",
            callable=compute_ariscat_pulmonary_risk,
            inputs={
                "age": "${input.age}",
                "preop_spo2_pct": "${input.preop_spo2_pct}",
                "surgical_incision_site":
                    "${input.surgical_incision_site}",
                "surgical_duration_hours":
                    "${input.surgical_duration_hours}",
            },
        ),
        WorkflowStep(
            id="vte_risk",
            specialist="trustedrisk-acute",
            tool_name="compute_caprini_vte_risk",
            callable=compute_caprini_vte_risk,
            inputs={
                "age": "${input.age}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="note",
            specialist="trustedrisk-scribe",
            tool_name="compute_progress_note_draft",
            callable=compute_progress_note_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "subjective_text":
                    "Pre-op risk stratification. RCRI + ARISCAT + Caprini "
                    "computed for anesthesia / surgical team.",
            },
        ),
    ],
)


POST_OP_RECOVERY = Workflow(
    id="post_op_recovery",
    title="Post-operative recovery review",
    description=(
        "Post-op day 1-3 review. Composes deterioration trend, "
        "polypharmacy DDI scan against the post-op regimen, and a "
        "progress-note draft with the recovery plan."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "vital_signs_obs", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="news2",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": "${input.vital_signs_obs}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="note",
            specialist="trustedrisk-scribe",
            tool_name="compute_progress_note_draft",
            callable=compute_progress_note_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "subjective_text":
                    "Post-op day 2: pain controlled, ambulating, "
                    "tolerating PO. Plan for discharge if stable overnight.",
            },
        ),
    ],
)


SURGICAL_CONSENT_WORKUP = Workflow(
    id="surgical_consent_workup",
    title="Surgical consent + shared-decision",
    description=(
        "Pre-op shared-decision conversation. Composes readmission "
        "risk for context, patient-language counseling about the "
        "procedure / risks / alternatives, and a caregiver hand-off "
        "package for the family."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
    ],
    steps=[
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": [],
                "lace_score": 6,
                "recommendation_action": "home_with_care",
                "extra_red_flags": [
                    "After surgery: persistent high fever, "
                    "uncontrolled bleeding, or sudden chest pain "
                    "are reasons to call us right away.",
                ],
            },
            optional=True,
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {"action": "elective_surgery_planned"},
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical G — clinical documentation (G39..G43)
# ─────────────────────────────────────────────────────────────────────

CHART_TO_CODES = Workflow(
    id="chart_to_codes",
    title="Chart-to-codes auto-coding pipeline",
    description=(
        "Bills inbox: a chart text comes in, we want billing-ready "
        "codes plus an audit. Composes clinical NER, ICD-10 + CPT "
        "suggestions, and a coding audit that flags unsupported "
        "codes before the bill goes out."
    ),
    required_inputs=[
        "chart_text", "procedure_text",
    ],
    steps=[
        WorkflowStep(
            id="ner",
            specialist="trustedrisk-evidence",
            tool_name="compute_clinical_ner",
            callable=compute_clinical_ner,
            inputs={"text": "${input.chart_text}", "use_llm": False},
        ),
        WorkflowStep(
            id="icd10",
            specialist="trustedrisk-coder",
            tool_name="compute_icd10_suggest",
            callable=compute_icd10_suggest,
            inputs={"chart_text": "${input.chart_text}"},
        ),
        WorkflowStep(
            id="cpt",
            specialist="trustedrisk-coder",
            tool_name="compute_cpt_suggest",
            callable=compute_cpt_suggest,
            inputs={"procedure_text": "${input.procedure_text}"},
        ),
        WorkflowStep(
            id="audit",
            specialist="trustedrisk-coder",
            tool_name="compute_coding_audit",
            callable=compute_coding_audit,
            inputs={
                "chart_text": "${input.chart_text}",
                "coded_artifact": {"icd10_codes": [], "cpt_codes": []},
            },
            optional=True,
        ),
    ],
)


CLINICAL_DOCUMENTATION_POLISH = Workflow(
    id="clinical_documentation_polish",
    title="Clinical documentation polish",
    description=(
        "End-of-shift documentation pass. Composes admission HnP "
        "draft, discharge summary draft, and a consult letter "
        "draft -- all grounded in chart cite-back IDs."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "chief_complaint",
        "discharge_disposition", "consultation_reason",
    ],
    steps=[
        WorkflowStep(
            id="hnp",
            specialist="trustedrisk-scribe",
            tool_name="compute_admission_hnp_draft",
            callable=compute_admission_hnp_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "chief_complaint": "${input.chief_complaint}",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="summary",
            specialist="trustedrisk-scribe",
            tool_name="compute_discharge_summary_draft",
            callable=compute_discharge_summary_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "discharge_disposition": "${input.discharge_disposition}",
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Hospitalist",
                "consultation_reason": "${input.consultation_reason}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "internal_medicine",
            },
        ),
    ],
)


PROGRESS_NOTE_WORKFLOW = Workflow(
    id="progress_note_workflow",
    title="Progress note documentation",
    description=(
        "Daily round documentation. Mixed mode: NER + med-recon run "
        "in parallel against the inputs, then the progress-note draft "
        "chains on the NER's full_text-augmented context so the SOAP "
        "block inherits the NER's clinical-entity coverage. "
        "Avoids the obvious failure mode where the note draft and the "
        "NER pass disagree on what's in the chart."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "subjective_text",
        "admission_meds", "discharge_meds",
    ],
    steps=[
        # Step 1 — NER first, so the SOAP draft can chain on it.
        WorkflowStep(
            id="ner",
            specialist="trustedrisk-evidence",
            tool_name="compute_clinical_ner",
            callable=compute_clinical_ner,
            description="Clinical entity extraction over the dictation",
            inputs={
                "text": "${input.subjective_text}", "use_llm": False,
            },
        ),
        # Step 2 — Med-rec runs in parallel with NER (independent).
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,
        ),
        # Step 3 — Note draft chains on NER's free-text summary so the
        # SOAP block inherits NER's entity coverage; the clinician sees
        # one coherent narrative instead of two disagreeing passes.
        WorkflowStep(
            id="note",
            specialist="trustedrisk-scribe",
            tool_name="compute_progress_note_draft",
            callable=compute_progress_note_draft,
            description="SOAP draft (chains on NER coverage)",
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "subjective_text": "${input.subjective_text}",
            },
        ),
    ],
)


CONSULT_LETTER_WORKFLOW = Workflow(
    id="consult_letter_workflow",
    title="Consult letter drafting",
    description=(
        "Specialist asks for our impression. Composes a grounded "
        "differential diagnosis, a consult-letter draft addressed "
        "to the requested specialty, and a patient-language FAQ "
        "to share with the patient afterwards."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "chief_complaint",
        "consultant_specialty", "consultation_reason",
    ],
    steps=[
        WorkflowStep(
            id="ddx",
            specialist="trustedrisk-evidence",
            tool_name="compute_differential_diagnosis_ranker",
            callable=compute_differential_diagnosis_ranker,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "structured_features": {},
            },
        ),
        WorkflowStep(
            id="letter",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Outpatient PCP",
                "consultation_reason": "${input.consultation_reason}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "${input.consultant_specialty}",
            },
        ),
        WorkflowStep(
            id="patient_faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            inputs={
                "question":
                    "What does my doctor mean when they want me to see a "
                    "specialist? What should I expect at the appointment?",
            },
        ),
    ],
)


DISCHARGE_SUMMARY_WORKFLOW = Workflow(
    id="discharge_summary_workflow",
    title="Discharge summary drafting",
    description=(
        "Day-of-discharge summary. Composes readmission-risk band "
        "(used for discharge timing wording), medication "
        "reconciliation against the home regimen, and a discharge "
        "summary draft."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "admission_meds", "discharge_meds",
        "discharge_disposition",
    ],
    steps=[
        WorkflowStep(
            id="risk",
            specialist="trustedrisk-discharge",
            tool_name="compute_readmission_risk",
            callable=compute_readmission_risk,
            inputs={"patient_id": "${input.patient_id}"},
            optional=True,
        ),
        WorkflowStep(
            id="med_recon",
            specialist="trustedrisk-discharge",
            tool_name="compute_medication_reconciliation",
            callable=compute_medication_reconciliation,
            inputs={
                "patient_id": "${input.patient_id}",
                "admission_meds": "${input.admission_meds}",
                "discharge_meds": "${input.discharge_meds}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="summary",
            specialist="trustedrisk-scribe",
            tool_name="compute_discharge_summary_draft",
            callable=compute_discharge_summary_draft,
            inputs={
                "patient_reference": "${input.patient_id}",
                "fhir_bundle": "${input.fhir_bundle}",
                "discharge_disposition": "${input.discharge_disposition}",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical H — payor & appeals (H44..H47)
# ─────────────────────────────────────────────────────────────────────

PRIOR_AUTH_PIPELINE = Workflow(
    id="prior_auth_pipeline",
    title="Prior authorization pipeline",
    description=(
        "End-to-end prior-auth submission. Composes evidence-pack "
        "extraction from the chart, payer-rules match for the "
        "requested service, PA letter draft, and the calibrated "
        "approval-likelihood model."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "payer", "requested_service",
    ],
    steps=[
        WorkflowStep(
            id="evidence",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_evidence_pack",
            callable=compute_pa_evidence_pack,
            inputs={
                "patient_reference": "${input.patient_id}",
                "requested_service": "${input.requested_service}",
                "payer": "${input.payer}",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="rules",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_payer_rules_match",
            callable=compute_pa_payer_rules_match,
            inputs={
                "evidence_pack":
                    "${steps.evidence.output}",
            },
        ),
        WorkflowStep(
            id="letter",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_letter_draft",
            callable=compute_pa_letter_draft,
            inputs={
                "evidence_pack":
                    "${steps.evidence.output}",
                "rules_match": "${steps.rules.output}",
            },
        ),
        WorkflowStep(
            id="likelihood",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_appeal_likelihood",
            callable=compute_pa_appeal_likelihood,
            inputs={
                "evidence_pack":
                    "${steps.evidence.output}",
                "rules_match": "${steps.rules.output}",
                "n_prior_denials": 0,
            },
        ),
    ],
)


DENIAL_APPEAL_PIPELINE = Workflow(
    id="denial_appeal_pipeline",
    title="Insurance denial appeal pipeline",
    description=(
        "Insurance denied a service we want covered. Composes "
        "denial-letter parsing, appeal-letter drafting (with the "
        "medical-necessity argument), and the escalation-path "
        "roadmap if the first appeal is denied."
    ),
    required_inputs=[
        "denial_letter_text", "patient_summary",
        "medical_necessity_argument",
    ],
    steps=[
        WorkflowStep(
            id="parse",
            specialist="trustedrisk-appeals",
            tool_name="compute_denial_letter_parse",
            callable=compute_denial_letter_parse,
            inputs={"letter_text": "${input.denial_letter_text}"},
        ),
        WorkflowStep(
            id="draft",
            specialist="trustedrisk-appeals",
            tool_name="compute_appeal_letter_draft",
            callable=compute_appeal_letter_draft,
            inputs={
                "parsed_denial": "${steps.parse.output}",
                "patient_summary": "${input.patient_summary}",
                "medical_necessity_argument":
                    "${input.medical_necessity_argument}",
            },
        ),
        WorkflowStep(
            id="escalation",
            specialist="trustedrisk-appeals",
            tool_name="compute_appeal_escalation_path",
            callable=compute_appeal_escalation_path,
            inputs={
                "payer": "${steps.parse.output.payer}",
                "starting_level": "internal_first_level",
            },
        ),
    ],
)


COST_EFFECTIVENESS_REVIEW = Workflow(
    id="cost_effectiveness_review",
    title="Cost-effectiveness review for shared decision",
    description=(
        "Shared-decision conversation over an expensive intervention. "
        "Composes EVOI / cost-effectiveness analysis at multiple WTP "
        "thresholds and a patient-language counseling note that "
        "translates the dollar / QALY trade-off into the patient's "
        "context."
    ),
    required_inputs=[
        "patient_id", "intervention",
        "baseline_event_probability", "current_medications",
    ],
    steps=[
        WorkflowStep(
            id="evoi",
            specialist="trustedrisk-population",
            tool_name="compute_expected_value_of_intervention",
            callable=compute_expected_value_of_intervention,
            inputs={
                "intervention": "${input.intervention}",
                "baseline_event_probability":
                    "${input.baseline_event_probability}",
                "cohort_size": 100,
                "wtp_threshold_per_qaly_usd": 100000.0,
            },
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                "lace_score": 5,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
        WorkflowStep(
            id="faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            inputs={
                "question":
                    "Is this medication worth the out-of-pocket cost? "
                    "What does cost-effectiveness mean for me?",
            },
        ),
    ],
)


COVERAGE_DETERMINATION = Workflow(
    id="coverage_determination",
    title="Coverage determination + value review",
    description=(
        "Pre-submission self-check: builds the PA evidence pack, "
        "matches it against the payer's published rules, and runs "
        "an EVOI / cost-effectiveness analysis so the requesting "
        "clinician sees both whether the payer will likely cover "
        "and whether the request is value-defensible."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "payer", "requested_service",
        "intervention", "baseline_event_probability",
    ],
    steps=[
        WorkflowStep(
            id="evidence",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_evidence_pack",
            callable=compute_pa_evidence_pack,
            inputs={
                "patient_reference": "${input.patient_id}",
                "requested_service": "${input.requested_service}",
                "payer": "${input.payer}",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="rules",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_payer_rules_match",
            callable=compute_pa_payer_rules_match,
            inputs={
                "evidence_pack":
                    "${steps.evidence.output}",
            },
        ),
        WorkflowStep(
            id="value",
            specialist="trustedrisk-population",
            tool_name="compute_expected_value_of_intervention",
            callable=compute_expected_value_of_intervention,
            inputs={
                "intervention": "${input.intervention}",
                "baseline_event_probability":
                    "${input.baseline_event_probability}",
                "cohort_size": 100,
                "wtp_threshold_per_qaly_usd": 100000.0,
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical I — population & quality (I48..I50)
# ─────────────────────────────────────────────────────────────────────

POPULATION_OUTREACH = Workflow(
    id="population_outreach",
    title="Population outreach pipeline",
    description=(
        "Quality officer plans next-quarter outreach. Composes "
        "syndromic surveillance + vaccine reminder cohort sizing + "
        "DP-Laplace outbreak heatmap so outreach prioritisation is "
        "anchored to both reach and equity."
    ),
    required_inputs=[
        "surveillance_period_start_iso", "surveillance_period_end_iso",
        "observed_counts_by_syndrome", "expected_counts_by_syndrome",
        "overdue_by_vaccine", "counts_by_geo_syndrome",
        "population_by_geo",
    ],
    steps=[
        WorkflowStep(
            id="surveillance",
            specialist="trustedrisk-pophealth",
            tool_name="compute_syndromic_surveillance",
            callable=compute_syndromic_surveillance,
            inputs={
                "surveillance_period_start_iso":
                    "${input.surveillance_period_start_iso}",
                "surveillance_period_end_iso":
                    "${input.surveillance_period_end_iso}",
                "observed_counts_by_syndrome":
                    "${input.observed_counts_by_syndrome}",
                "expected_counts_by_syndrome":
                    "${input.expected_counts_by_syndrome}",
            },
        ),
        WorkflowStep(
            id="vaccine",
            specialist="trustedrisk-pophealth",
            tool_name="compute_vaccine_reminder_cohort",
            callable=compute_vaccine_reminder_cohort,
            inputs={
                "overdue_by_vaccine":
                    "${input.overdue_by_vaccine}",
            },
        ),
        WorkflowStep(
            id="heatmap",
            specialist="trustedrisk-pophealth",
            tool_name="compute_outbreak_heatmap",
            callable=compute_outbreak_heatmap,
            inputs={
                "counts_by_geo_syndrome":
                    "${input.counts_by_geo_syndrome}",
                "population_by_geo":
                    "${input.population_by_geo}",
            },
        ),
    ],
)


CARE_GAP_CLOSURE_PIPELINE = Workflow(
    id="care_gap_closure_pipeline",
    title="Care-gap closure outreach",
    description=(
        "Outpatient care-coordinator closing HEDIS gaps for a "
        "specific patient. Composes care-gap detection on the "
        "active chart, a patient-language FAQ explaining the "
        "rationale, and a caregiver hand-off package."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "patient_age", "patient_sex",
        "patient_question",
    ],
    steps=[
        WorkflowStep(
            id="gaps",
            specialist="trustedrisk-discharge",
            tool_name="compute_care_gap_detector",
            callable=compute_care_gap_detector,
            inputs={
                "fhir_bundle": "${input.fhir_bundle}",
                "patient_age": "${input.patient_age}",
                "patient_sex": "${input.patient_sex}",
            },
        ),
        WorkflowStep(
            id="faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            inputs={"question": "${input.patient_question}"},
            # FAQ requires a real-time question; not derivable from FHIR.
            optional=True,
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {
                        "action": "outpatient_care_gap_closure",
                    },
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


HEDIS_QUALITY_IMPROVEMENT = Workflow(
    id="hedis_quality_improvement",
    title="HEDIS quality-improvement pipeline",
    description=(
        "Star Ratings forecast + care-gap priority ranking for an "
        "MA contract. Composes the per-measure aggregate, the "
        "Stars rating forecast, the care-gap priority ranking by "
        "expected QALY-lift, and a counseling note for the "
        "outreach team."
    ),
    required_inputs=[
        "patient_id", "measurement_year", "cohort_summary",
        "n_eligible_patients", "contract_size_thousand_members",
    ],
    steps=[
        WorkflowStep(
            id="aggregate",
            specialist="trustedrisk-quality",
            tool_name="compute_quality_measures_aggregate",
            callable=compute_quality_measures_aggregate,
            inputs={
                "measurement_year": "${input.measurement_year}",
                "cohort_summary": "${input.cohort_summary}",
                "n_eligible_patients":
                    "${input.n_eligible_patients}",
            },
        ),
        WorkflowStep(
            id="stars",
            specialist="trustedrisk-quality",
            tool_name="compute_stars_rating_forecast",
            callable=compute_stars_rating_forecast,
            inputs={
                "aggregate":
                    "${steps.aggregate.output}",
                "contract_size_thousand_members":
                    "${input.contract_size_thousand_members}",
            },
        ),
        WorkflowStep(
            id="priority",
            specialist="trustedrisk-quality",
            tool_name="compute_care_gap_priority_ranking",
            callable=compute_care_gap_priority_ranking,
            inputs={
                "aggregate":
                    "${steps.aggregate.output}",
                "top_n": 10,
            },
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": [],
                "lace_score": 4,
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Vertical J — horizontal expansion (J51..J58, hand-crafted base
# workflows filling clinical gaps in the original 50). Each carries an
# explicit chain on at least one downstream step; the pattern mirrors
# the existing flagship chains (sepsis, antibiotic stewardship, etc.).
# ─────────────────────────────────────────────────────────────────────


PRE_HOSPITAL_HANDOFF = Workflow(
    id="pre_hospital_handoff",
    title="EMS pre-hospital -> ED handoff",
    description=(
        "EMS calls in. Composes early triage on the dispatch summary, "
        "a deterioration baseline against the field vitals, and a "
        "consult-letter style trauma/medicine notification chained on "
        "the triage rationale so the ED team's prep matches the "
        "chosen acuity tier."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "chief_complaint", "vital_signs_dict", "vital_signs_obs",
        "age",
    ],
    steps=[
        WorkflowStep(
            id="triage",
            specialist="trustedrisk-acute",
            tool_name="compute_admission_triage",
            callable=compute_admission_triage,
            inputs={
                "chief_complaint": "${input.chief_complaint}",
                "vital_signs": "${input.vital_signs_dict}",
                "age": "${input.age}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="deterioration",
            specialist="trustedrisk-acute",
            tool_name="compute_clinical_deterioration_score",
            callable=compute_clinical_deterioration_score,
            inputs={
                "vital_signs": "${input.vital_signs_obs}",
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="ed_alert",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            description="ED alert (chains on triage rationale)",
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "EMS field team",
                # Chain: ED prep is driven by triage.disposition + the
                # ESI rationale, not the raw chief complaint.
                "consultation_reason": "${steps.triage.output.rationale}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "emergency_medicine",
            },
        ),
    ],
)


VTE_PROPHYLAXIS_REVIEW = Workflow(
    id="vte_prophylaxis_review",
    title="Inpatient VTE prophylaxis review",
    description=(
        "Universal inpatient VTE risk + prophylaxis selection. Caprini "
        "score in parallel with a DDI scan (anticoagulant interactions "
        "matter for the choice between LMWH / DOAC / mechanical), then "
        "a counseling note that chains on Caprini risk so the patient "
        "hears about the same chart-recorded discharge plan."
    ),
    required_inputs=[
        "patient_id", "current_medications",
        "age", "bmi_gt_25", "active_malignancy",
        "confined_to_bed_gt_72h", "major_surgery_planned",
    ],
    steps=[
        WorkflowStep(
            id="caprini",
            specialist="trustedrisk-acute",
            tool_name="compute_caprini_vte_risk",
            callable=compute_caprini_vte_risk,
            inputs={
                "age": "${input.age}",
                "bmi_gt_25": "${input.bmi_gt_25}",
                "active_malignancy": "${input.active_malignancy}",
                "confined_to_bed_gt_72h":
                    "${input.confined_to_bed_gt_72h}",
                "major_surgery_planned":
                    "${input.major_surgery_planned}",
            },
        ),
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="counseling",
            specialist="trustedrisk-discharge",
            tool_name="compute_discharge_counseling",
            callable=compute_discharge_counseling,
            description="Patient counseling (chains Caprini total score)",
            inputs={
                "patient_id": "${input.patient_id}",
                "medications": "${input.current_medications}",
                # Chain: drive the LACE proxy from the Caprini total
                # so a high-risk patient gets richer counseling.
                "lace_score": "${steps.caprini.output.score}",
                "recommendation_action": "home_with_care",
            },
            optional=True,
        ),
    ],
)


POSTPARTUM_FOLLOWUP = Workflow(
    id="postpartum_followup",
    title="Postpartum 6-week followup",
    description=(
        "Postpartum visit. Composes MEOWS-style maternal vitals review, "
        "a polypharmacy DDI for breastfeeding-compatible meds, and a "
        "caregiver hand-off chained on the MEOWS severity tier so the "
        "partner / family knows what red flags to watch for."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "respiratory_rate", "spo2", "heart_rate",
        "systolic_bp", "diastolic_bp", "temperature",
        "current_medications", "gestational_age_weeks",
    ],
    steps=[
        WorkflowStep(
            id="meows",
            specialist="trustedrisk-acute",
            tool_name="compute_maternal_early_warning",
            callable=compute_maternal_early_warning,
            inputs={
                "gestational_age_weeks": "${input.gestational_age_weeks}",
                "pregnancy_phase": "postpartum",
                "respiratory_rate": "${input.respiratory_rate}",
                "spo2": "${input.spo2}",
                "heart_rate": "${input.heart_rate}",
                "systolic_bp": "${input.systolic_bp}",
                "diastolic_bp": "${input.diastolic_bp}",
                "temperature": "${input.temperature}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            description="Family hand-off (chains on MEOWS severity tier)",
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {
                        # Chain: MEOWS green = home; yellow = same-day
                        # call; red = ED. The severity tier IS the
                        # action.
                        "action":
                            "${steps.meows.output.recommended_response}",
                    },
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
    ],
)


BEHAVIORAL_HEALTH_STEP_DOWN = Workflow(
    id="behavioral_health_step_down",
    title="Behavioral health step-down (post-crisis MAT followup)",
    description=(
        "Patient in step-down after a behavioral-health crisis or with "
        "active opioid use disorder on MAT. Composes C-SSRS + a "
        "psychiatric admission decision chained on the C-SSRS risk "
        "level (escalates back to inpatient when ideation worsens), a "
        "DDI scan focused on benzo + opioid + sedative interactions, "
        "and a peer-support handoff."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "ideation_lifetime_level", "ideation_past_30d_level",
        "behavior_lifetime_attempts", "behavior_past_30d_any",
        "warning_factors_count", "protective_factors_count",
        "current_medications", "patient_demographics",
    ],
    steps=[
        WorkflowStep(
            id="cssrs",
            specialist="trustedrisk-mental-health",
            tool_name="compute_suicide_risk_assessment",
            callable=compute_suicide_risk_assessment,
            inputs={
                "ideation_lifetime_level":
                    "${input.ideation_lifetime_level}",
                "ideation_past_30d_level":
                    "${input.ideation_past_30d_level}",
                "behavior_lifetime_attempts":
                    "${input.behavior_lifetime_attempts}",
                "behavior_past_30d_any":
                    "${input.behavior_past_30d_any}",
                "warning_factors_count":
                    "${input.warning_factors_count}",
                "protective_factors_count":
                    "${input.protective_factors_count}",
                "patient_demographics":
                    "${input.patient_demographics}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="admission",
            specialist="trustedrisk-mental-health",
            tool_name="compute_psychiatric_admission_decision",
            callable=compute_psychiatric_admission_decision,
            description="Admission decision (chains on C-SSRS risk level)",
            inputs={
                # Chain: the involuntary-hold decision is gated on the
                # C-SSRS risk band, not a freeform input.
                "risk_level": "${steps.cssrs.output.risk_level}",
                "danger_to_self": True,
                "danger_to_others": False,
                "grave_disability": False,
                "voluntary_capable": True,
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="ddi",
            specialist="trustedrisk-discharge",
            tool_name="detect_polypharmacy_concerns",
            callable=detect_polypharmacy_concerns,
            inputs={"medications": "${input.current_medications}"},
        ),
        WorkflowStep(
            id="handoff",
            specialist="trustedrisk-patient",
            tool_name="compute_caregiver_handoff",
            callable=compute_caregiver_handoff,
            description="Peer-support handoff (chains admission disposition)",
            inputs={
                "decision_card": {
                    "patient_reference": "${input.patient_id}",
                    "recommendation": {
                        "action":
                            "${steps.admission.output.disposition}",
                    },
                },
                "fhir_bundle": "${input.fhir_bundle}",
            },
            optional=True,
        ),
    ],
)


VACCINE_SCHEDULE_CHECK = Workflow(
    id="vaccine_schedule_check",
    title="ACIP vaccine schedule + gap closure",
    description=(
        "Identifies the patient's outstanding ACIP-recommended vaccines, "
        "checks for chart-side care gaps, and produces a patient-FAQ "
        "answer chained on the cohort summary so the patient hears "
        "about the actual outstanding vaccines."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "patient_age", "patient_sex",
        "overdue_by_vaccine",
    ],
    steps=[
        WorkflowStep(
            id="reminder",
            specialist="trustedrisk-pophealth",
            tool_name="compute_vaccine_reminder_cohort",
            callable=compute_vaccine_reminder_cohort,
            inputs={
                "overdue_by_vaccine": "${input.overdue_by_vaccine}",
                "outreach_channel": "phone",
            },
        ),
        WorkflowStep(
            id="gaps",
            specialist="trustedrisk-quality",
            tool_name="compute_care_gap_detector",
            callable=compute_care_gap_detector,
            inputs={
                "fhir_bundle": "${input.fhir_bundle}",
                "patient_age": "${input.patient_age}",
                "patient_sex": "${input.patient_sex}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            description="Patient FAQ (chains on cohort rationale)",
            inputs={
                # Chain: the FAQ is grounded on the actual outstanding
                # vaccine the panel surfaces (flu, pneumo, RSV, etc).
                "question":
                    "${steps.reminder.output.rationale}",
            },
        ),
    ],
)


WELL_CHILD_VISIT = Workflow(
    id="well_child_visit",
    title="Pediatric well-child visit",
    description=(
        "Pediatric routine visit. Composes PEWS baseline, a weight-"
        "based dosing safety check on currently-on meds, vaccine "
        "schedule check, and a caregiver-language FAQ chained on the "
        "PEWS rationale."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "age_months", "weight_kg",
        "heart_rate", "respiratory_rate", "spo2",
        "current_medication", "overdue_by_vaccine",
    ],
    steps=[
        WorkflowStep(
            id="pews",
            specialist="trustedrisk-pediatric",
            tool_name="compute_pediatric_early_warning",
            callable=compute_pediatric_early_warning,
            inputs={
                "age_months": "${input.age_months}",
                "heart_rate": "${input.heart_rate}",
                "respiratory_rate": "${input.respiratory_rate}",
                "spo2": "${input.spo2}",
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="dosing",
            specialist="trustedrisk-pediatric",
            tool_name="compute_weight_based_dosing",
            callable=compute_weight_based_dosing,
            inputs={
                "weight_kg": "${input.weight_kg}",
                "drug": "${input.current_medication}",
                "age_months": "${input.age_months}",
                "indication": "routine",
                "patient_id": "${input.patient_id}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="vaccines",
            specialist="trustedrisk-pophealth",
            tool_name="compute_vaccine_reminder_cohort",
            callable=compute_vaccine_reminder_cohort,
            inputs={
                "overdue_by_vaccine": "${input.overdue_by_vaccine}",
                "outreach_channel": "phone",
            },
            optional=True,
        ),
        WorkflowStep(
            id="faq",
            specialist="trustedrisk-patient",
            tool_name="compute_patient_faq",
            callable=compute_patient_faq,
            description="Caregiver FAQ (chains on PEWS rationale)",
            inputs={
                "question": "${steps.pews.output.rationale}",
            },
        ),
    ],
)


SSI_PREVENTION_BUNDLE = Workflow(
    id="ssi_prevention_bundle",
    title="Surgical-site infection prevention bundle",
    description=(
        "Pre-op SSI prevention: Caprini VTE risk + ARISCAT pulmonary "
        "risk + empiric peri-op abx selection chained on the worst-case "
        "risk band, plus a consult letter to surgery."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle",
        "age", "bmi_gt_25", "active_malignancy",
        "major_surgery_planned",
        "preop_spo2_pct", "surgical_incision_site",
        "surgical_duration_hours", "infection_source",
    ],
    steps=[
        WorkflowStep(
            id="caprini",
            specialist="trustedrisk-acute",
            tool_name="compute_caprini_vte_risk",
            callable=compute_caprini_vte_risk,
            inputs={
                "age": "${input.age}",
                "bmi_gt_25": "${input.bmi_gt_25}",
                "active_malignancy": "${input.active_malignancy}",
                "major_surgery_planned":
                    "${input.major_surgery_planned}",
            },
        ),
        WorkflowStep(
            id="ariscat",
            specialist="trustedrisk-acute",
            tool_name="compute_ariscat_pulmonary_risk",
            callable=compute_ariscat_pulmonary_risk,
            inputs={
                "age": "${input.age}",
                "preop_spo2_pct": "${input.preop_spo2_pct}",
                "surgical_incision_site":
                    "${input.surgical_incision_site}",
                "surgical_duration_hours":
                    "${input.surgical_duration_hours}",
            },
            optional=True,
        ),
        WorkflowStep(
            id="abx",
            specialist="trustedrisk-discharge",
            tool_name="compute_empiric_antibiotic_selection",
            callable=compute_empiric_antibiotic_selection,
            description="Peri-op abx (chains Caprini risk band)",
            inputs={
                "infection_source": "${input.infection_source}",
                "severity": "moderate",
                "patient_factors": {
                    "caprini_band":
                        "${steps.caprini.output.risk_band}",
                },
                "patient_id": "${input.patient_id}",
            },
        ),
        WorkflowStep(
            id="consult",
            specialist="trustedrisk-scribe",
            tool_name="compute_consult_letter_draft",
            callable=compute_consult_letter_draft,
            description="Surgery consult (chains on Caprini rationale)",
            inputs={
                "patient_reference": "${input.patient_id}",
                "referring_clinician": "Pre-op clinic",
                "consultation_reason":
                    "${steps.caprini.output.rationale}",
                "fhir_bundle": "${input.fhir_bundle}",
                "consultant_specialty": "general_surgery",
            },
        ),
    ],
)


OUT_OF_NETWORK_REFERRAL = Workflow(
    id="out_of_network_referral",
    title="Out-of-network specialist referral",
    description=(
        "Patient needs a sub-specialty in-network plan won't cover. "
        "Composes the PA evidence pack, payer-rules match for the "
        "OON benefit, an approval-likelihood projection chained on "
        "evidence + rules, and a pre-emptive appeal-escalation map "
        "for the predictable denial."
    ),
    required_inputs=[
        "patient_id", "fhir_bundle", "payer",
        "requested_service",  # PARequestedService dict
    ],
    steps=[
        WorkflowStep(
            id="evidence",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_evidence_pack",
            callable=compute_pa_evidence_pack,
            inputs={
                "patient_reference": "${input.patient_id}",
                "requested_service": "${input.requested_service}",
                "payer": "${input.payer}",
                "fhir_bundle": "${input.fhir_bundle}",
            },
        ),
        WorkflowStep(
            id="rules",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_payer_rules_match",
            callable=compute_pa_payer_rules_match,
            description="Payer rules match (chains evidence pack)",
            inputs={"evidence_pack": "${steps.evidence.output}"},
        ),
        WorkflowStep(
            id="likelihood",
            specialist="trustedrisk-pa",
            tool_name="compute_pa_appeal_likelihood",
            callable=compute_pa_appeal_likelihood,
            description="Approval likelihood (chains evidence + rules)",
            inputs={
                "evidence_pack": "${steps.evidence.output}",
                "rules_match": "${steps.rules.output}",
                "n_prior_denials": 0,
            },
        ),
        WorkflowStep(
            id="escalation_path",
            specialist="trustedrisk-appeals",
            tool_name="compute_appeal_escalation_path",
            callable=compute_appeal_escalation_path,
            description="Pre-emptive escalation map",
            inputs={
                "payer": "${input.payer}",
                "starting_level": "internal_first_level",
            },
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────
# Macro layer — Care Engine compositions (M1..M50)
#
# A macro is a workflow whose every step is a `sub_workflow_id` hop
# rather than a tool call. Each hop runs a base workflow recursively,
# and the macro's chain is the implicit care-arc dependency: hop_2's
# patient is the same patient hop_1 just operated on, hop_3 sees the
# same chart hop_2 documented. The full sub-workflow trace is
# accessible via ${steps.hop_<n>.output.<inner_step>.<field>} in case
# a downstream needs to read a specific inner artefact.
# ─────────────────────────────────────────────────────────────────────


def _make_macro(
    macro_id: str,
    title: str,
    chain: list[str],
    *,
    description_extra: str = "",
    required: list[str] | None = None,
) -> Workflow:
    """Build a macro workflow that runs `chain` sub-workflows in order.

    Each hop inherits the macro's full input dict, so any input the
    macro declares is forwarded to every sub-workflow that needs it.
    Missing required inputs at the sub-level are filled with empty
    strings; sub-workflow steps with `optional=True` skip cleanly.
    """
    arrows = " -> ".join(chain)
    desc = (
        f"Care-arc macro: {arrows}. " + description_extra
    ).strip()
    return Workflow(
        id=macro_id,
        title=title,
        description=desc,
        required_inputs=required or ["patient_id"],
        steps=[
            WorkflowStep(
                id=f"hop_{i + 1}_{sid}",
                specialist=f"care-engine",
                tool_name="_macro_call",
                callable=None,
                sub_workflow_id=sid,
                inputs={},
                optional=True,
            )
            for i, sid in enumerate(chain)
        ],
    )


_MACRO_DEFS: list[tuple[str, str, list[str]]] = [
    ("complete_chf_admission",
        "Complete CHF admission arc",
        ["chf_admission", "discharge_planning", "post_discharge_followup"]),
    ("complete_sepsis_pipeline",
        "Complete sepsis pipeline",
        ["sepsis_workup", "antibiotic_stewardship",
         "transitional_care_management"]),
    ("trauma_to_rehab",
        "Polytrauma full arc",
        ["trauma_resus_decision", "ssi_prevention_bundle",
         "post_op_recovery", "transitional_care_management"]),
    ("dka_to_clinic",
        "DKA inpatient -> outpatient arc",
        ["dka_management", "chronic_disease_followup",
         "transitional_care_management"]),
    ("ed_to_home",
        "Pre-hospital -> ED -> discharge arc",
        ["pre_hospital_handoff", "chf_admission", "discharge_planning"]),
    ("peds_acute_arc",
        "Pediatric acute -> follow-up arc",
        ["pediatric_acute_workup", "post_discharge_followup"]),
    ("mh_arc",
        "Mental-health crisis -> step-down arc",
        ["mental_health_crisis", "behavioral_health_step_down",
         "transitional_care_management"]),
    ("periop_arc",
        "Peri-operative full arc",
        ["pre_op_optimization", "periop_risk_stratification",
         "post_op_recovery", "transitional_care_management"]),
    ("oncology_arc",
        "Oncology cycle + chronic followup",
        ["oncology_cycle_review", "chronic_disease_followup"]),
    ("denial_arc",
        "Denial -> cost-effectiveness arc",
        ["denial_appeal_pipeline", "cost_effectiveness_review"]),
    ("pa_with_preemptive_appeal",
        "PA + preemptive appeal arc",
        ["prior_auth_pipeline", "denial_appeal_pipeline"]),
    ("quality_arc",
        "HEDIS + care-gap closure arc",
        ["hedis_quality_improvement", "care_gap_closure_pipeline"]),
    ("ob_full",
        "Obstetric emergency -> postpartum arc",
        ["maternal_obstetric_emergency", "postpartum_followup",
         "caregiver_handoff_prep"]),
    ("mat_arc",
        "MH crisis -> MAT step-down arc",
        ["mental_health_crisis", "behavioral_health_step_down",
         "post_discharge_followup"]),
    ("polypharm_arc",
        "Polypharmacy review arc",
        ["ddi_audit", "opioid_safety_review", "caregiver_handoff_prep"]),
    ("abx_arc",
        "Sepsis -> stewardship arc",
        ["sepsis_workup", "antibiotic_stewardship"]),
    ("document_arc",
        "Progress -> discharge -> coding arc",
        ["progress_note_workflow", "discharge_summary_workflow",
         "chart_to_codes"]),
    ("specialty_referral_arc",
        "Consult letter + OON referral arc",
        ["consult_letter_workflow", "out_of_network_referral"]),
    ("rapid_response_arc",
        "Floor deterioration -> ICU step-down",
        ["inpatient_deterioration_response", "icu_step_down"]),
    ("peds_full",
        "Pediatric pre-hospital -> handoff arc",
        ["pre_hospital_handoff", "pediatric_acute_workup",
         "caregiver_handoff_prep"]),
    ("resp_to_icu",
        "Respiratory failure -> ICU arc",
        ["respiratory_failure_workup", "icu_step_down"]),
    ("inpatient_geri_arc",
        "Inpatient geriatric arc",
        ["inpatient_falls_intervention", "geriatric_assessment",
         "transitional_care_management"]),
    ("peds_well_arc",
        "Pediatric well-child arc",
        ["well_child_visit", "pediatric_dosing_review",
         "vaccine_schedule_check"]),
    ("transplant_arc",
        "Transplant maintenance arc",
        ["transplant_immunosuppression_review", "ddi_audit"]),
    ("pgx_full_arc",
        "PGx + DDI + anticoagulant arc",
        ["pgx_prescribing_check", "ddi_audit", "anticoagulant_review"]),
    ("hf_clinic_arc",
        "Heart-failure clinic arc",
        ["chronic_disease_followup", "cost_effectiveness_review"]),
    ("pop_health_full",
        "Population health full arc",
        ["population_outreach", "care_gap_closure_pipeline",
         "hedis_quality_improvement"]),
    ("preop_clear_arc",
        "Pre-op clearance arc",
        ["pre_op_optimization", "ssi_prevention_bundle"]),
    ("infectious_full",
        "Infectious disease + stewardship arc",
        ["infectious_disease_consult", "antibiotic_stewardship"]),
    ("mat_post_arc",
        "MAT step-down -> chronic followup",
        ["behavioral_health_step_down", "chronic_disease_followup"]),
    ("recall_outreach",
        "Vaccine recall + population outreach",
        ["vaccine_schedule_check", "population_outreach"]),
    ("complex_admission",
        "Complex CHF + discharge + appeal arc",
        ["chf_admission", "discharge_planning",
         "denial_appeal_pipeline"]),
    ("ob_obs_to_dispo",
        "Obstetric obs -> caregiver handoff",
        ["maternal_obstetric_emergency", "caregiver_handoff_prep"]),
    ("peds_critical",
        "Pediatric critical -> family handoff",
        ["pediatric_acute_workup", "icu_step_down",
         "caregiver_handoff_prep"]),
    ("surgical_full",
        "Surgical full lifecycle arc",
        ["surgical_consent_workup", "pre_op_optimization",
         "ssi_prevention_bundle", "post_op_recovery"]),
    ("value_chain_arc",
        "Value defensibility arc",
        ["cost_effectiveness_review", "coverage_determination",
         "prior_auth_pipeline"]),
    ("claim_audit_arc",
        "Coding -> cost-effectiveness arc",
        ["chart_to_codes", "cost_effectiveness_review"]),
    ("acos_arc",
        "ACO contract arc",
        ["hedis_quality_improvement", "cost_effectiveness_review",
         "population_outreach"]),
    ("cardiac_clearance_arc",
        "Cardiac peri-op clearance arc",
        ["acute_chest_pain", "pre_op_optimization",
         "periop_risk_stratification"]),
    ("anticoag_lifecycle",
        "Anticoagulant lifecycle arc",
        ["pgx_prescribing_check", "anticoagulant_review", "ddi_audit"]),
    ("snf_handoff",
        "SNF handoff arc",
        ["transitional_care_management", "caregiver_handoff_prep"]),
    ("palliative_full",
        "Palliative transition arc",
        ["palliative_transition", "caregiver_handoff_prep",
         "transitional_care_management"]),
    ("discharge_full",
        "Full discharge handoff arc",
        ["discharge_planning", "caregiver_handoff_prep",
         "post_discharge_followup"]),
    ("infectious_pa",
        "PA + ID consult + stewardship",
        ["prior_auth_pipeline", "infectious_disease_consult",
         "antibiotic_stewardship"]),
    ("registries_full",
        "Registry-driven outreach arc",
        ["vaccine_schedule_check", "care_gap_closure_pipeline"]),
    ("complete_outpatient",
        "Complete outpatient cycle",
        ["outpatient_med_review", "chronic_disease_followup",
         "caregiver_handoff_prep"]),
    ("ed_psych_arc",
        "ED -> psych admission -> step-down",
        ["mental_health_crisis", "behavioral_health_step_down"]),
    ("post_discharge_recovery",
        "Post-discharge recovery + TCM",
        ["post_discharge_followup", "transitional_care_management"]),
    ("polytrauma_resilience_arc",
        "Polytrauma resilience arc",
        ["trauma_resus_decision", "post_op_recovery"]),
    ("vaccine_outreach_arc",
        "Vaccine outreach arc",
        ["vaccine_schedule_check", "population_outreach",
         "care_gap_closure_pipeline"]),
]

_MACRO_WORKFLOWS: list[Workflow] = [
    _make_macro(
        macro_id, title, chain,
        required=["patient_id", "fhir_bundle"],
    )
    for macro_id, title, chain in _MACRO_DEFS
]


# ─────────────────────────────────────────────────────────────────────
# Parametric layer — variant factory (P1..P150)
#
# Each parametric variant is a base workflow with its `id` suffixed by
# a parameter tuple and a small `inputs` overlay applied to every step
# that mentions the parameter axis. This is how the Care Engine
# advertises payer-specific, age-specific, and severity-specific
# variants without 150 hand-written copies.
# ─────────────────────────────────────────────────────────────────────


def _make_parametric_variant(
    base: Workflow,
    suffix: str,
    description_extra: str,
    input_overrides: dict[str, Any] | None = None,
) -> Workflow:
    """Clone `base` with a new id and an overlay on per-step inputs.

    The overlay matches by input-key name -- any step whose `inputs`
    dict has a key in `input_overrides` gets the overridden value
    instead of the base value. Steps without that key are unchanged.
    """
    overrides = input_overrides or {}
    new_steps: list[WorkflowStep] = []
    for s in base.steps:
        new_inputs = dict(s.inputs)
        for k, v in overrides.items():
            if k in new_inputs:
                new_inputs[k] = v
        new_steps.append(WorkflowStep(
            id=s.id, specialist=s.specialist, tool_name=s.tool_name,
            callable=s.callable, sub_workflow_id=s.sub_workflow_id,
            inputs=new_inputs, description=s.description,
            optional=s.optional,
        ))
    return Workflow(
        id=f"{base.id}__{suffix}",
        title=f"{base.title} ({suffix.replace('_', ' ')})",
        description=base.description + " " + description_extra,
        required_inputs=list(base.required_inputs),
        steps=new_steps,
    )


# (base_workflow, list-of-(suffix, description_extra, overrides))
_PARAMETRIC_AXES: list[tuple[Workflow, list[tuple[str, str, dict]]]] = [
    (DISCHARGE_PLANNING, [
        ("medicare_low_acuity",
            "Medicare beneficiary, low acuity (~LACE 4-6).",
            {"lace_score": 5, "discharge_action": "home_with_care"}),
        ("medicare_moderate",
            "Medicare beneficiary, moderate acuity (~LACE 7-9).",
            {"lace_score": 8, "discharge_action": "home_with_care"}),
        ("medicare_high_acuity",
            "Medicare beneficiary, high acuity (~LACE 10+).",
            {"lace_score": 11, "discharge_action": "snf"}),
        ("medicaid_low",
            "Medicaid beneficiary, low acuity.",
            {"lace_score": 5, "discharge_action": "home_with_care"}),
        ("medicaid_high",
            "Medicaid beneficiary, high acuity.",
            {"lace_score": 11, "discharge_action": "continued_admission"}),
        ("commercial_low",
            "Commercial insurance, low acuity.",
            {"lace_score": 4, "discharge_action": "home"}),
        ("commercial_high",
            "Commercial insurance, high acuity.",
            {"lace_score": 10, "discharge_action": "snf"}),
        ("uninsured_handoff",
            "Uninsured patient, handoff to safety-net clinic.",
            {"lace_score": 8, "discharge_action": "home_with_care"}),
    ]),
    (CHF_ADMISSION, [
        ("65to74", "Patient aged 65-74.", {"age": 70}),
        ("75to84", "Patient aged 75-84.", {"age": 80}),
        ("85plus", "Patient aged 85+.", {"age": 88}),
        ("under65_chronic",
            "Under-65 with established chronic CHF.", {"age": 58}),
    ]),
    (SEPSIS_WORKUP, [
        ("urinary_septic",
            "Urinary source, septic.",
            {"infection_source": "urinary", "severity": "septic"}),
        ("urinary_septic_shock",
            "Urinary source, septic shock.",
            {"infection_source": "urinary",
             "severity": "septic_shock"}),
        ("pneumonia_septic",
            "Pneumonia, septic.",
            {"infection_source": "pneumonia", "severity": "septic"}),
        ("pneumonia_shock",
            "Pneumonia, septic shock.",
            {"infection_source": "pneumonia",
             "severity": "septic_shock"}),
        ("skin_septic",
            "Skin/soft-tissue, septic.",
            {"infection_source": "skin", "severity": "septic"}),
        ("intra_abdominal_septic",
            "Intra-abdominal, septic.",
            {"infection_source": "intra_abdominal", "severity": "septic"}),
        ("bloodstream_shock",
            "Bloodstream, septic shock.",
            {"infection_source": "bloodstream",
             "severity": "septic_shock"}),
    ]),
    (PRIOR_AUTH_PIPELINE, [
        ("uhc",
            "UnitedHealthcare submission.",
            {"payer": "unitedhealth"}),
        ("anthem",
            "Anthem / BCBS submission.",
            {"payer": "anthem_bcbs"}),
        ("aetna",
            "Aetna submission.",
            {"payer": "aetna"}),
        ("cigna",
            "Cigna submission.",
            {"payer": "cigna"}),
        ("humana",
            "Humana submission.",
            {"payer": "humana"}),
        ("medicare",
            "Medicare submission.",
            {"payer": "medicare"}),
        ("medicaid",
            "Medicaid submission.",
            {"payer": "medicaid"}),
    ]),
    (DENIAL_APPEAL_PIPELINE, [
        ("uhc_first_level",
            "UnitedHealthcare, first-level appeal.",
            {"starting_level": "internal_first_level"}),
        ("anthem_first_level",
            "Anthem, first-level appeal.",
            {"starting_level": "internal_first_level"}),
        ("aetna_first_level",
            "Aetna, first-level appeal.",
            {"starting_level": "internal_first_level"}),
        ("uhc_second_level",
            "UnitedHealthcare, second-level appeal.",
            {"starting_level": "internal_second_level"}),
        ("medicare_external",
            "Medicare external review.",
            {"starting_level": "external_independent_review"}),
    ]),
    (HEDIS_QUALITY_IMPROVEMENT, [
        ("ma_50k",
            "MA contract, 50k members.",
            {"contract_size_thousand_members": 50.0}),
        ("ma_100k",
            "MA contract, 100k members.",
            {"contract_size_thousand_members": 100.0}),
        ("ma_500k",
            "MA contract, 500k members.",
            {"contract_size_thousand_members": 500.0}),
    ]),
    (POSTPARTUM_FOLLOWUP, [
        ("term",
            "Term postpartum (37+ weeks).",
            {"gestational_age_weeks": 39}),
        ("preterm",
            "Preterm postpartum (32-36 weeks).",
            {"gestational_age_weeks": 34}),
        ("very_preterm",
            "Very preterm postpartum (<32 weeks).",
            {"gestational_age_weeks": 30}),
    ]),
    (WELL_CHILD_VISIT, [
        ("infant_6m",
            "6-month visit.", {"age_months": 6}),
        ("toddler_18m",
            "18-month visit.", {"age_months": 18}),
        ("preschool_4y",
            "4-year visit.", {"age_months": 48}),
        ("school_age_7y",
            "7-year visit.", {"age_months": 84}),
        ("adolescent_13y",
            "13-year visit.", {"age_months": 156}),
    ]),
    (VTE_PROPHYLAXIS_REVIEW, [
        ("low_risk",
            "Low-risk patient (Caprini 0-2).",
            {"age": 40, "active_malignancy": False,
             "confined_to_bed_gt_72h": False}),
        ("moderate_risk",
            "Moderate-risk patient (Caprini 3-4).",
            {"age": 65, "active_malignancy": False,
             "confined_to_bed_gt_72h": True}),
        ("high_risk",
            "High-risk patient (Caprini 5+).",
            {"age": 75, "active_malignancy": True,
             "confined_to_bed_gt_72h": True}),
    ]),
    (BEHAVIORAL_HEALTH_STEP_DOWN, [
        ("low_risk_first_episode",
            "Low-risk, first episode.",
            {"ideation_lifetime_level": 2,
             "ideation_past_30d_level": 1,
             "behavior_lifetime_attempts": 0}),
        ("moderate_recurrent",
            "Moderate-risk, recurrent ideation.",
            {"ideation_lifetime_level": 3,
             "ideation_past_30d_level": 2,
             "behavior_lifetime_attempts": 1}),
        ("high_active_plan",
            "High-risk, active plan.",
            {"ideation_lifetime_level": 4,
             "ideation_past_30d_level": 4,
             "behavior_lifetime_attempts": 1}),
    ]),
    (DKA_MANAGEMENT, [
        ("mild",
            "Mild DKA (pH 7.25-7.30).",
            {"ph": 7.28, "bicarbonate_meq_l": 16}),
        ("moderate",
            "Moderate DKA (pH 7.0-7.24).",
            {"ph": 7.15, "bicarbonate_meq_l": 12}),
        ("severe",
            "Severe DKA (pH < 7.0).",
            {"ph": 6.95, "bicarbonate_meq_l": 8}),
    ]),
    (AKI_WORKUP, [
        ("stage1",
            "KDIGO stage 1.",
            {"creatinine_current_mg_dl": 1.6}),
        ("stage2",
            "KDIGO stage 2.",
            {"creatinine_current_mg_dl": 2.1}),
        ("stage3",
            "KDIGO stage 3.",
            {"creatinine_current_mg_dl": 4.0}),
    ]),
    (ACUTE_CHEST_PAIN, [
        ("low_heart",
            "HEART score 0-3 (low).",
            {"history_descriptor": "non_specific",
             "ecg_descriptor": "normal"}),
        ("moderate_heart",
            "HEART score 4-6 (moderate).",
            {"history_descriptor": "moderately_suspicious",
             "ecg_descriptor": "non_specific_repol"}),
        ("high_heart",
            "HEART score 7-10 (high).",
            {"history_descriptor": "highly_suspicious",
             "ecg_descriptor": "significant_st_depression"}),
    ]),
    (TRAUMA_RESUS_DECISION, [
        ("blunt_low",
            "Blunt low-energy.",
            {"glasgow_coma_score": 14, "systolic_bp": 130}),
        ("blunt_high",
            "Blunt high-energy.",
            {"glasgow_coma_score": 12, "systolic_bp": 95}),
        ("penetrating",
            "Penetrating thoracic.",
            {"glasgow_coma_score": 10, "systolic_bp": 80}),
        ("polytrauma_critical",
            "Critical polytrauma.",
            {"glasgow_coma_score": 6, "systolic_bp": 70}),
    ]),
    (COVERAGE_DETERMINATION, [
        ("uhc_check", "UnitedHealthcare coverage check.",
            {"payer": "unitedhealth"}),
        ("anthem_check", "Anthem coverage check.",
            {"payer": "anthem_bcbs"}),
        ("aetna_check", "Aetna coverage check.",
            {"payer": "aetna"}),
        ("medicare_check", "Medicare coverage check.",
            {"payer": "medicare"}),
        ("medicaid_check", "Medicaid coverage check.",
            {"payer": "medicaid"}),
    ]),
    (ONCOLOGY_CYCLE_REVIEW, [
        ("cycle_1", "Cycle 1.", {"cycle_number": 1}),
        ("cycle_2", "Cycle 2.", {"cycle_number": 2}),
        ("cycle_3", "Cycle 3.", {"cycle_number": 3}),
        ("cycle_4", "Cycle 4.", {"cycle_number": 4}),
        ("cycle_6", "Cycle 6.", {"cycle_number": 6}),
    ]),
    (OUT_OF_NETWORK_REFERRAL, [
        ("uhc_oon", "UnitedHealthcare OON request.",
            {"payer": "unitedhealth"}),
        ("anthem_oon", "Anthem OON request.",
            {"payer": "anthem_bcbs"}),
        ("aetna_oon", "Aetna OON request.",
            {"payer": "aetna"}),
        ("cigna_oon", "Cigna OON request.",
            {"payer": "cigna"}),
        ("humana_oon", "Humana OON request.",
            {"payer": "humana"}),
    ]),
    (POPULATION_OUTREACH, [
        ("phone", "Phone outreach.",
            {"outreach_channel": "phone"}),
        ("sms", "SMS outreach.",
            {"outreach_channel": "sms"}),
        ("email", "Email outreach.",
            {"outreach_channel": "email"}),
        ("postcard", "Postcard outreach.",
            {"outreach_channel": "postcard"}),
        ("ehr_portal", "EHR portal outreach.",
            {"outreach_channel": "ehr_portal"}),
    ]),
    (VACCINE_SCHEDULE_CHECK, [
        ("flu", "Flu reminders.",
            {"overdue_by_vaccine": {"FLU": 2400}}),
        ("pneumo", "Pneumococcal reminders.",
            {"overdue_by_vaccine": {"PNEUMO": 380}}),
        ("zoster", "Zoster reminders.",
            {"overdue_by_vaccine": {"ZOSTER": 95}}),
        ("rsv", "RSV reminders.",
            {"overdue_by_vaccine": {"RSV": 410}}),
        ("hpv", "HPV reminders.",
            {"overdue_by_vaccine": {"HPV": 740}}),
    ]),
    (CARE_GAP_CLOSURE_PIPELINE, [
        ("bcs", "Breast cancer screening (BCS).",
            {"patient_question": "Why do I need a mammogram?"}),
        ("col", "Colorectal cancer screening (COL).",
            {"patient_question": "When should I have a colonoscopy?"}),
        ("cdc_hba1c", "Diabetes A1C control (CDC-HBA1C).",
            {"patient_question":
                "Why is my A1C target lower this year?"}),
        ("cbp", "Blood-pressure control (CBP).",
            {"patient_question":
                "What blood-pressure number am I targeting?"}),
        ("supd", "Statin use in diabetes (SUPD).",
            {"patient_question":
                "Why is my doctor asking me to take a statin?"}),
    ]),
    (CONSULT_LETTER_WORKFLOW, [
        ("cardiology", "Cardiology consult.",
            {"consultant_specialty": "cardiology"}),
        ("pulmonology", "Pulmonology consult.",
            {"consultant_specialty": "pulmonology"}),
        ("nephrology", "Nephrology consult.",
            {"consultant_specialty": "nephrology"}),
        ("gastroenterology", "Gastroenterology consult.",
            {"consultant_specialty": "gastroenterology"}),
        ("hematology_oncology", "Heme/onc consult.",
            {"consultant_specialty": "hematology_oncology"}),
        ("infectious_diseases", "ID consult.",
            {"consultant_specialty": "infectious_diseases"}),
        ("neurology", "Neurology consult.",
            {"consultant_specialty": "neurology"}),
        ("rheumatology", "Rheumatology consult.",
            {"consultant_specialty": "rheumatology"}),
    ]),
    (DISCHARGE_SUMMARY_WORKFLOW, [
        ("home", "Discharge to home.",
            {"discharge_disposition": "home"}),
        ("home_with_care", "Discharge to home with services.",
            {"discharge_disposition": "home_with_care"}),
        ("snf", "Discharge to SNF.",
            {"discharge_disposition": "snf"}),
        ("rehab", "Discharge to inpatient rehab.",
            {"discharge_disposition": "rehab"}),
        ("hospice", "Discharge to hospice.",
            {"discharge_disposition": "hospice"}),
    ]),
    (PROGRESS_NOTE_WORKFLOW, [
        ("medicine_floor", "Medicine floor SOAP.",
            {"subjective_text":
                "Day 2 of admission, hospital course progressing."}),
        ("icu_progress", "ICU progress note.",
            {"subjective_text":
                "ICU day 3, weaning ventilator support."}),
        ("surgery_pod1", "Surgery POD-1 note.",
            {"subjective_text":
                "POD-1 after laparoscopic cholecystectomy, "
                "tolerating clears."}),
        ("peds_progress", "Pediatric progress note.",
            {"subjective_text":
                "5-year-old recovering from RSV bronchiolitis."}),
    ]),
    (INPATIENT_DETERIORATION_RESPONSE, [
        ("low_alert",
            "Low-tier alert.",
            {"news2_score": 3}),
        ("medium_alert",
            "Medium-tier alert.",
            {"news2_score": 5}),
        ("high_alert",
            "High-tier alert.",
            {"news2_score": 7}),
    ]),
    (TRANSITIONAL_CARE_MANAGEMENT, [
        ("low_acuity",
            "Low-acuity TCM.",
            {"acuity_band": "low"}),
        ("moderate_acuity",
            "Moderate-acuity TCM.",
            {"acuity_band": "moderate"}),
        ("high_acuity",
            "High-acuity TCM.",
            {"acuity_band": "high"}),
    ]),
    (ANTICOAGULANT_REVIEW, [
        ("warfarin",
            "Warfarin review.",
            {"current_medications":
                [{"name": "warfarin", "dose": "5 mg"}]}),
        ("apixaban",
            "Apixaban review.",
            {"current_medications":
                [{"name": "apixaban", "dose": "5 mg BID"}]}),
        ("rivaroxaban",
            "Rivaroxaban review.",
            {"current_medications":
                [{"name": "rivaroxaban", "dose": "20 mg daily"}]}),
        ("dabigatran",
            "Dabigatran review.",
            {"current_medications":
                [{"name": "dabigatran", "dose": "150 mg BID"}]}),
        ("edoxaban",
            "Edoxaban review.",
            {"current_medications":
                [{"name": "edoxaban", "dose": "60 mg daily"}]}),
    ]),
    (ANTIBIOTIC_STEWARDSHIP, [
        ("uti", "UTI stewardship.",
            {"infection_source": "urinary"}),
        ("pneumonia", "Pneumonia stewardship.",
            {"infection_source": "pneumonia"}),
        ("skin", "Skin/soft-tissue stewardship.",
            {"infection_source": "skin"}),
        ("intra_abdominal", "Intra-abdominal stewardship.",
            {"infection_source": "intra_abdominal"}),
        ("bloodstream", "Bloodstream stewardship.",
            {"infection_source": "bloodstream"}),
    ]),
    (PERIOP_RISK_STRATIFICATION, [
        ("ambulatory_low",
            "Ambulatory, low-risk procedure.",
            {"procedure_class": "ambulatory_minor"}),
        ("inpatient_moderate",
            "Inpatient, moderate-risk procedure.",
            {"procedure_class": "inpatient_moderate"}),
        ("inpatient_major",
            "Inpatient, major procedure.",
            {"procedure_class": "inpatient_major"}),
        ("emergency",
            "Emergency procedure.",
            {"procedure_class": "emergency"}),
    ]),
    (CHRONIC_DISEASE_FOLLOWUP, [
        ("diabetes_type2",
            "Type 2 diabetes followup.",
            {"chronic_condition": "diabetes_type2"}),
        ("hf",
            "Heart-failure followup.",
            {"chronic_condition": "heart_failure"}),
        ("copd",
            "COPD followup.",
            {"chronic_condition": "copd"}),
        ("ckd",
            "CKD followup.",
            {"chronic_condition": "chronic_kidney_disease"}),
        ("hypertension",
            "Hypertension followup.",
            {"chronic_condition": "hypertension"}),
    ]),
    (PALLIATIVE_TRANSITION, [
        ("early_palliative",
            "Early palliative.",
            {"prognosis_band": "months_to_years"}),
        ("late_palliative",
            "Late palliative.",
            {"prognosis_band": "weeks_to_months"}),
        ("hospice_eligible",
            "Hospice-eligible.",
            {"prognosis_band": "weeks"}),
        ("imminent",
            "Imminent end-of-life.",
            {"prognosis_band": "days"}),
    ]),
    (POST_DISCHARGE_FOLLOWUP, [
        ("day_2", "Day-2 phone call.",
            {"days_post_discharge": 2}),
        ("day_7", "Day-7 phone call.",
            {"days_post_discharge": 7}),
        ("day_14", "Day-14 phone call.",
            {"days_post_discharge": 14}),
        ("day_30", "Day-30 readmission check.",
            {"days_post_discharge": 30}),
    ]),
    (OPIOID_SAFETY_REVIEW, [
        ("chronic_low",
            "Chronic low-dose opioid review.",
            {"current_medications":
                [{"name": "tramadol", "dose": "50 mg q6h"}]}),
        ("chronic_high",
            "Chronic high-dose opioid review.",
            {"current_medications":
                [{"name": "morphine ER", "dose": "60 mg BID"}]}),
        ("oxycodone_benzo",
            "Oxycodone + benzo concomitant review.",
            {"current_medications": [
                {"name": "oxycodone", "dose": "10 mg q6h"},
                {"name": "lorazepam", "dose": "1 mg TID"},
            ]}),
        ("palliative_opioid",
            "Palliative opioid review.",
            {"current_medications":
                [{"name": "fentanyl_patch", "dose": "25 mcg/h"}]}),
    ]),
    (PEDIATRIC_DOSING_REVIEW, [
        ("amoxicillin",
            "Amoxicillin pediatric dosing.",
            {"current_medications":
                [{"name": "amoxicillin", "dose": "tbd"}]}),
        ("ibuprofen",
            "Ibuprofen pediatric dosing.",
            {"current_medications":
                [{"name": "ibuprofen", "dose": "tbd"}]}),
        ("ondansetron",
            "Ondansetron pediatric dosing.",
            {"current_medications":
                [{"name": "ondansetron", "dose": "tbd"}]}),
        ("prednisolone",
            "Prednisolone pediatric dosing.",
            {"current_medications":
                [{"name": "prednisolone", "dose": "tbd"}]}),
    ]),
]


_PARAMETRIC_WORKFLOWS: list[Workflow] = [
    _make_parametric_variant(base, suffix, desc, overrides)
    for base, axis in _PARAMETRIC_AXES
    for (suffix, desc, overrides) in axis
]


# ─────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────

_BASE_WORKFLOWS: list[Workflow] = [
        CHF_ADMISSION,
        SEPSIS_WORKUP,
        DISCHARGE_PLANNING,
        OUTPATIENT_MED_REVIEW,
        STROKE_ALERT,
        TRAUMA_RESUS_DECISION,
        ACUTE_CHEST_PAIN,
        DKA_MANAGEMENT,
        AKI_WORKUP,
        RESPIRATORY_FAILURE_WORKUP,
        INPATIENT_DETERIORATION_RESPONSE,
        ICU_STEP_DOWN,
        INPATIENT_GLYCEMIC_CONTROL_WORKFLOW,
        PERIOP_COMPLICATIONS_RESPONSE,
        INPATIENT_FALLS_INTERVENTION,
        POST_DISCHARGE_FOLLOWUP,
        CAREGIVER_HANDOFF_PREP,
        TRANSITIONAL_CARE_MANAGEMENT,
        CHRONIC_DISEASE_FOLLOWUP,
        PALLIATIVE_TRANSITION,
        PGX_PRESCRIBING_CHECK,
        DDI_AUDIT,
        ANTICOAGULANT_REVIEW,
        ANTIBIOTIC_STEWARDSHIP,
        OPIOID_SAFETY_REVIEW,
        PEDIATRIC_DOSING_REVIEW,
        PEDIATRIC_ACUTE_WORKUP,
        MENTAL_HEALTH_CRISIS,
        MATERNAL_OBSTETRIC_EMERGENCY,
        ONCOLOGY_CYCLE_REVIEW,
        GERIATRIC_ASSESSMENT,
        RHEUMATOLOGY_FOLLOWUP,
        TRANSPLANT_IMMUNOSUPPRESSION_REVIEW,
        INFECTIOUS_DISEASE_CONSULT,
        PRE_OP_OPTIMIZATION,
        PERIOP_RISK_STRATIFICATION,
        POST_OP_RECOVERY,
        SURGICAL_CONSENT_WORKUP,
        CHART_TO_CODES,
        CLINICAL_DOCUMENTATION_POLISH,
        PROGRESS_NOTE_WORKFLOW,
        CONSULT_LETTER_WORKFLOW,
        DISCHARGE_SUMMARY_WORKFLOW,
        PRIOR_AUTH_PIPELINE,
        DENIAL_APPEAL_PIPELINE,
        COST_EFFECTIVENESS_REVIEW,
        COVERAGE_DETERMINATION,
        POPULATION_OUTREACH,
        CARE_GAP_CLOSURE_PIPELINE,
        HEDIS_QUALITY_IMPROVEMENT,
        # Vertical J — horizontal expansion (J51..J58)
        PRE_HOSPITAL_HANDOFF,
        VTE_PROPHYLAXIS_REVIEW,
        POSTPARTUM_FOLLOWUP,
        BEHAVIORAL_HEALTH_STEP_DOWN,
        VACCINE_SCHEDULE_CHECK,
        WELL_CHILD_VISIT,
        SSI_PREVENTION_BUNDLE,
        OUT_OF_NETWORK_REFERRAL,
]


REGISTRY: dict[str, Workflow] = {
    w.id: w for w in (
        _BASE_WORKFLOWS + _MACRO_WORKFLOWS + _PARAMETRIC_WORKFLOWS
    )
}


def list_workflows() -> list[dict]:
    """Marketplace-style listing of available workflows."""
    return [
        {
            "id": w.id,
            "title": w.title,
            "description": w.description,
            "n_steps": len(w.steps),
            "specialists_consulted": sorted(
                {s.specialist for s in w.steps}
            ),
            "required_inputs": list(w.required_inputs),
        }
        for w in REGISTRY.values()
    ]
