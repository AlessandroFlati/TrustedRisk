"""TrustedRisk MCP tools -- registered into FastMCP and grouped by clinical bundle.

The 69 tools are grouped into 23 thematic bundles to aid agent discovery and
client-side filtering. Each bundle is a curated subset of tool function names
that an upstream A2A client can request via MCP capability negotiation.

Bundles (see `BUNDLES` below):
    Clinical-vertical (12):
        core_discharge / ed_acute / pediatric / mental_health / antimicrobial /
        oncology / stroke_acs / obstetric_geriatric / trauma_critical /
        endocrine_acute / imaging / nephrology
    Cross-cutting (8):
        economics / context_resolution / diagnosis / patient_facing /
        data_normalization / clinical_workflow / external_knowledge /
        chart_intelligence
    Phase 2 pain-point (3):
        prior_authorization (PA-1/2/3/4)        -- Phase 2.1
        clinical_documentation (SCRIBE-1/2/3/4) -- Phase 2.2
        patient_qa (PATIENT-Q1/2/3)             -- Phase 2.3

Note: tools generally useful (ground_claim, fairness_audit, lab_trend_analysis,
detect_phi) appear in multiple bundles.
"""

from . import (
    acs_disposition_decision,
    active_meds_resolver,
    fetch_patient_documents,
    admission_hnp_draft,
    admission_triage,
    aki_kdigo_stage,
    auto_coding,
    antibiotic_de_escalation,
    cardiology_depth,
    care_gap_detector,
    caregiver_handoff,
    caregiver_summary,
    causal_inference,
    charlson_elixhauser,
    chart_intelligence,
    chemo_dose_adjustment,
    clinical_deterioration_score,
    consult_letter_draft,
    contrast_safety_check,
    conversational_resolver,
    counterfactual_explanation,
    critical_care,
    decision_utility,
    delirium_screening_cam,
    detect_phi,
    dialysis_initiation_decision,
    differential_diagnosis_ranker,
    discharge_counseling,
    discharge_qa,
    discharge_summary_draft,
    dka_severity,
    empiric_antibiotic_selection,
    endocrinology_advanced,
    expected_value_of_intervention,
    fairness_advanced_tool,
    fairness_audit,
    falls_risk_morse,
    fhir_writeback,
    gi_hepatology_depth,
    ground_claim,
    heart_score,
    heme_onc_depth,
    imaging_appropriateness,
    infectious_disease,
    inpatient_glycemic_control,
    insurance_appeals,
    knowledge_integration,
    legacy_ehr_parsers,
    multimodal,
    neurology_depth,
    ob_peds_advanced,
    peri_op_risk,
    rheumatology,
    lab_trend_analysis,
    loinc_normalizer,
    massive_transfusion_protocol,
    maternal_early_warning,
    medication_adherence,
    medication_reconciliation,
    medication_what_if,
    model_research,
    n_of_1_trial_design,
    oncology_treatment_response,
    order_set_generator,
    pa_appeal_likelihood,
    pa_evidence_pack,
    pa_letter_draft,
    pa_payer_rules_match,
    patient_faq,
    pgx,
    preadmit_triage,
    progress_note_draft,
    prom_influence,
    pediatric_early_warning,
    polypharmacy_concerns,
    population_health,
    preeclampsia_assessment,
    psychiatric_admission_decision,
    quality_stars,
    readmission_risk,
    rxnorm_ddi,
    sleep_pain,
    specialty_clinics,
    stroke_severity,
    stroke_thrombolysis_eligibility,
    suicide_risk_assessment,
    translate_discharge_counseling,
    transplant,
    trauma_severity_score,
    treatment_selection,
    umls_mapper,
    weight_based_dosing,
)

__all__ = [
    "readmission_risk",
    "decision_utility",
    "ground_claim",
    "detect_phi",
    "medication_reconciliation",
    "polypharmacy_concerns",
    "fairness_audit",
    "counterfactual_explanation",
    "lab_trend_analysis",
    "discharge_counseling",
    "clinical_deterioration_score",
    "admission_triage",
    "treatment_selection",
    "pediatric_early_warning",
    "weight_based_dosing",
    "suicide_risk_assessment",
    "psychiatric_admission_decision",
    "empiric_antibiotic_selection",
    "antibiotic_de_escalation",
    "chemo_dose_adjustment",
    "oncology_treatment_response",
    "stroke_severity",
    "stroke_thrombolysis_eligibility",
    "heart_score",
    "acs_disposition_decision",
    "maternal_early_warning",
    "preeclampsia_assessment",
    "falls_risk_morse",
    "delirium_screening_cam",
    "trauma_severity_score",
    "massive_transfusion_protocol",
    "dka_severity",
    "inpatient_glycemic_control",
    "imaging_appropriateness",
    "contrast_safety_check",
    "aki_kdigo_stage",
    "dialysis_initiation_decision",
    "expected_value_of_intervention",
    "conversational_resolver",
    "differential_diagnosis_ranker",
    "translate_discharge_counseling",
    "patient_faq",
    "caregiver_summary",
    "charlson_elixhauser",
    "loinc_normalizer",
    "rxnorm_ddi",
    "umls_mapper",
    "order_set_generator",
    "medication_adherence",
    "care_gap_detector",
    "prom_influence",
    "knowledge_integration",
    "chart_intelligence",
    "pa_evidence_pack",
    "pa_payer_rules_match",
    "pa_appeal_likelihood",
    "pa_letter_draft",
    "progress_note_draft",
    "discharge_summary_draft",
    "consult_letter_draft",
    "admission_hnp_draft",
    "discharge_qa",
    "medication_what_if",
    "caregiver_handoff",
    "auto_coding",
    "pgx",
    "preadmit_triage",
    "n_of_1_trial_design",
    "quality_stars",
    "population_health",
    "insurance_appeals",
    "multimodal",
    "causal_inference",
    "fairness_advanced_tool",
    "critical_care",
    "specialty_clinics",
    "cardiology_depth",
    "heme_onc_depth",
    "endocrinology_advanced",
    "sleep_pain",
    "transplant",
    "fhir_writeback",
    "rheumatology",
    "peri_op_risk",
    "infectious_disease",
    "gi_hepatology_depth",
    "neurology_depth",
    "ob_peds_advanced",
    "model_research",
    "legacy_ehr_parsers",
]


# ─────────────────────────────────────────────────────────────────────
# Thematic bundles
# ─────────────────────────────────────────────────────────────────────
#
# Each entry maps a bundle id to the list of tool function names that the
# bundle exposes. The agent or an upstream client can request a bundle via
# the agent-card's skill capability declaration. Tools may appear in
# multiple bundles when generally useful (e.g., fairness_audit).

BUNDLES: dict[str, list[str]] = {
    "core_discharge": [
        "compute_readmission_risk",
        "compute_decision_utility",
        "ground_claim",
        "detect_phi",
        "compute_medication_reconciliation",
        "detect_polypharmacy_concerns",
        "compute_fairness_audit",
        "compute_counterfactual_explanation",
        "compute_lab_trend_analysis",
        "compute_discharge_counseling",
    ],
    "ed_acute": [
        "compute_admission_triage",
        "compute_clinical_deterioration_score",
        "detect_phi",
        "ground_claim",
        "compute_lab_trend_analysis",
    ],
    "pediatric": [
        "compute_pediatric_early_warning",
        "compute_weight_based_dosing",
    ],
    "mental_health": [
        "compute_suicide_risk_assessment",
        "compute_psychiatric_admission_decision",
        "compute_fairness_audit",   # bias guard pivotal here
        "detect_phi",
    ],
    "antimicrobial": [
        "compute_empiric_antibiotic_selection",
        "compute_antibiotic_de_escalation",
        "ground_claim",
    ],
    "oncology": [
        "compute_chemo_dose_adjustment",
        "compute_oncology_treatment_response",
        "compute_treatment_selection",
        "compute_lab_trend_analysis",
    ],
    "stroke_acs": [
        "compute_stroke_severity",
        "compute_stroke_thrombolysis_eligibility",
        "compute_heart_score",
        "compute_acs_disposition_decision",
        "ground_claim",
    ],
    "obstetric_geriatric": [
        "compute_maternal_early_warning",
        "compute_preeclampsia_assessment",
        "compute_falls_risk_morse",
        "compute_delirium_screening_cam",
        "compute_fairness_audit",
    ],
    "trauma_critical": [
        "compute_trauma_severity_score",
        "compute_massive_transfusion_protocol",
        "compute_imaging_appropriateness",   # blunt abdominal trauma scenario
    ],
    "endocrine_acute": [
        "compute_dka_severity",
        "compute_inpatient_glycemic_control",
    ],
    "imaging": [
        "compute_imaging_appropriateness",
        "compute_contrast_safety_check",
        "compute_aki_kdigo_stage",   # eGFR drives contrast safety
    ],
    "nephrology": [
        "compute_aki_kdigo_stage",
        "compute_dialysis_initiation_decision",
        "compute_lab_trend_analysis",   # creatinine trends
        "compute_contrast_safety_check",
    ],
    "economics": [
        "compute_expected_value_of_intervention",
        "compute_readmission_risk",   # baseline-risk feed
        "compute_fairness_audit",     # subgroup decomposition feeds CEA
        "compute_fairness_advanced",  # EOO/DP advanced audit (Phase 13.2)
    ],
    "context_resolution": [
        "compute_resolve_patient_from_query",   # LLM-3 conversational resolver
        "detect_phi",                           # PHI scrubbing on free text
        "ground_claim",                         # claim grounding for free text
    ],
    "diagnosis": [
        "compute_differential_diagnosis_ranker",  # LLM-4 grounded DDx ranker
        "ground_claim",                           # for citation grounding
        "compute_admission_triage",               # presents to a similar surface
    ],
    "patient_facing": [
        "compute_discharge_counseling",
        "compute_translate_discharge_counseling",  # PATIENT-1
        "compute_patient_faq",                       # PATIENT-2
        "compute_caregiver_summary",                 # PATIENT-3
    ],
    "data_normalization": [
        "compute_charlson_elixhauser_index",     # DATA-4
        "compute_normalize_observations",         # DATA-3 LOINC
        "compute_rxnorm_ddi_lookup",              # DATA-2 RxNorm DDI
        "compute_umls_concept_map",               # DATA-1 UMLS crosswalk
    ],
    "clinical_workflow": [
        "compute_order_set",                      # LIB-1
        "compute_medication_adherence_predictor", # LIB-2
        "compute_care_gap_detector",              # LIB-3
        "compute_prom_influence",                 # LIB-4
    ],
    "external_knowledge": [
        "compute_pubmed_search",                  # KNOWLEDGE-1
        "compute_clinical_trials_matcher",        # KNOWLEDGE-2
        "compute_drug_pricing",                   # KNOWLEDGE-3
        "compute_nih_reporter_search",            # KNOWLEDGE-4
    ],
    "chart_intelligence": [
        "compute_clinical_ner",                    # CHART-1
        "compute_negation_temporal",               # CHART-2
        "compute_structure_discharge_summary",     # CHART-3
    ],
    "prior_authorization": [
        "compute_pa_evidence_pack",                # PA-1
        "compute_pa_payer_rules_match",            # PA-2
        "compute_pa_appeal_likelihood",            # PA-3
        "compute_pa_letter_draft",                 # PA-4
    ],
    "clinical_documentation": [
        "compute_progress_note_draft",             # SCRIBE-1
        "compute_discharge_summary_draft",         # SCRIBE-2
        "compute_consult_letter_draft",            # SCRIBE-3
        "compute_admission_hnp_draft",             # SCRIBE-4
    ],
    "patient_qa": [
        "compute_discharge_qa",                    # PATIENT-Q1
        "compute_medication_what_if",              # PATIENT-Q2
        "compute_caregiver_handoff",               # PATIENT-Q3
    ],
    "auto_coding": [
        "compute_icd10_suggest",                   # CODE-1
        "compute_cpt_suggest",                     # CODE-2
        "compute_hcpcs_suggest",                   # CODE-3
        "compute_coding_audit",                    # CODE-4
    ],
    "pharmacogenomics": [
        "compute_pgx_dose_adjustment",             # PGX-1
        "compute_pgx_drug_alternatives",           # PGX-2
        "compute_pgx_eligibility_check",           # PGX-3
    ],
    "preadmit_triage": [
        "compute_symptom_red_flag_check",          # PREADMIT-1
        "compute_when_to_seek_care",               # PREADMIT-2
        "compute_symptom_followup_questions",      # PREADMIT-3
    ],
    "research_design": [
        "compute_n_of_1_trial_design",             # NOFONE-1
        "compute_causal_treatment_effect",         # CAUSAL-1 (Phase 12.6)
    ],
    "quality_stars": [
        "compute_quality_measures_aggregate",      # STARS-1
        "compute_stars_rating_forecast",           # STARS-2
        "compute_care_gap_priority_ranking",       # STARS-3
    ],
    "population_health": [
        "compute_syndromic_surveillance",          # POPHEALTH-1
        "compute_vaccine_reminder_cohort",         # POPHEALTH-2
        "compute_outbreak_heatmap",                # POPHEALTH-3
    ],
    "insurance_appeals": [
        "compute_denial_letter_parse",             # APPEALS-1
        "compute_appeal_letter_draft",             # APPEALS-2
        "compute_appeal_escalation_path",          # APPEALS-3
    ],
    "multimodal": [
        "compute_ecg_qt_analyzer",                 # MULTIMODAL-1
        "compute_dicom_sr_ingest",                 # MULTIMODAL-2
    ],
    "critical_care": [
        "compute_apache_ii_score",                 # CRIT-1
        "compute_sofa_score",                      # CRIT-2
        "compute_meld_score",                      # CRIT-3
        "compute_rifle_aki_classification",        # CRIT-4
    ],
    "specialty_clinics": [
        "compute_lesion_triage",                   # DERM-1
        "compute_diabetic_retinopathy_severity",   # OPHTHO-1
        "compute_pft_interpretation",              # PULMO-1
    ],
    "cardiology_depth": [
        "compute_cha2ds2_vasc",                    # CARDIO-1
        "compute_has_bled",                        # CARDIO-2
        "compute_timi_acs_score",                  # CARDIO-3
        "compute_grace_acs_score",                 # CARDIO-4
    ],
    "heme_onc_depth": [
        "compute_iss_myeloma_staging",             # HEME-1
        "compute_ipss_r_mds_score",                # HEME-2
        "compute_ecog_performance_status",         # HEME-3
        "compute_karnofsky_performance",           # HEME-4
    ],
    "endocrinology_advanced": [
        "compute_thyroid_management",              # ENDO-1
        "compute_adrenal_insufficiency_workup",    # ENDO-2
        "compute_hypocalcemia_severity",           # ENDO-3
    ],
    "sleep_pain": [
        "compute_epworth_sleepiness_scale",        # SLEEP-1
        "compute_stop_bang_osa_screen",            # SLEEP-2
        "compute_dn4_neuropathic_pain",            # PAIN-1
    ],
    "transplant": [
        "compute_kdpi_kidney_donor",               # TX-1
        "compute_epts_recipient_score",            # TX-2
        "compute_immunosuppression_dose_check",    # TX-3
    ],
    "fhir_writeback": [
        "compute_write_decision_to_fhir",          # WB-1
    ],
    "rheumatology": [
        "compute_das28_rheumatoid_arthritis",      # RHEUM-1
        "compute_asdas_axspa",                     # RHEUM-2
        "compute_acr_eular_ra_classification",     # RHEUM-3
    ],
    "peri_op_risk": [
        "compute_rcri_cardiac_risk",               # SURG-1
        "compute_ariscat_pulmonary_risk",          # SURG-2
        "compute_caprini_vte_risk",                # SURG-3
    ],
    "infectious_disease": [
        "compute_qsofa_score",                     # ID-1
        "compute_lactate_clearance",               # ID-2
        "compute_hiv_management_tier",             # ID-3
        "compute_tb_risk_screen",                  # ID-4
    ],
    "gi_hepatology_depth": [
        "compute_maddrey_alcoholic_hepatitis",     # GI-1
        "compute_fib4_liver_fibrosis",             # GI-2
        "compute_glasgow_blatchford_ugib",         # GI-3
        "compute_rome_iv_ibs",                     # GI-4
    ],
    "neurology_depth": [
        "compute_hunt_hess_sah",                   # NEURO-1
        "compute_ich_score",                       # NEURO-2
        "compute_modified_rankin",                 # NEURO-3
        "compute_hauser_ambulation_index",         # NEURO-4
    ],
    "ob_peds_advanced": [
        "compute_bishop_induction_score",          # OBPED-1
        "compute_apgar_score",                     # OBPED-2
        "compute_bell_nec_stage",                  # OBPED-3
        "compute_bilirubin_nomogram",              # OBPED-4
    ],
    "model_research": [
        "compute_cox_proportional_hazards",        # MR-1
        "compute_shap_attribution",                # MR-2
        "compute_ensemble_stacking",               # MR-3
    ],
    "legacy_ehr_parsers": [
        "compute_hl7v2_message_parse",             # LEGACY-1
        "compute_ccda_document_parse",             # LEGACY-2
    ],
}


def list_bundles() -> dict[str, list[str]]:
    """Return the bundle map (read-only convenience for agent-card builders)."""
    return {k: list(v) for k, v in BUNDLES.items()}


def register_all(mcp) -> None:
    """Register all tools onto the shared FastMCP instance."""
    active_meds_resolver.register(mcp)
    fetch_patient_documents.register(mcp)
    readmission_risk.register(mcp)
    decision_utility.register(mcp)
    ground_claim.register(mcp)
    detect_phi.register(mcp)
    medication_reconciliation.register(mcp)
    polypharmacy_concerns.register(mcp)
    fairness_audit.register(mcp)
    counterfactual_explanation.register(mcp)
    lab_trend_analysis.register(mcp)
    discharge_counseling.register(mcp)
    clinical_deterioration_score.register(mcp)
    admission_triage.register(mcp)
    treatment_selection.register(mcp)
    pediatric_early_warning.register(mcp)
    weight_based_dosing.register(mcp)
    suicide_risk_assessment.register(mcp)
    psychiatric_admission_decision.register(mcp)
    empiric_antibiotic_selection.register(mcp)
    antibiotic_de_escalation.register(mcp)
    chemo_dose_adjustment.register(mcp)
    oncology_treatment_response.register(mcp)
    stroke_severity.register(mcp)
    stroke_thrombolysis_eligibility.register(mcp)
    heart_score.register(mcp)
    acs_disposition_decision.register(mcp)
    maternal_early_warning.register(mcp)
    preeclampsia_assessment.register(mcp)
    falls_risk_morse.register(mcp)
    delirium_screening_cam.register(mcp)
    trauma_severity_score.register(mcp)
    massive_transfusion_protocol.register(mcp)
    dka_severity.register(mcp)
    inpatient_glycemic_control.register(mcp)
    imaging_appropriateness.register(mcp)
    contrast_safety_check.register(mcp)
    aki_kdigo_stage.register(mcp)
    dialysis_initiation_decision.register(mcp)
    expected_value_of_intervention.register(mcp)
    conversational_resolver.register(mcp)
    differential_diagnosis_ranker.register(mcp)
    translate_discharge_counseling.register(mcp)
    patient_faq.register(mcp)
    caregiver_summary.register(mcp)
    charlson_elixhauser.register(mcp)
    loinc_normalizer.register(mcp)
    rxnorm_ddi.register(mcp)
    umls_mapper.register(mcp)
    order_set_generator.register(mcp)
    medication_adherence.register(mcp)
    care_gap_detector.register(mcp)
    prom_influence.register(mcp)
    knowledge_integration.register(mcp)
    chart_intelligence.register(mcp)
    pa_evidence_pack.register(mcp)
    pa_payer_rules_match.register(mcp)
    pa_appeal_likelihood.register(mcp)
    pa_letter_draft.register(mcp)
    progress_note_draft.register(mcp)
    discharge_summary_draft.register(mcp)
    consult_letter_draft.register(mcp)
    admission_hnp_draft.register(mcp)
    discharge_qa.register(mcp)
    medication_what_if.register(mcp)
    caregiver_handoff.register(mcp)
    auto_coding.register(mcp)
    pgx.register(mcp)
    preadmit_triage.register(mcp)
    n_of_1_trial_design.register(mcp)
    quality_stars.register(mcp)
    population_health.register(mcp)
    insurance_appeals.register(mcp)
    multimodal.register(mcp)
    causal_inference.register(mcp)
    fairness_advanced_tool.register(mcp)
    critical_care.register(mcp)
    specialty_clinics.register(mcp)
    cardiology_depth.register(mcp)
    heme_onc_depth.register(mcp)
    endocrinology_advanced.register(mcp)
    sleep_pain.register(mcp)
    transplant.register(mcp)
    fhir_writeback.register(mcp)
    rheumatology.register(mcp)
    peri_op_risk.register(mcp)
    infectious_disease.register(mcp)
    gi_hepatology_depth.register(mcp)
    neurology_depth.register(mcp)
    ob_peds_advanced.register(mcp)
    model_research.register(mcp)
    legacy_ehr_parsers.register(mcp)

    _apply_byo_filter(mcp)


# Tools hidden from MCP `tools/list` when TRUSTEDRISK_MCP_BYO_MODE=1.
# Gemini caps function declarations at 128 per request; the full 145-tool
# surface trips the cap and PO's BYO connector hangs before any tools/call.
# These 47 long-tail scores stay importable for backend A2A flows and tests
# (the BYO backend reaches them by Python call, not MCP), only their
# protocol visibility is suppressed.
_BYO_EXCLUDED_TOOLS: frozenset[str] = frozenset({
    # Statistical / research helpers (not directly clinically actionable in chat)
    "compute_causal_treatment_effect",
    "compute_cox_proportional_hazards",
    "compute_ensemble_stacking",
    "compute_n_of_1_trial_design",
    "compute_prom_influence",
    "compute_shap_attribution",
    "compute_medication_adherence_predictor",
    # Niche specialty scores -- covered by broader siblings or rare in BYO chat
    "compute_acr_eular_ra_classification",
    "compute_apache_ii_score",            # qSOFA + SOFA cover sepsis
    "compute_apgar_score",                # neonatal niche
    "compute_asdas_axspa",                # axSpA depth
    "compute_bell_nec_stage",             # neonatal NEC
    "compute_bilirubin_nomogram",         # neonatal jaundice
    "compute_bishop_induction_score",     # L&D-specific induction score
    "compute_cha2ds2_vasc",               # AFib stroke prevention
    "compute_das28_rheumatoid_arthritis",
    "compute_diabetic_retinopathy_severity",
    "compute_dn4_neuropathic_pain",
    "compute_drug_pricing",               # economics niche
    "compute_ecog_performance_status",    # Karnofsky kept
    "compute_epts_recipient_score",
    "compute_epworth_sleepiness_scale",
    "compute_fib4_liver_fibrosis",
    "compute_glasgow_blatchford_ugib",
    "compute_grace_acs_score",            # HEART score covers ACS
    "compute_has_bled",
    "compute_hauser_ambulation_index",
    "compute_hiv_management_tier",
    "compute_hunt_hess_sah",
    "compute_hypocalcemia_severity",
    "compute_ich_score",
    "compute_immunosuppression_dose_check",
    "compute_ipss_r_mds_score",
    "compute_iss_myeloma_staging",
    "compute_kdpi_kidney_donor",
    "compute_lactate_clearance",
    "compute_lesion_triage",              # ECG QT + DICOM SR cover multimodal
    "compute_maddrey_alcoholic_hepatitis",
    "compute_modified_rankin",
    "compute_pft_interpretation",
    "compute_rifle_aki_classification",   # KDIGO kept as the active staging tool
    "compute_rome_iv_ibs",
    "compute_stop_bang_osa_screen",
    "compute_tb_risk_screen",
    "compute_thyroid_management",
    "compute_timi_acs_score",             # HEART score covers ACS
    "compute_umls_concept_map",           # LOINC + RxNorm cover normalization
})


def _apply_byo_filter(mcp) -> None:
    """Remove long-tail tools from MCP visibility when BYO mode is enabled.

    Activated by `TRUSTEDRISK_MCP_BYO_MODE=1` in the environment.
    The hidden tools remain importable from Python; only `tools/list`
    omits them so PO's Gemini connector stays under the 128
    function-declaration cap.
    """
    import os

    flag = os.environ.get("TRUSTEDRISK_MCP_BYO_MODE", "").strip().lower()
    if flag not in ("1", "true", "yes", "on"):
        return

    provider = getattr(mcp, "local_provider", None)
    remove = (
        provider.remove_tool
        if provider is not None and hasattr(provider, "remove_tool")
        else mcp.remove_tool
    )
    for name in _BYO_EXCLUDED_TOOLS:
        try:
            remove(name)
        except Exception:
            # Tool may not be registered (renamed, refactored). Skip silently;
            # the goal is best-effort trimming, not strict invariant enforcement.
            pass
