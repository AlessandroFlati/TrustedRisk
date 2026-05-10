"""Pydantic models for the 4 MCP tool inputs + outputs + Decision Card composition.

Schemas here are the contract surface -- MCP tool signatures derive from them,
the regression suite asserts against them, and the A2A agent consumes them.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────
# Version compatibility -- populated by the porting script via env vars
# ─────────────────────────────────────────────────────────────────────

SUPPORTED_MODEL_NAMES: list[str] = ["lace-plus-bayesian-v1"]


# Generic outcome identifiers -- every calibrated-risk tool MUST tag the
# RiskEstimate it produces with one of these. New outcomes are added by
# extending this Literal + providing a coefficient bundle.
OutcomeId = Literal[
    "readmission_30d",     # 30-day readmission after discharge (W1 calibrated)
    "ed_bounceback_72h",   # 72-hour ED return after ED disposition
    "deterioration_24h",   # 24-hour clinical deterioration on inpatient ward
    "sepsis_24h",          # 24-hour sepsis onset
    "mortality_30d",       # 30-day mortality (any cause)
    "surgical_complication_30d",  # 30-day post-op complication
]

SUPPORTED_COEFF_VERSIONS: list[str] = [
    # Populated at runtime from env var + verify_artifacts.py
]

SUPPORTED_ABSTAIN_POLICY_VERSIONS: list[str] = [
    # Populated at runtime
]

SUPPORTED_EMBEDDERS: list[str] = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
]


# ─────────────────────────────────────────────────────────────────────
# Core action + probability types
# ─────────────────────────────────────────────────────────────────────

class Action(str, Enum):
    """Discharge-decision action space."""
    DISCHARGE_HOME = "discharge_home"
    HOME_WITH_CARE = "home_with_care"
    SNF = "snf"
    CONTINUED_ADMISSION = "continued_admission"


class ProbInterval(BaseModel):
    """Probability estimate with 95% CI."""
    mean: float = Field(..., ge=0.0, le=1.0)
    ci95: tuple[float, float]

    @field_validator("ci95")
    @classmethod
    def ci95_ordered_and_in_unit(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError(f"CI95 must satisfy 0 <= lo <= hi <= 1, got ({lo}, {hi})")
        return v


# ─────────────────────────────────────────────────────────────────────
# 1. healthcare.compute_readmission_risk -- output: RiskEstimate
# ─────────────────────────────────────────────────────────────────────

class Factor(BaseModel):
    """Contributing factor to the readmission risk estimate."""
    name: str
    raw_value: float
    lace_points: int = Field(..., ge=0, le=7)
    weight: float = Field(..., ge=0.0, le=1.0)


class RiskEstimate(BaseModel):
    """Calibrated probability of a clinical outcome over a fixed horizon.

    Originally produced by `compute_readmission_risk` for 30-day readmission;
    `outcome_id` (added in v0.5) lets the same schema describe ED bounce-back,
    inpatient deterioration, sepsis onset, mortality, etc. -- any calibrated
    binary outcome with a CI. Default stays `readmission_30d` for backward
    compatibility with existing artifacts (coefficients.json + tests)."""
    model_name: Literal["lace-plus-bayesian-v1"] = "lace-plus-bayesian-v1"
    model_version: str
    outcome_id: OutcomeId = "readmission_30d"
    horizon_days: int = Field(..., ge=1)
    lace_raw_score: int = Field(..., ge=0, le=19)
    probability_mean: float = Field(..., ge=0.0, le=1.0)
    probability_ci95: tuple[float, float]
    probability_ci_width: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: list[Factor]
    fhir_observations_used: list[str] = Field(default_factory=list)
    computed_at: datetime
    # v0.3 addition -- confidence flag from coefficients.json degraded pass
    confidence: Literal["preferred", "degraded"] = "preferred"
    # v0.4 addition -- temporal validity window. After valid_until the runtime
    # MUST refuse to act on the cached recommendation and force a re-call.
    valid_for_minutes: int = Field(default=720, ge=0)
    valid_until: datetime | None = None
    # Phase 11.1 -- conformal prediction surface (optional). Populated when
    # data/conformal_readmission.json is present and ships a calibrated
    # threshold; remains None when only CI95 is available.
    conformal_interval_lower: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Split-conformal interval lower bound on the probability "
            "(score function = abs_residual). Marginal coverage at the "
            "calibrated 1-α target."
        ),
    )
    conformal_interval_upper: float | None = Field(
        default=None, ge=0.0, le=1.0,
    )
    conformal_prediction_set: list[Literal[0, 1]] | None = Field(
        default=None,
        description=(
            "Split-conformal binary prediction set for the realised "
            "outcome (1=event, 0=no-event) under the binary one-minus-p "
            "score. An empty set indicates the calibrated band excludes "
            "both classes -- caller should abstain."
        ),
    )
    conformal_target_coverage: float | None = Field(
        default=None, ge=0.0, le=1.0,
    )
    # Out-of-distribution / calibration-plateau abstain flag. Set when
    # the LACE bin maps onto a region of coefficients.json where the
    # training set was underpowered and the lookup table degenerates
    # to a constant -- the probability_mean shown is then the bucket
    # mean of an underpowered population rather than a calibrated
    # estimate for this patient.
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 2. healthcare.compute_decision_utility -- output: UtilityAnalysis
# ─────────────────────────────────────────────────────────────────────

class UtilityAnalysis(BaseModel):
    """Output of compute_decision_utility."""
    action_scores_qaly_weeks: dict[Action, float]
    action_scores_ci95: dict[Action, tuple[float, float]]
    action_costs_usd: dict[Action, float]
    dominant_action: Action | Literal["INCONCLUSIVE"]
    dominance_confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_trace: str


# ─────────────────────────────────────────────────────────────────────
# 3. healthcare.ground_claim -- output: ClaimGrounding
# ─────────────────────────────────────────────────────────────────────

class EvidenceSource(BaseModel):
    source_type: Literal["fhir_observation", "fhir_condition", "guideline_passage"]
    source_id: str
    excerpt: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    recency_days: int | None = None


class SubClaim(BaseModel):
    text: str
    atomic_kind: Literal["vital", "symptom", "medication", "procedure", "general"]
    verdict: Literal["supported", "partially_supported", "unsupported"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    reason_if_unsupported: str | None = None


class ClaimGrounding(BaseModel):
    """Output of ground_claim."""
    claim_text: str
    sub_claims: list[SubClaim]
    overall_verdict: Literal["supported", "partially_supported", "unsupported"]
    context_fingerprint: str
    grounded_at: datetime


# ─────────────────────────────────────────────────────────────────────
# 4. healthcare.compute_medication_reconciliation -- output: MedReconReport
# ─────────────────────────────────────────────────────────────────────

class Medication(BaseModel):
    """Minimal medication record. Maps to a FHIR MedicationRequest snapshot
    or to a structured row in an admission/discharge med-list export."""
    name: str
    dose: str | None = None
    route: str | None = None
    rxnorm_code: str | None = None
    drug_class: str | None = None  # e.g. "anticoagulant", "insulin"
    status: Literal["active", "stopped", "completed"] = "active"


class MedicationConcern(BaseModel):
    """A discharge-quality concern about a medication or med-class."""
    medication_name: str
    drug_class: str
    severity: Literal["low", "medium", "high"]
    concern_type: Literal[
        "missing_monitoring",
        "dose_change_without_followup",
        "ddi_potential",
        "polypharmacy_high_risk",
        "missing_indication",
    ]
    detail: str
    monitoring_required: list[str] = Field(default_factory=list)
    monitoring_observed: list[str] = Field(default_factory=list)


class MedReconReport(BaseModel):
    """Output of compute_medication_reconciliation."""
    added: list[Medication] = Field(default_factory=list)
    removed: list[Medication] = Field(default_factory=list)
    dose_changed: list[Medication] = Field(default_factory=list)
    concerns: list[MedicationConcern] = Field(default_factory=list)
    severity_counts: dict[str, int] = Field(default_factory=dict)
    discharge_contract_satisfied: bool
    n_admission_meds: int
    n_discharge_meds: int
    monitoring_window_hours: int
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 5. healthcare.detect_polypharmacy_concerns -- output: PolypharmacyReport
# ─────────────────────────────────────────────────────────────────────

class DrugInteraction(BaseModel):
    """A flagged drug-drug interaction pair."""
    drug_a: str
    drug_b: str
    severity: Literal["low", "medium", "high"]
    mechanism: str
    detail: str


class PolypharmacyReport(BaseModel):
    """Output of detect_polypharmacy_concerns. Stateless -- no FHIR context."""
    n_medications: int
    n_high_risk: int
    class_counts: dict[str, int] = Field(default_factory=dict)
    interactions: list[DrugInteraction] = Field(default_factory=list)
    polypharmacy_severity: Literal["none", "low", "medium", "high"]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 6. healthcare.compute_fairness_audit -- output: FairnessReport
# ─────────────────────────────────────────────────────────────────────

class SubgroupCalibration(BaseModel):
    """Per-subgroup calibration drift assessment."""
    subgroup_name: str
    subgroup_value: str
    expected_rate_baseline: float = Field(..., ge=0.0, le=1.0)
    predicted_rate_for_patient: float = Field(..., ge=0.0, le=1.0)
    relative_drift: float  # signed; (predicted - expected) / expected
    severity: Literal["none", "low", "medium", "high"]
    citation: str  # which baseline study this subgroup rate came from


class FairnessReport(BaseModel):
    """Output of compute_fairness_audit."""
    n_subgroups_assessed: int
    subgroup_drifts: list[SubgroupCalibration] = Field(default_factory=list)
    max_relative_drift: float
    confidence_action: Literal[
        "no_action",
        "flag_for_review",
        "downgrade_confidence",
        "abstain_recommended",
    ]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    audit_disclaimer: str = (
        "Fairness audit is a heuristic v1 layer using literature baselines "
        "(HRRP HCUP 2023, AHRQ subgroup analyses). Production deployments "
        "must re-calibrate against the receiving institution's empirical "
        "subgroup rates -- see the W5 calibration workflow."
    )


# ─────────────────────────────────────────────────────────────────────
# 7. healthcare.compute_counterfactual_explanation -- output: CounterfactualReport
# ─────────────────────────────────────────────────────────────────────

class FactorCounterfactual(BaseModel):
    """Per-factor counterfactual sweep: what if this factor were zero / max?"""
    factor_name: str
    current_points: int = Field(..., ge=0, le=7)
    current_raw: float | str
    if_zero_lace_total: int = Field(..., ge=0, le=19)
    if_zero_prob_mean: float = Field(..., ge=0.0, le=1.0)
    if_max_lace_total: int = Field(..., ge=0, le=19)
    if_max_prob_mean: float = Field(..., ge=0.0, le=1.0)
    delta_prob_if_zero: float  # signed: prob_if_zero - current_prob
    delta_prob_if_max: float
    modifiability: Literal["fixed", "partial", "modifiable"]
    rationale: str


class CounterfactualPath(BaseModel):
    """A minimum-modification path to flip the recommendation."""
    target_action: str
    target_lace_max: int = Field(..., ge=0, le=19)
    factors_to_change: list[str]
    cumulative_delta_points: int
    achievable: bool
    explanation: str


class CounterfactualReport(BaseModel):
    """Output of compute_counterfactual_explanation."""
    current_lace_total: int = Field(..., ge=0, le=19)
    current_prob_mean: float = Field(..., ge=0.0, le=1.0)
    factors: list[FactorCounterfactual]
    most_influential_factor: str
    flip_path_safer: CounterfactualPath | None = None
    flip_path_riskier: CounterfactualPath | None = None
    rationale: str


# ─────────────────────────────────────────────────────────────────────
# 8. healthcare.compute_lab_trend_analysis -- output: LabTrendReport
# ─────────────────────────────────────────────────────────────────────

class LabDataPoint(BaseModel):
    """Single observation in a lab time series."""
    value: float
    unit: str | None = None
    observed_at: datetime


class LabTrend(BaseModel):
    """Trend analysis for a single lab over time."""
    lab_name: str
    loinc_code: str | None = None
    n_observations: int = Field(..., ge=0)
    direction: Literal["up", "down", "flat", "insufficient_data"]
    trend_clinical: Literal[
        "improving", "declining", "flat", "insufficient_data", "borderline",
    ]
    rate_per_day: float | None = None  # slope of value vs time, in unit/day
    latest_value: float | None = None
    reference_range: tuple[float, float] | None = None
    in_normal_range: bool | None = None
    rationale: str


class LabTrendReport(BaseModel):
    """Output of compute_lab_trend_analysis."""
    patient_id: str | None
    n_labs_analyzed: int
    n_observations_total: int
    trends: list[LabTrend]
    flagged_labs_count: int  # labs with trend_clinical = declining or borderline
    summary: str


# ─────────────────────────────────────────────────────────────────────
# 9. healthcare.detect_phi -- output: PHIReport
# ─────────────────────────────────────────────────────────────────────

class PHIEntity(BaseModel):
    type: str
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    suggested_replacement: str


class PHIReport(BaseModel):
    """Output of detect_phi."""
    entities_found: list[PHIEntity]
    entity_count_by_type: dict[str, int]
    redaction_map: dict[str, str]
    risk_level: Literal["none", "low", "medium", "high"]


# ─────────────────────────────────────────────────────────────────────
# Decision Card -- composed by the A2A agent from the 4 tool outputs
# ─────────────────────────────────────────────────────────────────────

class AbstainTrigger(BaseModel):
    type: Literal[
        "confidence_interval_too_wide",
        "evidence_insufficient",
        "out_of_distribution",
    ]
    detail: str
    threshold_exceeded: dict[str, float | str] = Field(default_factory=dict)


class DecisionReasoning(BaseModel):
    risk_estimate: RiskEstimate
    utility_analysis: UtilityAnalysis


class DecisionValidation(BaseModel):
    grounding: ClaimGrounding
    phi_check: PHIReport


class ToolInvocationTrace(BaseModel):
    tool: str
    invoked_at: datetime
    duration_ms: int = Field(..., ge=0)
    status: Literal["ok", "error"]


class AuditBlock(BaseModel):
    """Audit trail embedded in every Decision Card."""
    request_id: str
    context_fingerprint: str
    tool_trace: list[ToolInvocationTrace]
    server_version: str
    agent_version: str
    model_coefficients_version: str
    timestamp: datetime
    # v0.3 additions
    clinical_disclaimer: str = (
        "TrustedRisk is a clinical decision support research prototype. It is NOT a "
        "medical device, has NOT been clinically validated, and MUST NOT be used as "
        "the sole basis for clinical decisions. Always consult qualified healthcare "
        "professionals."
    )
    fairness_disclaimer: str = (
        "Model calibration may vary by demographic subgroup. See model documentation."
    )


class Recommendation(BaseModel):
    action: Action
    confidence: Literal["high", "medium", "low", "none"]


class CritiqueDecision(BaseModel):
    """Output of a single critic sub-agent. Applied to a candidate
    DecisionCard either alone (legacy single-critic path) or as one vote
    in the multi-critic ensemble (see CriticEnsembleVerdict)."""
    verdict: Literal[
        "approved",
        "downgrade_confidence",
        "force_abstain",
        "request_replay",
    ]
    rationale: str
    abstain_trigger_to_add: AbstainTrigger | None = None
    confidence_after: Literal["high", "medium", "low", "none", "preferred", "degraded"] | None = None
    critic_role: Literal[
        "structural",        # legacy / fallback role
        "clinical_safety",   # AMB-5: clinical risk + medication safety
        "fairness",          # AMB-5: subgroup drift + bias guard
        "evidence",          # AMB-5: grounding + claim verification
        "llm_judge",         # GENAI-3: LLM-driven adversarial review
    ] = "structural"


class CriticEnsembleVerdict(BaseModel):
    """Aggregated decision from a 3-critic ensemble (clinical_safety +
    fairness + evidence). Each critic emits an independent CritiqueDecision;
    the ensemble combines them via:
      1. If ANY critic emits force_abstain -> ensemble = force_abstain.
      2. If ANY critic emits request_replay AND retries_remaining > 0 ->
         ensemble = request_replay.
      3. If ≥2 critics emit downgrade_confidence (or any combination of
         downgrade + force_abstain that isn't already (1)) -> ensemble =
         downgrade_confidence.
      4. Otherwise (majority of approvals) -> ensemble = approved.

    The `applied_verdict` is the verdict the orchestrator actually applies
    after considering retries_remaining (e.g. request_replay collapses to
    force_abstain after the budget is exhausted).
    """
    individual_verdicts: list[CritiqueDecision]
    aggregated_verdict: Literal[
        "approved", "downgrade_confidence", "force_abstain", "request_replay",
    ]
    applied_verdict: Literal[
        "approved", "downgrade_confidence", "force_abstain", "request_replay",
    ]
    retries_remaining: int = Field(default=0, ge=0, le=2)
    rationale: str       # composite rationale combining the critic notes
    n_force_abstain: int = Field(..., ge=0, le=8)
    n_downgrade: int = Field(..., ge=0, le=8)
    n_replay: int = Field(..., ge=0, le=8)
    n_approved: int = Field(..., ge=0, le=8)


class PatientTimelineEntry(BaseModel):
    """One past encounter on the patient's longitudinal timeline (AMB-5.3)."""
    encounter_id: str
    encounter_at: datetime
    encounter_type: Literal[
        "discharge", "ed_visit", "outpatient", "inpatient_review",
        "treatment_review",
    ]
    recommendation_action: str | None = None
    recommendation_confidence: str | None = None
    risk_probability_mean: float | None = None
    risk_outcome_id: str | None = None
    abstain_triggers_count: int = 0
    notes: str | None = None


class PatientTimeline(BaseModel):
    """Aggregated longitudinal view of a patient's encounters with
    TrustedRisk over time. Used by the cross-session memory layer to
    support drift detection and trend visualization."""
    patient_id: str
    n_encounters: int = Field(..., ge=0)
    first_encounter_at: datetime | None = None
    last_encounter_at: datetime | None = None
    entries: list[PatientTimelineEntry] = Field(default_factory=list)
    drift_flags: list[str] = Field(default_factory=list)


class DecisionCard(BaseModel):
    """The final output of safe_discharge_review skill."""
    recommendation: Recommendation | None
    reasoning: DecisionReasoning | None
    validation: DecisionValidation
    abstain: list[AbstainTrigger] | None
    audit: AuditBlock
    self_critique: CritiqueDecision | None = None


# ─────────────────────────────────────────────────────────────────────
# Batch endpoint -- POST /api/batch/decision-cards
# ─────────────────────────────────────────────────────────────────────
#
# Batch is a *deterministic* fast-path: it bypasses the LlmAgent and runs the
# tool functions directly per patient. The output drops the validation block
# (ground_claim/detect_phi require a free-text claim or note input that
# doesn't apply at the patient-id level) and the LLM-mediated UtilityAnalysis
# (decision_utility's qaly_weights schema is meant for LLM-constructed
# outcome_probs and isn't cleanly invokable from a RiskEstimate alone).
#
# A simple risk-threshold rule produces the recommendation; when an abstain
# trigger fires, recommendation is None and the caller is expected to
# escalate to clinician review or to the full single-patient agent.

class BatchPatientRequest(BaseModel):
    """One row in a batch request body."""
    patient_id: str
    horizon_days: int = Field(default=30, ge=1, le=365)
    demographics: dict[str, str | int | float] | None = None
    include_med_recon: bool = True
    include_fairness: bool = True


class BatchPatientResult(BaseModel):
    """Per-patient outcome of a batch run."""
    patient_id: str
    status: Literal["ok", "error"]
    error: str | None = None
    recommendation: Recommendation | None = None
    risk_estimate: RiskEstimate | None = None
    medication_reconciliation: MedReconReport | None = None
    fairness: FairnessReport | None = None
    abstain: list[AbstainTrigger] = Field(default_factory=list)
    valid_until: datetime | None = None


class BatchRequest(BaseModel):
    """Body of POST /api/batch/decision-cards."""
    patients: list[BatchPatientRequest]
    max_concurrency: int = Field(default=8, ge=1, le=64)


class BatchResponse(BaseModel):
    n_requested: int
    n_succeeded: int
    n_failed: int
    n_abstained: int
    started_at: datetime
    completed_at: datetime
    duration_ms: int = Field(..., ge=0)
    results: list[BatchPatientResult]


# ─────────────────────────────────────────────────────────────────────
# 10. healthcare.compute_discharge_counseling -- output: DischargeCounseling
# ─────────────────────────────────────────────────────────────────────
#
# Patient-facing discharge instructions assembled deterministically from the
# DecisionCard's structured outputs (medications, recommendation, risk).
#
# Why not LLM-only?
#   - Hallucinated medications or fabricated dosages are an outright safety
#     hazard at the discharge handoff. The template approach binds every
#     statement to a structured input.
#   - Reproducibility: regression tests can assert that medication class X
#     always renders with the correct red-flag set.
#   - The optional LLM polish stage (TRUSTEDRISK_COUNSELING_LLM_POLISH=1)
#     re-phrases without inventing -- original facts come from the template.

class COINSkillMatch(BaseModel):
    """One skill match candidate from COIN-3 handshake discovery."""
    skill_id: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str


class COINDialogResult(BaseModel):
    """Output of dialog_with_partner (COIN-1)."""
    target_agent_id: str
    source_prompt: str
    candidate_skills: list[COINSkillMatch]
    matched_skill_id: str | None = None
    structured_result: dict[str, Any] | None = None
    nl_response: str
    method: Literal["llm", "deterministic_template"]
    safety_warnings: list[str] = Field(default_factory=list)


class ComorbidityIndices(BaseModel):
    """Output of compute_charlson_elixhauser_index (DATA-4)."""
    icd10_codes_input: list[str]
    icd10_codes_recognized: list[str]
    charlson_conditions_present: list[str]
    charlson_score: int = Field(ge=0, le=40)
    elixhauser_conditions_present: list[str]
    elixhauser_score: int = Field(ge=0, le=40)
    method: Literal["quan_2005_charlson_aahrq_elixhauser"] = \
        "quan_2005_charlson_aahrq_elixhauser"
    references: list[str] = Field(
        default_factory=lambda: [
            "Quan H, et al. Coding algorithms for defining comorbidities "
            "in ICD-9-CM and ICD-10 administrative data. Med Care. "
            "2005;43:1130.",
            "AHRQ HCUP. Elixhauser Comorbidity Software Refined for ICD-10-CM. "
            "https://www.hcup-us.ahrq.gov/toolssoftware/comorbidityicd10/comorbidity_icd10.jsp",
            "Charlson ME, et al. A new method of classifying prognostic "
            "comorbidity in longitudinal studies. J Chronic Dis. 1987;40:373.",
            "van Walraven C, et al. A modification of the Elixhauser "
            "comorbidity measures into a point system for hospital death "
            "using administrative data. Med Care 2009;47:626.",
        ],
    )


class RxNormDDIInteraction(BaseModel):
    """One drug-drug interaction returned by RxNav."""
    drug_a: str
    drug_b: str
    severity: Literal["high", "moderate", "low", "unknown"] = "unknown"
    description: str
    source: str = "RxNav-NLM"


class RxNormDDIReport(BaseModel):
    """Output of compute_rxnorm_ddi_lookup (DATA-2)."""
    drugs_requested: list[str]
    rxnorm_ids_resolved: dict[str, str | None]
    interactions: list[RxNormDDIInteraction]
    n_interactions: int = Field(ge=0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    method: Literal["rxnav_live", "offline_cache", "deterministic_fallback"]
    references: list[str] = Field(
        default_factory=lambda: [
            "U.S. National Library of Medicine. RxNav and RxNorm. "
            "https://lhncbc.nlm.nih.gov/RxNav/",
        ],
    )


class LOINCNormalizedObservation(BaseModel):
    """One Observation after LOINC normalization."""
    raw_code: str
    raw_system: str
    normalized_loinc_code: str | None = None
    canonical_name: str | None = None
    short_name: str | None = None
    unit_canonical: str | None = None
    component_class: str | None = None


class LOINCNormalizationReport(BaseModel):
    """Output of compute_normalize_observations (DATA-3)."""
    n_input: int = Field(ge=0)
    n_normalized: int = Field(ge=0)
    n_unmapped: int = Field(ge=0)
    observations: list[LOINCNormalizedObservation]
    method: Literal["loinc_table", "deterministic_fallback"]
    references: list[str] = Field(
        default_factory=lambda: [
            "Logical Observation Identifiers Names and Codes (LOINC). "
            "https://loinc.org",
        ],
    )


class UMLSConceptMapping(BaseModel):
    """One concept mapping result."""
    source_code: str
    source_vocabulary: Literal["ICD10", "ICD10CM", "SNOMEDCT_US", "RXNORM"]
    cui: str | None = None
    preferred_name: str | None = None
    target_codes: dict[str, list[str]] = Field(default_factory=dict)


class UMLSMappingReport(BaseModel):
    """Output of compute_umls_concept_map (DATA-1)."""
    n_input: int = Field(ge=0)
    n_mapped: int = Field(ge=0)
    mappings: list[UMLSConceptMapping]
    method: Literal["umls_rrf_local", "deterministic_fallback"]
    references: list[str] = Field(
        default_factory=lambda: [
            "U.S. National Library of Medicine. Unified Medical Language "
            "System (UMLS). https://www.nlm.nih.gov/research/umls/",
        ],
    )


class ClinicalEntity(BaseModel):
    """One entity extracted from clinical free text."""
    entity_type: Literal[
        "problem", "medication", "allergy", "family_history",
        "procedure", "lab_result", "vital_sign",
    ]
    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    normalized_concept_id: str | None = None
    normalized_vocabulary: Literal[
        "SNOMEDCT_US", "ICD10CM", "RXNORM", "LOINC", "none",
    ] = "none"
    is_negated: bool = False
    temporal_context: Literal[
        "current", "historical", "family_history", "future_planned",
        "rule_out", "unspecified",
    ] = "unspecified"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ClinicalNERReport(BaseModel):
    """Output of compute_clinical_ner (CHART-1)."""
    source_text_length: int = Field(ge=0)
    n_entities: int = Field(ge=0)
    entities_by_type: dict[str, int] = Field(default_factory=dict)
    entities: list[ClinicalEntity]
    method: Literal["llm", "rule_based", "hybrid"]
    safety_warnings: list[str] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=lambda: [
            "Savova GK, et al. Mayo Clinical Text Analysis and Knowledge "
            "Extraction System (cTAKES). JAMIA 2010;17:507.",
        ],
    )


class StructuredDischargeSummary(BaseModel):
    """Output of compute_structure_discharge_summary (CHART-3)."""
    raw_text_length: int = Field(ge=0)
    extracted_recommendation: str | None = None
    extracted_confidence: str | None = None
    extracted_medications: list[dict[str, str]] = Field(default_factory=list)
    extracted_problems: list[str] = Field(default_factory=list)
    extracted_allergies: list[str] = Field(default_factory=list)
    extracted_followup_window_days: tuple[int, int] | None = None
    extracted_abstain_triggers: list[str] = Field(default_factory=list)
    extraction_confidence: Literal["high", "medium", "low"] = "low"
    method: Literal["llm", "rule_based", "hybrid"]
    safety_warnings: list[str] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=lambda: [
            "ONC USCDI v4 -- Care Plan & Discharge Summary content.",
        ],
    )


# ─────────────────────── EVAL block ───────────────────────

class HEDISMeasure(BaseModel):
    """One NCQA HEDIS measure result (EVAL-1)."""
    measure_id: Literal["CDC-HM2", "CBP", "COL", "CCS", "BCS", "AAB"]
    measure_name: str
    eligible: bool
    in_compliance: bool
    rationale: str
    last_seen_iso: str | None = None
    references: list[str] = Field(default_factory=list)


class HEDISReport(BaseModel):
    """Output of compute_hedis_score (EVAL-1)."""
    patient_age: int | None = None
    patient_sex: str | None = None
    n_measures_evaluated: int = Field(ge=0)
    n_compliant: int = Field(ge=0)
    n_non_compliant: int = Field(ge=0)
    n_not_eligible: int = Field(ge=0)
    measures: list[HEDISMeasure]
    composite_score_pct: float = Field(ge=0.0, le=100.0)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "NCQA HEDIS 2024 Technical Specifications.",
        ],
    )


class HospitalCompareBenchmark(BaseModel):
    """Output of compute_hospital_compare_benchmark (EVAL-2)."""
    institution_label: str
    institution_kpis: dict[str, float]
    peer_tier: Literal[
        "academic_major", "community_large", "community_small",
        "rural", "specialty",
    ]
    peer_medians: dict[str, float]
    deltas_vs_median: dict[str, float]
    percentile_rank: dict[str, int]
    composite_quality_score: float = Field(ge=0.0, le=100.0)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Centers for Medicare & Medicaid Services. Hospital Compare. "
            "https://data.cms.gov/provider-data/topics/hospitals",
        ],
    )


class SchwartzQualityScore(BaseModel):
    """Output of compute_schwartz_quality_score (EVAL-3)."""
    request_id: str | None = None
    trustworthy_score: float = Field(ge=0.0, le=1.0)
    relevant_score: float = Field(ge=0.0, le=1.0)
    actionable_score: float = Field(ge=0.0, le=1.0)
    usable_score: float = Field(ge=0.0, le=1.0)
    composite_score: float = Field(ge=0.0, le=1.0)
    grade: Literal["A", "B", "C", "D", "F"]
    rationale: str
    weak_dimensions: list[str] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=lambda: [
            "Schwartz JM, et al. Best Practices for Designing High-Quality "
            "Clinical Decision Support. JAMIA Open 2017;1:235.",
        ],
    )


class CohortComparativeReport(BaseModel):
    """Output of compute_comparative_effectiveness (EVAL-4)."""
    n_patients: int = Field(ge=1)
    arr_30d: float
    rr_30d: float
    nnt: float | None = None
    intervention_arm_event_rate: float = Field(ge=0.0, le=1.0)
    control_arm_event_rate: float = Field(ge=0.0, le=1.0)
    p_value: float | None = None
    ci95_arr_low: float | None = None
    ci95_arr_high: float | None = None
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Pocock SJ, McMurray JJV, Collier TJ. Statistical Controversies "
            "in Reporting of Clinical Trials. JACC 2015;66:2536.",
        ],
    )


class AdversarialPerturbation(BaseModel):
    """One perturbation found by the adversarial generator (SAFE-1)."""
    delta_l: int = Field(default=0)
    delta_a: int = Field(default=0)
    delta_c: int = Field(default=0)
    delta_e: int = Field(default=0)
    perturbed_lace_total: int = Field(ge=0, le=19)
    new_probability: float = Field(ge=0.0, le=1.0)
    new_action_prediction: str
    l1_distance: int = Field(ge=0)


class FragilityReport(BaseModel):
    """Output of compute_adversarial_fragility (SAFE-1)."""
    original_lace_total: int = Field(ge=0, le=19)
    original_probability: float = Field(ge=0.0, le=1.0)
    original_action: str
    n_perturbations_evaluated: int = Field(ge=0)
    boundary_crossing_perturbations: list[AdversarialPerturbation]
    minimum_l1_to_flip: int | None = Field(default=None, ge=1)
    fragility_score: float = Field(ge=0.0, le=1.0)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Goodfellow IJ, et al. Explaining and Harnessing Adversarial "
            "Examples. ICLR 2015. arXiv:1412.6572.",
        ],
    )


class OODDetectionReport(BaseModel):
    """Output of compute_ood_detector (SAFE-2)."""
    lace_components: dict[str, float]
    mahalanobis_distance: float = Field(ge=0.0)
    chi2_p_value: float = Field(ge=0.0, le=1.0)
    is_out_of_distribution: bool
    confidence_recommendation: Literal["preferred", "degraded",
                                              "abstain_recommended"]
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Mahalanobis PC. On the Generalised Distance in Statistics. "
            "Proc Natl Inst Sci India 1936;2:49.",
            "Lee K, et al. A Simple Unified Framework for Detecting "
            "Out-of-Distribution Samples. NeurIPS 2018.",
        ],
    )


class PenTestFinding(BaseModel):
    """One penetration-test attempt + result."""
    attack_id: str
    attack_category: Literal[
        "auth_bypass", "fhir_injection", "oauth_replay",
        "smart_launch_tampering", "ssrf", "xss",
    ]
    description: str
    target_endpoint: str
    expected_status: int
    observed_status: int
    blocked: bool
    severity: Literal["critical", "high", "moderate", "low", "info"]


class PenTestReport(BaseModel):
    """Output of compute_pentest_suite (SAFE-3)."""
    n_findings: int = Field(ge=0)
    n_blocked: int = Field(ge=0)
    n_unblocked: int = Field(ge=0)
    findings: list[PenTestFinding]
    overall_posture: Literal["pass", "warn", "fail"]
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "OWASP Top 10 (2021).",
            "FDA Cybersecurity in Medical Devices: Quality System "
            "Considerations (2023).",
        ],
    )


class ChaosScenarioResult(BaseModel):
    """One chaos-engineering scenario outcome."""
    scenario_id: str
    fault_type: Literal[
        "tool_timeout", "tool_500", "tool_malformed_json",
        "network_partition", "auth_revoke_mid_call",
    ]
    iterations: int = Field(ge=1)
    success_rate: float = Field(ge=0.0, le=1.0)
    error_rate: float = Field(ge=0.0, le=1.0)
    abstain_rate: float = Field(ge=0.0, le=1.0)
    rationale: str


class ChaosEngineeringReport(BaseModel):
    """Output of compute_chaos_run (SAFE-4)."""
    scenarios: list[ChaosScenarioResult]
    overall_resilience_score: float = Field(ge=0.0, le=1.0)
    weakest_scenario: str | None = None
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Basiri A, et al. Chaos Engineering. IEEE Software 2016;33:35.",
        ],
    )


class HL7v2ParsedMessage(BaseModel):
    """Output of parse_hl7v2_adt (EHR-1)."""
    message_type: Literal["A01", "A02", "A03", "A04", "A08", "unknown"]
    sending_application: str | None = None
    sending_facility: str | None = None
    message_control_id: str | None = None
    message_datetime_iso: str | None = None
    fhir_bundle: dict[str, Any]
    n_resources: int = Field(ge=0)
    parse_warnings: list[str] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=lambda: [
            "HL7 Version 2.x Standard. https://www.hl7.org/implement/"
            "standards/product_section.cfm?section=13",
        ],
    )


class CCDAParsedDocument(BaseModel):
    """Output of parse_ccda_document (EHR-2)."""
    document_type: str | None = None
    patient_id: str | None = None
    fhir_bundle: dict[str, Any]
    n_resources: int = Field(ge=0)
    sections_extracted: list[str] = Field(default_factory=list)
    parse_warnings: list[str] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=lambda: [
            "ONC Consolidated CDA (C-CDA) Implementation Guide R2.1.",
        ],
    )


class SDOHReport(BaseModel):
    """Output of compute_sdoh_score (EHR-3)."""
    patient_zip: str | None = None
    z_codes_present: list[str] = Field(default_factory=list)
    z_code_categories: list[str] = Field(default_factory=list)
    svi_overall_percentile: float | None = Field(default=None, ge=0.0,
                                                          le=1.0)
    svi_themes: dict[str, float] = Field(default_factory=dict)
    composite_sdoh_burden_score: float = Field(ge=0.0, le=1.0)
    fairness_audit_modifier: float = Field(ge=0.0, le=1.0)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "ICD-10-CM Z55-Z65 -- Persons with potential health hazards "
            "related to socioeconomic and psychosocial circumstances.",
            "AHRQ / CDC Social Vulnerability Index. "
            "https://www.atsdr.cdc.gov/placeandhealth/svi/",
        ],
    )


class MerkleProofStep(BaseModel):
    """One sibling-hash step in a Merkle inclusion proof."""
    sibling_hash: str
    is_left: bool


class MerkleAuditChainResult(BaseModel):
    """Output of compute_merkle_audit_root (AUDIT-1)."""
    n_events: int = Field(ge=0)
    leaf_hashes: list[str]
    merkle_root: str
    height: int = Field(ge=0)
    method: Literal["sha256_merkle"] = "sha256_merkle"


class MerkleInclusionProof(BaseModel):
    """Proof that an event_id is included in a published Merkle root."""
    event_id: str
    leaf_hash: str
    merkle_root: str
    proof_steps: list[MerkleProofStep]
    verified: bool


class ScenarioCounterfactualFlip(BaseModel):
    """One minimum-modification perturbation that flips a clinical
    decision in a scenario."""
    factor_name: str
    factor_description: str
    original_value: Any
    modified_value: Any
    original_outcome: str
    modified_outcome: str
    flip_distance: float = Field(
        ge=0.0,
        description=(
            "Magnitude of the perturbation in factor-natural units "
            "(beats per minute, mg/dL, dichotomous flip = 1.0)."
        ),
    )


class FHIRWriteBackReport(BaseModel):
    """Output of compute_write_decision_to_fhir (Phase 13.10 F1)."""
    fhir_server_url: str
    patient_id: str
    composition_id: str
    composition_identifier: str
    provenance_id: str
    composition_status_code: int = Field(ge=100, le=599)
    provenance_status_code: int = Field(ge=100, le=599)
    write_succeeded: bool
    written_at_iso: str
    rationale: str
    references: list[str] = Field(default_factory=list)


class APACHEIIReport(BaseModel):
    """Output of compute_apache_ii_score (Knaus 1985)."""
    aps_score: int = Field(ge=0, le=60)
    age_points: int = Field(ge=0, le=6)
    chronic_health_points: int = Field(ge=0, le=5)
    apache_ii_total: int = Field(ge=0, le=71)
    predicted_mortality_pct: float = Field(ge=0.0, le=100.0)
    severity_tier: Literal["mild", "moderate", "severe", "critical"]
    icu_admission_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class SOFAReport(BaseModel):
    """Output of compute_sofa_score (Vincent 1996; Sepsis-3 2016)."""
    respiratory_points: int = Field(ge=0, le=4)
    coagulation_points: int = Field(ge=0, le=4)
    liver_points: int = Field(ge=0, le=4)
    cardiovascular_points: int = Field(ge=0, le=4)
    cns_points: int = Field(ge=0, le=4)
    renal_points: int = Field(ge=0, le=4)
    sofa_total: int = Field(ge=0, le=24)
    sepsis_3_dysfunction: bool = Field(
        description=(
            "Sepsis-3 (2016): SOFA increase ≥ 2 points from baseline "
            "in the setting of suspected infection."
        ),
    )
    estimated_mortality_pct: float = Field(ge=0.0, le=100.0)
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class MELDReport(BaseModel):
    """Output of compute_meld_score (Kamath 2001 / MELD-Na 2008)."""
    bilirubin_mg_dl: float
    creatinine_mg_dl: float
    inr: float
    sodium_mmol_l: float | None = None
    on_dialysis: bool
    meld_classic: int = Field(ge=6, le=40)
    meld_na: int | None = Field(default=None, ge=6, le=40)
    estimated_3mo_mortality_pct: float = Field(ge=0.0, le=100.0)
    severity_tier: Literal["low", "moderate", "high", "critical"]
    transplant_eligibility_threshold_met: bool = Field(
        description="MELD ≥ 15 -- UNOS minimum listing threshold.",
    )
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class RIFLEReport(BaseModel):
    """Output of compute_rifle_aki_classification (Bellomo 2004)."""
    serum_creatinine_baseline_mg_dl: float
    serum_creatinine_current_mg_dl: float
    creatinine_ratio: float
    urine_output_ml_per_kg_per_hour: float | None
    rifle_class: Literal[
        "no_aki", "risk", "injury", "failure", "loss", "esrd",
    ]
    aki_etiology_clue: str
    nephrology_consult_recommended: bool
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class SubgroupFairnessMetric(BaseModel):
    """Per-subgroup fairness metric snapshot for one outcome."""
    subgroup: str
    n: int = Field(ge=0)
    n_positive: int = Field(ge=0)
    n_predicted_positive: int = Field(ge=0)
    n_true_positive: int = Field(ge=0)
    n_false_positive: int = Field(ge=0)
    selection_rate: float = Field(ge=0.0, le=1.0)
    true_positive_rate: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0)
    selection_rate_ci95_lower: float = Field(ge=0.0, le=1.0)
    selection_rate_ci95_upper: float = Field(ge=0.0, le=1.0)


class FairnessAdvancedReport(BaseModel):
    """Output of compute_fairness_advanced (Phase 13.2 I1).

    Implements **Demographic Parity** (selection-rate parity across
    subgroups) and **Equalized Odds** (TPR + FPR parity), with chi-
    squared independence tests + bootstrap 95% CI on each rate.
    """
    sensitive_attribute: str
    n_subgroups: int = Field(ge=2)
    n_total: int = Field(ge=0)
    metrics: list[SubgroupFairnessMetric]
    demographic_parity_max_gap: float = Field(
        ge=0.0,
        description=(
            "Maximum |selection_rate(g_a) - selection_rate(g_b)| across "
            "all subgroup pairs. Values above 0.10 are typically flagged "
            "as material disparate-impact under EEOC's 4/5ths rule."
        ),
    )
    equalized_odds_tpr_max_gap: float = Field(ge=0.0)
    equalized_odds_fpr_max_gap: float = Field(ge=0.0)
    chi2_independence_p_value: float | None = Field(
        default=None,
        description=(
            "Chi-squared p-value for the null `outcome ⊥ subgroup`. "
            "p < 0.05 indicates the predicted positive count and the "
            "subgroup label are not independent."
        ),
    )
    posture: Literal[
        "fair", "monitor", "investigate", "violation",
    ] = Field(
        description=(
            "fair: max gap < 0.05; monitor: 0.05-0.10; "
            "investigate: 0.10-0.20; violation: > 0.20."
        ),
    )
    rationale: str
    references: list[str] = Field(default_factory=list)


class ScenarioCounterfactualReport(BaseModel):
    """Per-clinical-scenario counterfactual analysis.

    Lists the perturbations evaluated and the minimum-modification
    flips that change the recommended disposition. Used as the
    Phase 12.5 'right-to-explanation' surface for the 5 end-to-end
    clinical scenarios."""
    scenario_id: str
    title: str
    baseline_outcome: str
    perturbations_evaluated: int = Field(ge=0)
    flips_found: list[ScenarioCounterfactualFlip]
    n_flips: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(default_factory=list)


class MerkleAuditVerificationReport(BaseModel):
    """Output of compute_verify_audit_chain (Phase 12.2 -- AUDIT-3).

    Posture is one of:
      `verified`       -- the recomputed leaf hash matches AND the proof
                         reconstructs to the expected root.
      `tamper_event`   -- the recomputed leaf hash diverges from the
                         claimed leaf -- a byte of the audited event was
                         changed.
      `tamper_chain`   -- the leaf hash is intact but the proof does NOT
                         reconstruct to the expected root -- a sibling
                         along the chain was tampered.
      `unknown_event`  -- the target event_id is not present in the
                         supplied ledger.
    """
    event_id: str
    expected_merkle_root: str
    recomputed_leaf_hash: str
    claimed_leaf_hash: str
    recomputed_root: str
    posture: Literal[
        "verified", "tamper_event", "tamper_chain", "unknown_event",
    ]
    rationale: str
    references: list[str] = Field(default_factory=list)


class DPNoisedEquityDashboard(BaseModel):
    """Output of compute_dp_equity_dashboard (AUDIT-2)."""
    epsilon: float = Field(gt=0.0)
    sensitivity: float = Field(gt=0.0)
    n_total_decisions_noised: int
    segments: list[dict[str, Any]]
    suppressed_segments: list[str]
    suppression_threshold: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Dwork C. Differential Privacy. ICALP 2006.",
            "Dwork C, Roth A. The Algorithmic Foundations of Differential "
            "Privacy. Foundations and Trends in TCS 2014;9:211.",
        ],
    )


class RightToExplanationReport(BaseModel):
    """Output of compute_right_to_explanation (AUDIT-3)."""
    request_id: str | None = None
    decision_action: str | None = None
    decision_confidence: str | None = None
    plain_language_rationale: str
    factors_considered: list[str]
    factors_weight: dict[str, float]
    data_sources: list[str]
    method: Literal["llm", "deterministic_template"]
    safety_warnings: list[str] = Field(default_factory=list)
    legal_basis: str = (
        "GDPR Article 22 (right to explanation of automated decisions) + "
        "EU AI Act Article 13 (transparency requirements for high-risk "
        "AI systems)."
    )
    references: list[str] = Field(
        default_factory=lambda: [
            "Regulation (EU) 2016/679 (GDPR), Art. 22.",
            "Regulation (EU) 2024/1689 (EU AI Act), Art. 13 + Annex IV.",
        ],
    )


class PatientAuditSummary(BaseModel):
    """Output of compute_patient_audit_summary (AUDIT-4)."""
    request_id: str | None = None
    headline_question: str
    plain_language_answer: str
    features_considered: list[dict[str, Any]]
    citations_used: list[dict[str, str]]
    abstain_triggers_present: list[str]
    confidence: str | None = None
    further_info_contact: str = (
        "If you have questions about this summary, please contact your "
        "care team or the patient-relations office at your hospital."
    )
    references: list[str] = Field(
        default_factory=lambda: [
            "AHRQ -- Patient Engagement and Health Literacy.",
        ],
    )


class EventLogEntry(BaseModel):
    """One append-only event record (TIME-1)."""
    event_id: str
    patient_id: str
    fhir_resource_type: str
    fhir_resource_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str = "ehr"
    created_at_iso: str
    sequence_number: int = Field(ge=0)


class PatientStateSnapshot(BaseModel):
    """Output of replay_to_state (TIME-1)."""
    patient_id: str
    n_events_replayed: int = Field(ge=0)
    as_of_iso: str | None = None
    resources_by_type: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict)
    state_hash: str


class IncrementalRiskUpdate(BaseModel):
    """Output of recompute_with_observation (TIME-2)."""
    prior_probability: float = Field(ge=0.0, le=1.0)
    posterior_probability: float = Field(ge=0.0, le=1.0)
    delta: float
    affected_lace_components: list[str]
    trigger_observation_type: str
    rationale: str
    full_recompute_recommended: bool = False


class DeteriorationNowcast(BaseModel):
    """Output of nowcast_deterioration (TIME-3)."""
    window_start_iso: str
    window_end_iso: str
    n_observations: int = Field(ge=0)
    trends: dict[str, float]
    alert_tier: Literal["ok", "watch", "alert"]
    triggered_signals: list[str]
    rationale: str


class TrajectoryPoint(BaseModel):
    """One point in a predicted risk trajectory."""
    hour_offset: int = Field(ge=0, le=720)
    probability_mean: float = Field(ge=0.0, le=1.0)
    probability_ci95_low: float = Field(ge=0.0, le=1.0)
    probability_ci95_high: float = Field(ge=0.0, le=1.0)


class TrajectoryPrediction(BaseModel):
    """Output of predict_trajectory (TIME-4)."""
    horizon_hours: int = Field(ge=1, le=720)
    points: list[TrajectoryPoint]
    expected_peak_probability: float = Field(ge=0.0, le=1.0)
    expected_peak_hour: int = Field(ge=0, le=720)
    method: Literal["bayesian_state_space", "deterministic_baseline"]
    rationale: str


class PubMedAbstract(BaseModel):
    """One PubMed citation returned by Entrez."""
    pmid: str
    title: str
    journal: str | None = None
    pub_year: int | None = Field(default=None, ge=1900, le=2100)
    authors: list[str] = Field(default_factory=list)
    abstract: str
    doi: str | None = None


class PubMedSearchReport(BaseModel):
    """Output of compute_pubmed_search (KNOWLEDGE-1)."""
    query: str
    n_results: int = Field(ge=0)
    results: list[PubMedAbstract]
    method: Literal["entrez_live", "offline_cache",
                       "deterministic_fallback", "abstain"]
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "U.S. National Library of Medicine. NCBI E-utilities Help. "
            "https://www.ncbi.nlm.nih.gov/books/NBK25501/",
        ],
    )


class ClinicalTrial(BaseModel):
    """One ClinicalTrials.gov match."""
    nct_id: str
    title: str
    status: str
    phase: str | None = None
    conditions: list[str] = Field(default_factory=list)
    eligibility_min_age: str | None = None
    eligibility_max_age: str | None = None
    eligibility_sex: str | None = None
    locations: list[str] = Field(default_factory=list)
    sponsor: str | None = None
    summary: str = ""


class ClinicalTrialsMatchReport(BaseModel):
    """Output of compute_clinical_trials_matcher (KNOWLEDGE-2)."""
    conditions_searched: list[str]
    n_matches: int = Field(ge=0)
    matches: list[ClinicalTrial]
    method: Literal["clinicaltrials_live", "offline_cache",
                       "deterministic_fallback", "abstain"]
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "ClinicalTrials.gov API v2. https://clinicaltrials.gov/api/",
        ],
    )


class DrugPriceTier(BaseModel):
    """Pricing-tier estimate for a single drug."""
    drug_name: str
    rxcui: str | None = None
    therapeutic_class: str
    is_generic_available: bool
    estimated_monthly_cost_usd: float = Field(ge=0.0)
    cost_tier: Literal["very_low", "low", "moderate", "high", "specialty"]
    has_patient_assistance: bool = False
    notes: str = ""


class DrugPricingReport(BaseModel):
    """Output of compute_drug_pricing (KNOWLEDGE-3)."""
    n_drugs: int = Field(ge=0)
    drugs: list[DrugPriceTier]
    total_monthly_cost_usd: float = Field(ge=0.0)
    high_cost_drugs: list[str] = Field(default_factory=list)
    method: Literal["embedded_table", "deterministic_fallback"]
    disclaimer: str = (
        "Prices are USD reference estimates from public sources and DO "
        "NOT reflect the patient's actual cost (which depends on their "
        "specific insurance, copay tier, deductible, and pharmacy). "
        "Confirm with the dispensing pharmacy."
    )
    references: list[str] = Field(
        default_factory=lambda: [
            "Centers for Medicare & Medicaid Services (CMS) NADAC pricing.",
            "U.S. Food and Drug Administration Orange Book.",
        ],
    )


class NIHGrant(BaseModel):
    """One NIH-funded research project."""
    project_number: str
    project_title: str
    abstract_text: str
    organization_name: str
    fiscal_year: int | None = None
    award_amount_usd: float | None = None
    pi_names: list[str] = Field(default_factory=list)


class NIHReporterReport(BaseModel):
    """Output of compute_nih_reporter_search (KNOWLEDGE-4)."""
    query_terms: list[str]
    n_grants: int = Field(ge=0)
    grants: list[NIHGrant]
    method: Literal["reporter_live", "offline_cache",
                       "deterministic_fallback", "abstain"]
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "NIH RePORTER API v2. https://api.reporter.nih.gov/",
        ],
    )


class OrderSetEntry(BaseModel):
    """One discrete order in a discharge order set."""
    category: Literal[
        "medication", "lab_monitoring", "vital_monitoring",
        "followup_appointment", "dietary", "activity",
        "patient_education", "imaging", "consult",
    ]
    order_text: str
    rationale: str = ""
    timing: str = ""
    priority: Literal["routine", "urgent", "stat"] = "routine"
    references: list[str] = Field(default_factory=list)


class OrderSet(BaseModel):
    """Output of compute_order_set (LIB-1)."""
    discharge_action: str
    risk_tier: Literal["low", "moderate", "high"]
    next_visit_window_days: tuple[int, int]
    orders: list[OrderSetEntry]
    n_orders: int = Field(ge=0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Forster AJ, et al. The incidence and severity of adverse "
            "events affecting patients after discharge from the hospital. "
            "Ann Intern Med. 2003;138:161.",
            "Jack BW, et al. A reengineered hospital discharge program. "
            "Ann Intern Med. 2009;150:178.",
        ],
    )


class AdherenceFactor(BaseModel):
    """One factor contributing to the adherence prediction."""
    name: str
    direction: Literal["positive", "negative", "neutral"]
    weight: float
    detail: str


class AdherencePredictionReport(BaseModel):
    """Output of compute_medication_adherence_predictor (LIB-2)."""
    n_medications: int = Field(ge=0)
    pills_per_day: int = Field(ge=0)
    distinct_dose_times: int = Field(ge=0)
    regimen_complexity_index: float = Field(ge=0.0)
    adherence_30d_probability: float = Field(ge=0.0, le=1.0)
    risk_tier: Literal["low", "moderate", "high"]
    contributing_factors: list[AdherenceFactor]
    interventions_recommended: list[str]
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Osterberg L, Blaschke T. Adherence to medication. "
            "NEJM 2005;353:487.",
            "George J, et al. Development and validation of the medication "
            "regimen complexity index. Ann Pharmacother 2004;38:1369.",
        ],
    )


class CareGap(BaseModel):
    """One identified care gap in a patient FHIR Bundle."""
    gap_id: str
    title: str
    domain: Literal[
        "cancer_screening", "vaccination", "metabolic_monitoring",
        "cardiovascular_screening", "preventive_screening",
        "chronic_disease_followup",
    ]
    age_range: tuple[int, int]
    sex_specific: Literal["any", "female", "male"] = "any"
    last_seen_iso: str | None = None
    overdue_days: int | None = None
    priority: Literal["high", "moderate", "low"]
    recommendation: str
    references: list[str] = Field(default_factory=list)


class CareGapReport(BaseModel):
    """Output of compute_care_gap_detector (LIB-3)."""
    patient_age: int | None = None
    patient_sex: str | None = None
    n_gaps_found: int = Field(ge=0)
    high_priority_gaps: list[CareGap]
    moderate_priority_gaps: list[CareGap]
    low_priority_gaps: list[CareGap]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "U.S. Preventive Services Task Force (USPSTF) recommendations.",
            "Centers for Disease Control and Prevention (CDC) ACIP "
            "vaccination schedules.",
            "American Diabetes Association Standards of Care.",
        ],
    )


class PROMScores(BaseModel):
    """Patient-Reported Outcome Measures input."""
    instrument: Literal["PROMIS-29", "EQ-5D-5L", "PROMIS-10"]
    physical_function: float | None = Field(default=None, ge=0.0, le=100.0)
    pain_intensity: float | None = Field(default=None, ge=0.0, le=10.0)
    fatigue: float | None = Field(default=None, ge=0.0, le=100.0)
    depression: float | None = Field(default=None, ge=0.0, le=100.0)
    anxiety: float | None = Field(default=None, ge=0.0, le=100.0)
    sleep_disturbance: float | None = Field(default=None, ge=0.0, le=100.0)
    social_role_satisfaction: float | None = Field(default=None,
                                                          ge=0.0, le=100.0)
    eq5d_index_score: float | None = Field(default=None, ge=-1.0, le=1.0)


class PROMInfluenceReport(BaseModel):
    """Output of compute_prom_influence (LIB-4)."""
    instrument: str
    qaly_delta_30d: float
    distress_burden_score: float = Field(ge=0.0, le=100.0)
    high_burden_dimensions: list[str]
    action_dominance_shift: dict[str, float]
    recommendation: str
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Cella D, et al. PROMIS: Patient-Reported Outcomes Measurement "
            "Information System. JAMIA 2010;17:213.",
            "Herdman M, et al. EQ-5D-5L crosswalk. Qual Life Res 2011;20:1727.",
        ],
    )


class CounselingSection(BaseModel):
    """One section of the discharge counseling document."""
    section_id: Literal[
        "your_medications",
        "follow_up",
        "warning_signs",
        "activities_self_care",
        "questions_to_ask",
    ]
    title: str
    plain_text: str
    bullets: list[str] = Field(default_factory=list)


class TranslatedDischargeCounseling(BaseModel):
    """Output of compute_translate_discharge_counseling (PATIENT-1)."""
    source_locale: Literal["en"] = "en"
    target_locale: Literal["en", "es", "it", "zh", "ar"]
    translation_method: Literal["deterministic_passthrough", "llm"]
    reading_level_grade: int = Field(default=6, ge=1, le=12)
    sections: list[CounselingSection]
    disclaimer: str
    safety_warnings: list[str] = Field(default_factory=list)
    notes: str = (
        "This is a translation of a discharge summary. Always follow the "
        "official paper instructions; this is supplementary."
    )


class PatientFAQAnswer(BaseModel):
    """Output of compute_patient_faq (PATIENT-2)."""
    question: str
    in_scope: bool
    answer_text: str
    citations: list[EvidenceSource] = Field(default_factory=list)
    refusal_reason: str | None = None
    method: Literal["llm", "deterministic_template"]
    safety_warnings: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class CaregiverInstructionSet(BaseModel):
    """Output of compute_caregiver_summary (PATIENT-3)."""
    target_audience: Literal["caregiver", "guardian", "family_proxy"]
    reading_level_grade: int = Field(default=10, ge=1, le=14)
    summary: str
    red_flag_actions: list[str]
    daily_observation_checklist: list[str]
    when_to_call_pcp: list[str]
    when_to_call_911: list[str]
    method: Literal["llm", "deterministic_template"]


class FormattedNotificationChannel(BaseModel):
    """One output channel rendering of the discharge counseling."""
    channel: Literal["sms", "email_html", "print_markdown"]
    chunks: list[str]
    n_chunks: int
    total_characters: int


class FormattedNotificationBundle(BaseModel):
    """Output of format_discharge_notifications (PATIENT-4)."""
    sections_formatted: int
    channels: list[FormattedNotificationChannel]


class DischargeCounseling(BaseModel):
    """Output of compute_discharge_counseling."""
    patient_id: str | None
    locale: str = "en"
    reading_level_grade: int = Field(default=6, ge=1, le=12)
    sections: list[CounselingSection]
    follow_up_window_days: tuple[int, int]
    n_medications_explained: int
    n_red_flags: int
    disclaimer: str = (
        "This is a summary written from your discharge plan. It does NOT replace "
        "your discharge paperwork or your conversation with your nurse and doctor. "
        "Always follow the official discharge instructions you were given on paper, "
        "and call your doctor if anything is unclear."
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 11. healthcare.compute_clinical_deterioration_score -- output: DeteriorationReport
# ─────────────────────────────────────────────────────────────────────
#
# NEWS2 (Royal College of Physicians, 2017) -- validated early-warning score
# for inpatient deterioration. Each parameter contributes 0-3 points; the
# total triggers an escalation tier. We add a "trend" sub-score that captures
# rate of change over the last 4-6h to surface deteriorations earlier than
# point-in-time NEWS2 alone.

class VitalSign(BaseModel):
    """Single vital-sign observation."""
    type: Literal[
        "respiratory_rate",
        "spo2",
        "supplemental_oxygen",  # boolean: any O2 supplementation = +2
        "systolic_bp",
        "heart_rate",
        "consciousness",         # AVPU: A=0, V/P/U=3
        "temperature",
    ]
    value: float | str
    unit: str | None = None
    observed_at: datetime


class DeteriorationFactor(BaseModel):
    """One NEWS2 parameter contribution."""
    parameter: str
    value: float | str
    points: int = Field(..., ge=0, le=3)
    rationale: str


class DeteriorationReport(BaseModel):
    """Output of compute_clinical_deterioration_score.

    NEWS2 thresholds (RCP 2017):
      total 0-4   -> routine 12-h observation
      total 5-6   -> urgent clinical review within 1h, increase obs to 1-h
      total ≥7    -> emergency response (rapid response / ICU consult), continuous obs
      any single 3 -> urgent ward-based response within 1h
    """
    patient_id: str | None
    score_total: int = Field(..., ge=0, le=20)
    parameter_contributions: list[DeteriorationFactor]
    severity_tier: Literal["low", "low_medium", "medium", "high"]
    recommended_response: Literal[
        "routine_12h_obs",
        "increase_obs_4_6h",
        "urgent_review_1h",
        "emergency_response",
    ]
    trend_delta: int = Field(...)  # change in score vs prior assessment; must be explicitly computed
    trend_direction: Literal["improving", "stable", "worsening", "unknown"] = "unknown"
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Royal College of Physicians. National Early Warning Score (NEWS) 2. 2017.",
        ],
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 12. healthcare.compute_admission_triage -- output: AdmissionTriage
# ─────────────────────────────────────────────────────────────────────
#
# Hybrid of the Emergency Severity Index (ESI v4, AHRQ 2020) and Manchester
# Triage System for chief-complaint-aware acuity. Maps to a 5-level priority
# (1=immediate, 5=non-urgent) plus a recommended disposition.

class AdmissionTriage(BaseModel):
    """Output of compute_admission_triage."""
    patient_id: str | None
    chief_complaint: str
    esi_level: int = Field(..., ge=1, le=5)
    priority: Literal["immediate", "emergent", "urgent", "less_urgent", "non_urgent"]
    disposition: Literal[
        "resuscitation_room",
        "admit_inpatient",
        "admit_observation",
        "ed_workup_then_dispose",
        "discharge_from_ed",
    ]
    recommended_unit: Literal[
        "icu", "stepdown", "telemetry", "general_ward",
        "observation_unit", "ed_treatment", "outpatient_followup",
    ] | None = None
    red_flag_findings: list[str] = Field(default_factory=list)
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "AHRQ Emergency Severity Index (ESI), Version 4 implementation handbook (2020).",
            "Manchester Triage Group. Emergency Triage. 3rd ed. Wiley-Blackwell, 2014.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 13. healthcare.compute_treatment_selection -- output: TreatmentSelection
# ─────────────────────────────────────────────────────────────────────
#
# Given a clinical condition + structured patient factors, rank candidate
# treatment options applying contraindications and patient preferences,
# with each option grounded against the W3 guideline corpus.

class TreatmentOption(BaseModel):
    """One ranked treatment option."""
    option_id: str           # e.g. "warfarin_continue", "doac_switch", "bridge_lmwh"
    label: str               # patient-language label
    rank: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)  # composite suitability score
    contraindications: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    expected_benefit: str
    expected_harm: str
    grounding_excerpt: str | None = None  # snippet supporting this option
    grounding_source: str | None = None   # citation / source_id


class TreatmentSelection(BaseModel):
    """Output of compute_treatment_selection."""
    patient_id: str | None
    condition: str
    options: list[TreatmentOption]
    top_pick_id: str | None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# 14. healthcare.compute_pediatric_early_warning -- output: PEWSReport
# ─────────────────────────────────────────────────────────────────────
#
# Pediatric Early Warning Score (PEWS) -- Brighton/Monaghan derivative
# adapted for age-specific vital cutoffs (0-11mo / 1-4y / 5-11y / 12-17y).
# Each of 4 components scores 0-3; total 0-12 with escalation thresholds.

class PEWSAgeBand(BaseModel):
    """Reference range that varies by age band -- included for audit transparency."""
    band_label: Literal["0-11mo", "1-4y", "5-11y", "12-17y"]
    hr_normal: tuple[int, int]
    rr_normal: tuple[int, int]
    sbp_normal: tuple[int, int]


class PEWSComponent(BaseModel):
    """One PEWS sub-score."""
    component: Literal["behavior", "cardiovascular", "respiratory", "other_concern"]
    raw_value: str
    points: int = Field(..., ge=0, le=3)
    rationale: str


class PEWSReport(BaseModel):
    """Output of compute_pediatric_early_warning."""
    patient_id: str | None
    age_months: int = Field(..., ge=0, le=216)
    age_band: Literal["0-11mo", "1-4y", "5-11y", "12-17y"]
    score_total: int = Field(..., ge=0, le=12)
    components: list[PEWSComponent]
    severity_tier: Literal["low", "medium", "high"]
    recommended_response: Literal[
        "routine_obs",
        "increase_to_hourly_obs",
        "urgent_pediatric_review",
        "rapid_response_picu_consult",
    ]
    age_band_reference: PEWSAgeBand
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Monaghan A. Detecting and managing deterioration in children. "
            "Paediatric Nursing 2005;17(1):32-35.",
            "Brighton & Sussex University Hospitals NHS PEWS Implementation Guide (2022).",
        ],
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 15. healthcare.compute_weight_based_dosing -- output: PediatricDoseRecommendation
# ─────────────────────────────────────────────────────────────────────

class PediatricDoseRecommendation(BaseModel):
    """Output of compute_weight_based_dosing."""
    patient_id: str | None
    drug: str
    indication: str | None = None
    weight_kg: float = Field(..., gt=0, le=300)
    age_months: int = Field(..., ge=0, le=216)
    dose_mg_per_kg: float = Field(..., ge=0)
    calculated_dose_mg: float = Field(..., ge=0)
    adult_max_dose_mg: float | None = None
    capped_at_adult_max: bool = False
    final_dose_mg: float = Field(..., ge=0)
    route: Literal["PO", "IV", "IM", "PR", "SC", "intranasal"]
    formulation: str  # e.g. "amoxicillin 250 mg/5 mL suspension"
    volume_to_administer_ml: float | None = None  # for liquid formulations
    contraindications: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    rationale: str


# ─────────────────────────────────────────────────────────────────────
# 16. healthcare.compute_suicide_risk_assessment -- output: SuicideRiskAssessment
# ─────────────────────────────────────────────────────────────────────
#
# Columbia-Suicide Severity Rating Scale (C-SSRS) -- Posner et al. 2011.
# 5 ideation severity levels + 5 behavior categories. Output captures
# both the worst-ever and last-30-days assessments.

class CSSRSIdeationLevel(BaseModel):
    """Ideation severity 1-5 (lifetime worst + recent)."""
    level: int = Field(..., ge=0, le=5)
    label: Literal[
        "none",
        "wish_to_be_dead",                   # 1
        "non_specific_active_suicidal",      # 2
        "active_with_method_no_plan",        # 3
        "active_with_plan_no_intent",        # 4
        "active_with_plan_and_intent",       # 5
    ]


class SuicideRiskAssessment(BaseModel):
    """Output of compute_suicide_risk_assessment."""
    patient_id: str | None
    ideation_lifetime: CSSRSIdeationLevel
    ideation_past_30d: CSSRSIdeationLevel
    behavior_lifetime_attempts: int = Field(..., ge=0)
    behavior_past_30d: bool          # any actual / interrupted / aborted attempt
    behavior_self_injury_no_intent: bool   # NSSI vs suicidal
    risk_level: Literal["low", "moderate", "high", "imminent"]
    abstain_recommended: bool = False  # always True for low-confidence settings
    abstain_reason: str | None = None
    rationale: str
    safety_plan_indicated: bool
    references: list[str] = Field(
        default_factory=lambda: [
            "Posner K, et al. The Columbia–Suicide Severity Rating Scale. "
            "Am J Psychiatry 2011;168(12):1266-1277.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 17. healthcare.compute_psychiatric_admission_decision -- output: PsychiatricAdmission
# ─────────────────────────────────────────────────────────────────────

class PsychiatricAdmission(BaseModel):
    """Output of compute_psychiatric_admission_decision."""
    patient_id: str | None
    risk_level: Literal["low", "moderate", "high", "imminent"]
    danger_to_self: bool
    danger_to_others: bool
    grave_disability: bool       # unable to provide for basic personal needs
    voluntary_capable: bool      # patient has decisional capacity for voluntary admission
    disposition: Literal[
        "outpatient_followup",
        "crisis_stabilization_unit",
        "voluntary_inpatient",
        "involuntary_hold_evaluation",  # M1 / 5150 type pathway
        "emergency_court_petition",
    ]
    rationale: str
    legal_basis: str | None = None    # state-specific hold criteria description
    safety_plan_required: bool
    follow_up_within_hours: int = Field(..., ge=0, le=336)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "American Psychiatric Association. APA Practice Guideline for the "
            "Assessment and Treatment of Patients with Suicidal Behaviors (2022).",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 18. healthcare.compute_empiric_antibiotic_selection -- output: AntibioticSelection
# ─────────────────────────────────────────────────────────────────────

class AntibioticOption(BaseModel):
    """One ranked antibiotic regimen."""
    regimen_id: str         # e.g. "ceftriaxone_iv"
    label: str              # patient-language drug + route
    rank: int = Field(..., ge=1)
    score: float = Field(..., ge=0.0, le=1.0)
    spectrum_coverage: list[str] = Field(default_factory=list)  # e.g. ["GNB","ESBL"]
    contraindications: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    expected_duration_days: tuple[int, int]
    dose_per_administration: str
    route: Literal["IV", "PO", "IM"]
    cost_tier: Literal["low", "medium", "high"]
    grounding_excerpt: str | None = None
    grounding_source: str | None = None


class AntibioticSelection(BaseModel):
    """Output of compute_empiric_antibiotic_selection."""
    patient_id: str | None
    infection_source: Literal[
        "urinary", "pneumonia", "skin_soft_tissue", "intra_abdominal",
        "cns", "bloodstream", "unknown",
    ]
    severity: Literal["uncomplicated", "complicated", "sepsis", "septic_shock"]
    options: list[AntibioticOption]
    top_pick_id: str | None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    rationale: str
    local_antibiogram_used: bool = False
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# 19. healthcare.compute_antibiotic_de_escalation -- output: DeEscalationPlan
# ─────────────────────────────────────────────────────────────────────

class DeEscalationPlan(BaseModel):
    """Output of compute_antibiotic_de_escalation."""
    patient_id: str | None
    current_regimen: str
    pathogen_identified: str | None    # e.g. "E. coli, ESBL-positive"
    susceptibility_pattern: dict[str, Literal["S", "I", "R"]] = Field(default_factory=dict)
    de_escalation_recommended: bool
    target_regimen: str | None        # narrower antibiotic recommended
    target_regimen_route: Literal["IV", "PO"] | None = None
    iv_to_po_switch_eligible: bool = False
    iv_to_po_criteria_met: list[str] = Field(default_factory=list)
    iv_to_po_criteria_unmet: list[str] = Field(default_factory=list)
    duration_total_days: int = Field(..., ge=0, le=42)
    duration_remaining_days: int = Field(..., ge=0, le=42)
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "IDSA / SHEA Antimicrobial Stewardship Implementation Guidelines (2016).",
            "Cunha BA. Antibiotic stewardship: switching from intravenous to "
            "oral therapy. Infect Dis Clin North Am 2017;31(3):485-503.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 20. healthcare.compute_chemo_dose_adjustment -- output: ChemoDoseAdjustment
# ─────────────────────────────────────────────────────────────────────

class ChemoDoseAdjustment(BaseModel):
    """Output of compute_chemo_dose_adjustment."""
    patient_id: str | None
    regimen: str                      # e.g. "carboplatin_pemetrexed"
    cycle_number: int = Field(..., ge=1, le=24)
    egfr_ml_min: float | None = None
    bilirubin_mg_dl: float | None = None
    ast_ul: float | None = None
    alt_ul: float | None = None
    anc_per_ul: float | None = None        # absolute neutrophil count
    platelets_per_ul: float | None = None
    ecog_performance_status: int | None = Field(default=None, ge=0, le=4)
    decision: Literal[
        "proceed_full_dose",
        "proceed_reduced_dose",
        "delay_one_week",
        "hold_until_recovery",
        "discontinue_consider_alternative",
    ]
    dose_reduction_pct: float = Field(..., ge=0.0, le=100.0)
    delay_reason: str | None = None
    growth_factor_indicated: bool = False  # e.g. pegfilgrastim
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "NCCN Clinical Practice Guidelines (Treatment of Cancer by Site, current edition).",
            "ASCO Clinical Practice Guideline: Recommendations for Use of "
            "WBC Growth Factors (Smith et al., JCO 2015).",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 21. healthcare.compute_oncology_treatment_response -- output: TumorResponse
# ─────────────────────────────────────────────────────────────────────

class TargetLesion(BaseModel):
    """One target tumor lesion (RECIST 1.1 categorical)."""
    lesion_id: str
    location: str
    baseline_longest_diameter_mm: float = Field(..., ge=0)
    current_longest_diameter_mm: float = Field(..., ge=0)


# ─────────────────────────────────────────────────────────────────────
# 22. healthcare.compute_stroke_severity -- output: NIHSSReport
# ─────────────────────────────────────────────────────────────────────
#
# NIH Stroke Scale (NIHSS): 15-item 0-42 stroke deficit score. Validated
# correlates: <5 minor, 5-15 moderate, 16-20 mod-severe, ≥21 severe.
# LVO (large-vessel occlusion) prediction: NIHSS ≥6 in anterior circulation
# triggers EVT eligibility consideration.

class NIHSSItem(BaseModel):
    """One NIHSS item with 0-4 scoring."""
    item: Literal[
        "loc_responsiveness",        # 1a: 0-3
        "loc_questions",              # 1b: 0-2
        "loc_commands",               # 1c: 0-2
        "best_gaze",                  # 2: 0-2
        "visual_fields",              # 3: 0-3
        "facial_palsy",               # 4: 0-3
        "motor_arm_left",             # 5a: 0-4
        "motor_arm_right",            # 5b: 0-4
        "motor_leg_left",             # 6a: 0-4
        "motor_leg_right",            # 6b: 0-4
        "limb_ataxia",                # 7: 0-2
        "sensory",                    # 8: 0-2
        "best_language",              # 9: 0-3
        "dysarthria",                 # 10: 0-2
        "extinction_inattention",     # 11: 0-2
    ]
    points: int = Field(..., ge=0, le=4)
    rationale: str


class NIHSSReport(BaseModel):
    """Output of compute_stroke_severity."""
    patient_id: str | None
    score_total: int = Field(..., ge=0, le=42)
    items: list[NIHSSItem]
    severity_tier: Literal["minor", "moderate", "moderate_severe", "severe"]
    lvo_suspected: bool                 # NIHSS ≥6 + cortical signs
    recommended_response: Literal[
        "outpatient_workup",
        "stroke_unit_admission",
        "thrombolysis_evaluation",
        "endovascular_thrombectomy_evaluation",
    ]
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Brott T et al. Measurements of acute cerebral infarction: a "
            "clinical examination scale. Stroke 1989;20:864-870.",
            "AHA/ASA 2019 Guidelines for the Early Management of Patients "
            "with Acute Ischemic Stroke (Powers WJ et al., Stroke 2019;50:e344).",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 23. healthcare.compute_stroke_thrombolysis_eligibility -- output: ThrombolysisDecision
# ─────────────────────────────────────────────────────────────────────

class ThrombolysisDecision(BaseModel):
    """Output of compute_stroke_thrombolysis_eligibility."""
    patient_id: str | None
    last_known_well_minutes_ago: int = Field(..., ge=0)
    nihss_total: int = Field(..., ge=0, le=42)
    iv_tpa_eligible: bool
    iv_tpa_window: Literal["0_3h", "3_4_5h", "outside_window"]
    iv_tpa_inclusion_met: list[str] = Field(default_factory=list)
    iv_tpa_exclusion_present: list[str] = Field(default_factory=list)
    evt_eligible: bool                   # endovascular thrombectomy
    evt_window: Literal["0_6h", "6_24h_dawn_defuse", "outside_window"]
    evt_criteria_met: list[str] = Field(default_factory=list)
    evt_criteria_unmet: list[str] = Field(default_factory=list)
    decision: Literal[
        "iv_tpa_only",
        "evt_only",
        "iv_tpa_plus_evt",
        "no_reperfusion_supportive_care",
        "abstain_clinician_decision",
    ]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "AHA/ASA 2019 Guidelines for the Early Management of AIS.",
            "Nogueira RG et al. Thrombectomy 6 to 24 Hours after Stroke "
            "with a Mismatch (DAWN). NEJM 2018;378:11.",
            "Albers GW et al. Thrombectomy for Stroke at 6 to 16 Hours with "
            "Selection by Perfusion Imaging (DEFUSE 3). NEJM 2018;378:708.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 24. healthcare.compute_heart_score -- output: HEARTScore
# ─────────────────────────────────────────────────────────────────────
#
# HEART score: ED chest pain rule-out tool (Six AJ et al. 2008). 5 items
# (History, ECG, Age, Risk factors, initial Troponin), 0-2 each, total 0-10.
# Risk bands: 0-3 low (~1.7% MACE/30d), 4-6 moderate (~16.6%), 7-10 high (~50%).

class HEARTScore(BaseModel):
    """Output of compute_heart_score."""
    patient_id: str | None
    history_points: int = Field(..., ge=0, le=2)
    ecg_points: int = Field(..., ge=0, le=2)
    age_points: int = Field(..., ge=0, le=2)
    risk_factors_points: int = Field(..., ge=0, le=2)
    troponin_points: int = Field(..., ge=0, le=2)
    total_score: int = Field(..., ge=0, le=10)
    risk_band: Literal["low", "moderate", "high"]
    estimated_30d_mace_risk_pct: float
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Six AJ et al. Chest pain in the emergency room: value of the "
            "HEART score. Neth Heart J 2008;16:191-196.",
            "Backus BE et al. A prospective validation of the HEART score "
            "for chest pain patients at the emergency department. "
            "Int J Cardiol 2013;168:2153.",
        ],
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 25. healthcare.compute_acs_disposition_decision -- output: ACSDisposition
# ─────────────────────────────────────────────────────────────────────

class ACSDisposition(BaseModel):
    """Output of compute_acs_disposition_decision."""
    patient_id: str | None
    heart_score_total: int = Field(..., ge=0, le=10)
    risk_band: Literal["low", "moderate", "high"]
    has_stemi: bool                            # ECG STEMI criteria met
    has_dynamic_troponin: bool                 # rise/fall pattern
    has_high_risk_features: bool               # ongoing pain, hemodynamic instability, etc.
    disposition: Literal[
        "discharge_with_outpatient_followup",
        "ed_observation_serial_troponin",
        "admit_telemetry_for_workup",
        "cath_lab_activation_immediate",
    ]
    recommended_followup_hours: int
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Gulati M et al. 2021 AHA/ACC/ASE Guideline for the Evaluation "
            "and Diagnosis of Chest Pain. Circulation 2021;144:e368.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 26. healthcare.compute_maternal_early_warning -- output: MEOWSReport
# ─────────────────────────────────────────────────────────────────────
#
# MEOWS -- Modified Obstetric Early Warning Score. Pregnancy-specific physiology:
# normal pregnancy has higher HR (~85-100 normal in 3rd trimester) and lower
# DBP (~60-80 normal); standard NEWS2 over-triggers in pregnant women.

class MEOWSReport(BaseModel):
    """Output of compute_maternal_early_warning."""
    patient_id: str | None
    gestational_age_weeks: int = Field(..., ge=0, le=44)
    pregnancy_phase: Literal["antepartum", "intrapartum", "postpartum"]
    score_total: int = Field(..., ge=0, le=20)
    severity_tier: Literal["green", "yellow", "red"]   # MEOWS color codes
    recommended_response: Literal[
        "routine_obs_4h",
        "increase_to_hourly_obs",
        "urgent_obstetric_review",
        "obstetric_emergency_response",
    ]
    contributing_parameters: list[str] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Singh S et al. A validation study of the CEMACH recommended "
            "modified early obstetric warning system (MEOWS). Anaesthesia "
            "2012;67:12-18.",
            "Royal College of Obstetricians and Gynaecologists Greentop "
            "Guideline 56: Maternal Collapse in Pregnancy and the Puerperium.",
        ],
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 27. healthcare.compute_preeclampsia_assessment -- output: PreeclampsiaReport
# ─────────────────────────────────────────────────────────────────────

class PreeclampsiaReport(BaseModel):
    """Output of compute_preeclampsia_assessment."""
    patient_id: str | None
    gestational_age_weeks: int = Field(..., ge=0, le=44)
    systolic_bp: float
    diastolic_bp: float
    proteinuria_present: bool
    severity_features_present: list[str] = Field(default_factory=list)
    classification: Literal[
        "no_preeclampsia",
        "gestational_hypertension",
        "preeclampsia_without_severe_features",
        "preeclampsia_with_severe_features",
        "eclampsia",
        "hellp_syndrome",
    ]
    delivery_recommended: bool
    magnesium_sulfate_indicated: bool
    antihypertensive_indicated: bool
    recommended_disposition: Literal[
        "outpatient_close_followup",
        "antepartum_admission_for_monitoring",
        "labor_and_delivery_for_workup",
        "labor_and_delivery_for_delivery",
        "icu_obstetric_anesthesia_consult",
    ]
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "ACOG Practice Bulletin No. 222 (2020): Gestational Hypertension "
            "and Preeclampsia. Obstet Gynecol 2020;135:e237.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 28. healthcare.compute_falls_risk_morse -- output: MorseFallsReport
# ─────────────────────────────────────────────────────────────────────
#
# Morse Falls Scale: 6-item inpatient fall-risk score, 0-125. Banks:
# 0-24 low risk, 25-44 moderate, ≥45 high risk -> falls precautions.

class MorseFallsReport(BaseModel):
    """Output of compute_falls_risk_morse."""
    patient_id: str | None
    history_falls_points: int                  # 0 or 25
    secondary_diagnosis_points: int            # 0 or 15
    ambulatory_aid_points: int                 # 0 / 15 / 30
    iv_or_heparin_lock_points: int             # 0 or 20
    gait_points: int                            # 0 / 10 / 20
    mental_status_points: int                  # 0 (oriented) or 15 (forgets limitations)
    score_total: int = Field(..., ge=0, le=125)
    risk_tier: Literal["low", "moderate", "high"]
    recommended_intervention: Literal[
        "routine_safety",
        "standard_falls_precautions",
        "high_risk_falls_protocol",
    ]
    contributing_medications_flagged: list[str] = Field(default_factory=list)
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Morse JM. Preventing Patient Falls: Establishing a Fall "
            "Intervention Program. Springer 2009.",
            "AGS Beers Criteria 2023 (medications increasing falls risk).",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 29. healthcare.compute_delirium_screening_cam -- output: CAMReport
# ─────────────────────────────────────────────────────────────────────

class CAMReport(BaseModel):
    """Output of compute_delirium_screening_cam."""
    patient_id: str | None
    feature1_acute_onset_or_fluctuating: bool
    feature2_inattention: bool
    feature3_disorganized_thinking: bool
    feature4_altered_consciousness: bool
    cam_positive: bool                          # F1 + F2 + (F3 OR F4)
    delirium_subtype: Literal[
        "hyperactive", "hypoactive", "mixed", "not_applicable",
    ] = "not_applicable"
    contributing_factors: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Inouye SK et al. Clarifying confusion: the confusion assessment "
            "method. A new method for detection of delirium. Ann Intern Med "
            "1990;113:941-948.",
            "Wei LA et al. The Confusion Assessment Method (CAM): a "
            "systematic review. JAGS 2008;56:823.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 30. healthcare.compute_trauma_severity_score -- output: TraumaSeverityReport
# ─────────────────────────────────────────────────────────────────────
#
# Composite trauma severity:
#   - Injury Severity Score (ISS): sum of squares of the top-3 AIS by body
#     region (0-75); ≥16 = major trauma.
#   - Revised Trauma Score (RTS): GCS + SBP + RR coded weights, 0-12,
#     <12 = abnormal physiology.

class TraumaInjury(BaseModel):
    """One AIS-coded injury per body region."""
    body_region: Literal[
        "head_neck", "face", "chest", "abdomen_pelvis",
        "extremities_pelvic_girdle", "external",
    ]
    ais_severity: int = Field(..., ge=1, le=6)  # 1=minor, 6=unsurvivable
    description: str | None = None


class TraumaSeverityReport(BaseModel):
    """Output of compute_trauma_severity_score."""
    patient_id: str | None
    iss: int = Field(..., ge=0, le=75)
    rts: float = Field(..., ge=0.0, le=12.0)
    iss_band: Literal["minor", "moderate", "major", "severe", "unsurvivable"]
    rts_band: Literal["normal_physiology", "abnormal_physiology", "critical"]
    injuries: list[TraumaInjury]
    triage_priority: Literal[
        "minor_injury_outpatient",
        "trauma_team_activation",
        "trauma_center_transfer",
        "operating_room_immediate",
    ]
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Baker SP et al. The Injury Severity Score. J Trauma 1974;14:187.",
            "Champion HR et al. A Revision of the Trauma Score (RTS). "
            "J Trauma 1989;29:623.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 31. healthcare.compute_massive_transfusion_protocol -- output: MTPDecision
# ─────────────────────────────────────────────────────────────────────
#
# ABC score (Assessment of Blood Consumption, Nunez 2009): 4 yes/no items,
# ≥2 -> activate massive transfusion. Targets a 1:1:1 plasma:platelets:RBC
# ratio per the PROPPR trial (Holcomb 2015).

class MTPDecision(BaseModel):
    """Output of compute_massive_transfusion_protocol."""
    patient_id: str | None
    abc_score: int = Field(..., ge=0, le=4)
    abc_components: list[str]              # which items scored
    mtp_activated: bool
    target_ratio: str                       # "1:1:1 plasma:platelets:RBC"
    estimated_initial_request: dict[str, int]  # # units of each
    txa_indicated: bool                    # tranexamic acid within 3h of injury
    txa_window_open: bool                   # within 3h
    additional_recommendations: list[str] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Nunez TC et al. Early prediction of MTP need: ABC score. "
            "J Trauma 2009;66:346.",
            "Holcomb JB et al. PROPPR randomized trial. JAMA 2015;313:471.",
            "CRASH-2 trial. Lancet 2010;376:23 (TXA in trauma hemorrhage).",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 32. healthcare.compute_dka_severity -- output: DKASeverityReport
# ─────────────────────────────────────────────────────────────────────
#
# ADA / pediatric ESPE-LWPES classification of DKA severity:
#   mild     pH 7.25-7.30  bicarb 15-18  alert
#   moderate pH 7.00-7.24  bicarb 10-15  alert / drowsy
#   severe   pH < 7.00     bicarb < 10   stupor / coma; ICU admission

class DKASeverityReport(BaseModel):
    """Output of compute_dka_severity."""
    patient_id: str | None
    ph: float
    bicarbonate_meq_l: float
    glucose_mg_dl: float
    anion_gap: float | None = None
    ketones_present: bool
    mental_status: Literal["alert", "drowsy", "stupor_coma"]
    severity: Literal["mild", "moderate", "severe", "not_dka"]
    icu_admission_indicated: bool
    fluid_protocol: str           # initial NS bolus + maintenance rate suggestion
    insulin_protocol: str         # 0.1 U/kg/h IV continuous, etc.
    potassium_replacement_at_initiation: bool
    bicarbonate_indicated: bool
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Kitabchi AE et al. ADA Position Statement: Hyperglycemic Crises. "
            "Diabetes Care 2009;32:1335.",
            "Wolfsdorf JI et al. ISPAD Consensus Guideline: DKA / Hyperglycemic "
            "Hyperosmolar State. Pediatr Diabetes 2018;19(S27):155.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 33. healthcare.compute_inpatient_glycemic_control -- output: InpatientGlycemicPlan
# ─────────────────────────────────────────────────────────────────────

class InpatientGlycemicPlan(BaseModel):
    """Output of compute_inpatient_glycemic_control."""
    patient_id: str | None
    target_range_mg_dl: tuple[int, int]
    average_glucose_24h: float | None
    n_hypoglycemic_episodes_24h: int
    n_severe_hyperglycemic_episodes_24h: int    # >300
    current_regimen: str
    recommended_regimen: str
    basal_dose_change_pct: float                 # signed
    correctional_scale_change: str | None = None
    hypoglycemia_risk: Literal["low", "moderate", "high"]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "ADA Standards of Care: 16. Diabetes Care in the Hospital. "
            "Diabetes Care 2024;47(Suppl 1):S295.",
            "Endocrine Society Clinical Practice Guideline: Management of "
            "Hyperglycemia in Hospitalized Adult Patients. JCEM 2022.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 34. healthcare.compute_imaging_appropriateness -- output: ImagingAppropriateness
# ─────────────────────────────────────────────────────────────────────
#
# ACR Appropriateness Criteria + Choosing Wisely lens. Maps clinical
# scenario -> ranked imaging options with appropriateness rating (1-9):
# 1-3 usually not appropriate, 4-6 may be appropriate, 7-9 usually
# appropriate.

class ImagingOption(BaseModel):
    """One imaging modality with appropriateness rating."""
    modality: str                          # "CT chest with IV contrast", etc.
    appropriateness_rating: int = Field(..., ge=1, le=9)
    radiation_dose_msv: float              # estimated effective dose
    iv_contrast_required: bool
    rank: int = Field(..., ge=1)
    rationale: str
    relative_cost_tier: Literal["low", "medium", "high"]


class ImagingAppropriateness(BaseModel):
    """Output of compute_imaging_appropriateness."""
    patient_id: str | None
    clinical_scenario: str
    options: list[ImagingOption]
    top_pick_modality: str | None
    cumulative_radiation_caution: bool
    pediatric_alara_caution: bool
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "American College of Radiology. ACR Appropriateness Criteria. "
            "https://www.acr.org/Clinical-Resources/ACR-Appropriateness-Criteria",
            "Choosing Wisely campaign -- ACR/SNMMI/ABIM Foundation lists.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 35. healthcare.compute_contrast_safety_check -- output: ContrastSafetyReport
# ─────────────────────────────────────────────────────────────────────

class ContrastSafetyReport(BaseModel):
    """Output of compute_contrast_safety_check."""
    patient_id: str | None
    contrast_type: Literal["iodinated_iv", "gadolinium_iv", "oral", "no_contrast"]
    egfr_ml_min: float | None
    contrast_induced_nephropathy_risk: Literal["low", "moderate", "high", "contraindicated"]
    nsf_risk_for_gadolinium: Literal["low", "moderate", "contraindicated", "n_a"]
    metformin_hold_recommended: bool
    metformin_hold_duration_hours: int = Field(...)  # must be explicitly set: 0 if no hold, otherwise duration in hours
    iodine_allergy_documented: bool
    premedication_recommended: bool         # corticosteroids + diphenhydramine
    pregnancy_caution: bool
    pre_hydration_recommended: bool
    proceed_with_contrast: bool
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "ACR Manual on Contrast Media (2024 ed.).",
            "Davenport MS et al. Use of IV iodinated contrast media in patients "
            "with kidney disease: consensus statements from the ACR & NKF. "
            "Radiology 2020;294:660.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# 36. healthcare.compute_aki_kdigo_stage -- output: AKIStagingReport
# ─────────────────────────────────────────────────────────────────────
#
# KDIGO AKI staging:
#   Stage 1: Cr ↑ ≥0.3 mg/dL within 48h, OR Cr ↑ to 1.5-1.9× baseline within 7 days,
#            OR UOP <0.5 mL/kg/h × 6-12h
#   Stage 2: Cr ↑ to 2.0-2.9× baseline,
#            OR UOP <0.5 mL/kg/h × ≥12h
#   Stage 3: Cr ↑ to ≥3.0× baseline OR Cr ≥4.0 OR initiation of RRT,
#            OR UOP <0.3 mL/kg/h ≥24h OR anuria ≥12h

class AKIStagingReport(BaseModel):
    """Output of compute_aki_kdigo_stage."""
    patient_id: str | None
    creatinine_baseline_mg_dl: float
    creatinine_current_mg_dl: float
    creatinine_change_ratio: float
    creatinine_change_absolute: float
    urine_output_ml_per_kg_per_hour: float | None
    aki_stage: Literal["no_aki", "stage_1", "stage_2", "stage_3"]
    aki_etiology_clue: Literal["pre_renal", "intrinsic", "post_renal", "unclear"] = "unclear"
    nephrotoxin_review_recommended: bool
    nephrology_consult_indicated: bool
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "KDIGO Clinical Practice Guideline for Acute Kidney Injury. "
            "Kidney Int Suppl 2012;2:1.",
        ],
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# 37. healthcare.compute_dialysis_initiation_decision -- output: DialysisInitiationReport
# ─────────────────────────────────────────────────────────────────────
#
# AEIOU mnemonic for emergent dialysis indications:
#   A -- Acidosis (refractory metabolic acidosis, pH < 7.1)
#   E -- Electrolytes (hyperkalemia >6.5 refractory to medical management)
#   I -- Ingestion (dialyzable toxins: methanol, ethylene glycol, salicylates,
#       lithium, theophylline)
#   O -- Overload (volume overload refractory to diuretics)
#   U -- Uremia (encephalopathy, pericarditis, bleeding diathesis)

class DialysisInitiationReport(BaseModel):
    """Output of compute_dialysis_initiation_decision."""
    patient_id: str | None
    aeiou_indications_met: list[str]
    dialysis_indicated: bool
    urgency: Literal["non_urgent", "urgent_within_24h", "emergent_immediately"]
    modality_suggested: Literal["intermittent_hd", "crrt", "peritoneal_dialysis", "n_a"] | None
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "KDIGO 2012 AKI Guidelines, recommendation 5.1.",
            "STARRT-AKI Investigators. Timing of Initiation of RRT in AKI. "
            "NEJM 2020;383:240.",
        ],
    )


class TumorResponse(BaseModel):
    """Output of compute_oncology_treatment_response (RECIST 1.1)."""
    patient_id: str | None
    target_lesions: list[TargetLesion]
    baseline_sum_mm: float
    current_sum_mm: float
    sum_change_pct: float                # signed; negative = regression
    new_lesions_present: bool = False
    non_target_progression: bool = False
    overall_response: Literal[
        "complete_response",     # CR -- disappearance of all lesions
        "partial_response",      # PR -- ≥30% decrease in sum
        "stable_disease",        # SD -- between PR and PD
        "progressive_disease",   # PD -- ≥20% increase OR new lesions
        "not_evaluable",
    ]
    decision_implication: Literal[
        "continue_current_therapy",
        "consider_dose_escalation",
        "switch_therapy",
        "transition_palliative",
        "abstain_repeat_imaging",
    ]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Eisenhauer EA, et al. New response evaluation criteria in solid "
            "tumours: revised RECIST guideline (version 1.1). Eur J Cancer "
            "2009;45:228-247.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Healthcare economics (IMPACT-1)
# ─────────────────────────────────────────────────────────────────────

class InterventionEvidence(BaseModel):
    """Evidence-based effect estimate for a proposed clinical intervention.

    Either `relative_risk_reduction` OR `absolute_risk_reduction` must be set.
    When both are provided, ARR wins (less assumption-laden).
    """
    name: str
    relative_risk_reduction: float | None = Field(default=None, ge=0.0, le=1.0)
    absolute_risk_reduction: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_per_patient_usd: float = Field(ge=0.0)
    qaly_gained_per_avoided_event: float | None = Field(default=None, ge=0.0)
    horizon_days: int = Field(default=30, ge=1, le=3650)
    evidence_grade: Literal["A", "B", "C", "D", "expert_opinion"] = "B"
    citation: str | None = None


class ImpactKPIs(BaseModel):
    """Aggregate impact metrics over a cohort of DecisionCards (IMPACT-2).

    Each KPI is a point estimate under the modeling assumption encoded in
    `intervention_relative_risk_reduction` (the RRR attributed to any non-
    `discharge_home` recommendation). Treat as illustrative -- the
    cost-effectiveness verdict on individual decisions is more rigorously
    handled by `compute_expected_value_of_intervention`.
    """
    n_decisions: int = Field(ge=0)
    n_with_recommendation: int = Field(ge=0)
    n_abstained: int = Field(ge=0)
    action_counts: dict[str, int]
    confidence_counts: dict[str, int]
    avg_baseline_risk: float = Field(ge=0.0, le=1.0)
    avg_post_intervention_risk: float = Field(ge=0.0, le=1.0)
    intervention_relative_risk_reduction: float = Field(ge=0.0, le=1.0)
    estimated_events_avoided: float = Field(ge=0.0)
    avoided_event_cost_per_event_usd: float = Field(ge=0.0)
    estimated_cost_avoided_usd: float = Field(ge=0.0)
    n_critic_force_abstain: int = Field(ge=0)
    n_critic_downgrade: int = Field(ge=0)
    confidence_caveat: str = (
        "These are point estimates under a fixed RRR assumption -- not RCT-"
        "grade efficacy. Use compute_expected_value_of_intervention for "
        "decision-grade cost-effectiveness analysis on a specific cohort."
    )
    generated_at_iso: str


class CostEffectivenessReport(BaseModel):
    """Output of compute_expected_value_of_intervention.

    All monetary values are in USD. Per-patient values are scaled to the
    cohort by `cohort_size`. The `decision` field encodes the standard
    health-economics ladder: cost-saving -> cost-effective -> not -- modulated
    by evidence grade (low evidence escalates to `uncertain_evidence`).
    """
    intervention_name: str
    cohort_size: int = Field(ge=1)
    baseline_event_probability: float = Field(ge=0.0, le=1.0)
    absolute_risk_reduction: float = Field(ge=0.0, le=1.0)
    number_needed_to_treat: float | None = Field(default=None, ge=1.0)
    expected_events_avoided: float = Field(ge=0.0)
    intervention_cost_total_usd: float = Field(ge=0.0)
    avoided_event_cost_total_usd: float = Field(ge=0.0)
    net_cost_total_usd: float
    cost_per_event_avoided_usd: float | None = None
    cost_per_qaly_usd: float | None = None
    wtp_threshold_per_qaly_usd: float = Field(default=100_000.0, ge=0.0)
    decision: Literal[
        "cost_saving",          # net cost negative (avoided cost > intervention cost)
        "cost_effective",       # ICER < WTP threshold
        "not_cost_effective",   # ICER ≥ WTP threshold
        "uncertain_evidence",   # evidence_grade C/D -- no decision
        "dominated",            # less effective AND more costly than comparator
    ]
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Sanders GD, et al. Recommendations for Conduct, Methodological "
            "Practices, and Reporting of Cost-effectiveness Analyses (Second "
            "Panel). JAMA 2016;316:1093.",
            "Neumann PJ, et al. Updating Cost-Effectiveness -- The Curious "
            "Resilience of the $50,000-per-QALY Threshold. NEJM 2014;371:796.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Conversational SHARP resolver (LLM-3) + DDx ranker (LLM-4)
# ─────────────────────────────────────────────────────────────────────

class PatientFeatureExtraction(BaseModel):
    """Structured features extracted from a natural-language patient query.

    The deterministic extractor is the fallback; the LLM-enhanced extractor
    populates the same shape so downstream consumers don't branch.
    """
    chief_complaint_terms: list[str] = Field(default_factory=list)
    age_min_years: int | None = Field(default=None, ge=0, le=130)
    age_max_years: int | None = Field(default=None, ge=0, le=130)
    sex: Literal["male", "female", "other"] | None = None
    time_anchor_hours: int = Field(default=24, ge=1, le=720)
    care_role: Literal["any", "admitting", "attending", "consultant"] = "any"
    confidence: Literal["high", "medium", "low"] = "low"
    extraction_method: Literal["llm", "regex"] = "regex"


class PatientCandidate(BaseModel):
    patient_id: str
    match_score: float = Field(ge=0.0, le=1.0)
    matched_features: list[str]
    encounter_start: str | None = None
    chief_complaint: str | None = None
    rationale: str


# ─────────────────────────────────────────────────────────────────────
# A2A INPUT_REQUIRED lifecycle (Phase 3.1)
# ─────────────────────────────────────────────────────────────────────
#
# Some tool calls cannot complete deterministically without a follow-up
# clarification from the client. Rather than terminating the task with
# `abstain_recommended=True`, the tool sets `task_state="input_required"`
# and populates `clarification_request`. The BYO orchestrator
# (any A2A v1 chat client) reads `task_state`, asks the user the
# clarification question, and re-runs the tool with the extended
# context (`provided_clarification` field on the request).

ClarificationAnswerKind = Literal[
    "patient_id",              # one of N candidates by ID / name
    "narrower_diagnosis",      # a specific dx among a vague DDx
    "missing_observation",     # the value of a specific lab / vital
    "specific_drug_name",      # which medication the user means
    "free_text",               # any free-text follow-up
    "yes_no",                  # boolean confirmation
]


class ClarificationRequest(BaseModel):
    """A structured follow-up question emitted when a tool transitions
    to A2A INPUT_REQUIRED state.

    The `expected_answer_kind` lets the BYO orchestrator render the
    clarification appropriately (e.g. as a multiple-choice picker for
    `patient_id`, a free-text input for `free_text`, etc.).
    """
    question: str
    expected_answer_kind: ClarificationAnswerKind
    candidates: list[str] | None = Field(
        default=None,
        description=(
            "When `expected_answer_kind` enumerates options "
            "(e.g. patient_id, narrower_diagnosis, specific_drug_name) "
            "list the candidate values here."
        ),
    )
    cite_back_section: str | None = Field(
        default=None,
        description=(
            "Optional pointer back to the upstream artefact section "
            "that triggered the clarification."
        ),
    )


# Used by the DDx ranker + patient resolver in their result schemas.
TaskState = Literal["completed", "input_required", "auth_required"]


class AuthRequiredHint(BaseModel):
    """Returned in the body of a 401 response when the FHIR upstream
    returned 401 mid-task and the auto-refresh attempt failed (or no
    refresh token was supplied by the SHARP client).

    The BYO orchestrator (any A2A v1 chat client) reads this
    payload, prompts the user / EHR to re-authorise, and re-runs the
    task with a fresh access token.
    """
    error: Literal["auth_required"] = "auth_required"
    message: str
    refresh_attempted: bool = False
    refresh_failure_reason: str | None = None
    fhir_server_url: str | None = None
    spec_reference: str = (
        "https://docs.promptopinion.ai/fhir-context/mcp-fhir-context.html"
    )


class PatientResolutionResult(BaseModel):
    """Output of compute_resolve_patient_from_query (LLM-3).

    When the resolver cannot pick a top match (score < 0.6 OR
    margin < 0.2), it transitions to A2A INPUT_REQUIRED state via
    `task_state` + populates `clarification_request` with a follow-up
    question listing the candidates.
    """
    query_text: str
    extracted_features: PatientFeatureExtraction
    candidates: list[PatientCandidate]
    n_candidates: int = Field(ge=0)
    resolved_patient_id: str | None = None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    task_state: TaskState = "completed"
    clarification_request: ClarificationRequest | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "SHARP-on-MCP spec -- https://www.sharponmcp.com/key-components.html",
            "FHIR R4 Search Parameters -- https://www.hl7.org/fhir/search.html",
            "A2A Protocol v1.0 -- INPUT_REQUIRED state semantics.",
        ],
    )


class DifferentialItem(BaseModel):
    """One ranked entry in a differential diagnosis list."""
    diagnosis: str
    probability_estimate: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    supporting_features: list[str]
    contradicting_features: list[str] = Field(default_factory=list)
    citations: list[EvidenceSource] = Field(default_factory=list)
    grounding_verdict: Literal[
        "supported", "partially_supported", "unsupported", "ungrounded",
    ] = "ungrounded"
    ground_claim_text: str | None = None


class EquitySegment(BaseModel):
    """Per-subgroup equity slice across a cohort of decisions."""
    subgroup_dimension: Literal["age_band", "race", "sex", "insurance_type"]
    subgroup_value: str
    n_decisions: int = Field(ge=0)
    action_counts: dict[str, int]
    n_abstained: int = Field(ge=0)
    n_with_risk: int = Field(default=0, ge=0)
    avg_risk: float | None = Field(default=None, ge=0.0, le=1.0)
    intervention_rate: float = Field(ge=0.0, le=1.0)  # non-discharge_home / total
    abstention_rate: float = Field(ge=0.0, le=1.0)
    confidence_high_rate: float = Field(ge=0.0, le=1.0)


class EquityDashboard(BaseModel):
    """Output of compute_equity_dashboard (SIM-2)."""
    n_total_decisions: int = Field(ge=0)
    segments: list[EquitySegment]
    max_intervention_rate_disparity: float = Field(ge=0.0)
    max_abstention_rate_disparity: float = Field(ge=0.0)
    max_avg_risk_disparity: float = Field(ge=0.0)
    rationale: str
    flagged_segments: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Disparity metrics are descriptive -- not causal. Disparity does NOT "
        "imply algorithmic bias on its own; investigate confounders, "
        "indication, and access patterns before action."
    )
    references: list[str] = Field(
        default_factory=lambda: [
            "AHRQ. Healthcare Disparities and Inequalities Report.",
            "Obermeyer Z, et al. Dissecting racial bias in an algorithm used "
            "to manage the health of populations. Science 2019;366:447.",
        ],
    )


class SyntheaCohortReport(BaseModel):
    """Output of evaluate_synthea_cohort (SCALE-1)."""
    cohort_label: str
    n_patients: int = Field(ge=0)
    n_with_readmission_label: int = Field(ge=0)
    n_readmitted: int = Field(ge=0)
    base_rate_readmission: float = Field(ge=0.0, le=1.0)
    ece: float | None = Field(default=None, ge=0.0)
    auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    brier_score: float | None = Field(default=None, ge=0.0)
    calibration_buckets: dict[str, dict[str, float]] = Field(default_factory=dict)
    coefficients_version: str = ""
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Walonoski J, et al. Synthea: An approach, method, and software "
            "mechanism for generating synthetic patients and the synthetic "
            "electronic health care record. JAMIA 2018;25:230.",
            "Walraven C, et al. LACE index. CMAJ 2010;182:551.",
        ],
    )


class CausalRefutation(BaseModel):
    """One refutation result from a causal-effect estimate."""
    name: Literal[
        "random_common_cause", "placebo_treatment", "data_subset",
    ]
    new_effect: float
    p_value: float | None = None
    detail: str = ""


class CausalATEReport(BaseModel):
    """Output of compute_average_treatment_effect (SCALE-2)."""
    treatment_name: str
    outcome_name: str
    n_observations: int = Field(ge=1)
    n_treated: int = Field(ge=0)
    n_untreated: int = Field(ge=0)
    estimation_method: str
    ate_point: float
    ate_ci95_low: float | None = None
    ate_ci95_high: float | None = None
    ate_std_error: float | None = None
    naive_difference_in_means: float
    identified_estimand_text: str = ""
    refutations: list[CausalRefutation] = Field(default_factory=list)
    assumptions_warnings: list[str] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Pearl J. Causality: Models, Reasoning, and Inference. "
            "2nd ed. Cambridge University Press, 2009.",
            "Sharma A, Kiciman E. DoWhy: An End-to-End Library for Causal "
            "Inference. arXiv:2011.04216.",
        ],
    )


class MDPActionValue(BaseModel):
    """Expected value of one discharge action over the simulated horizon."""
    action: Literal["discharge_home", "home_with_care", "snf",
                       "continued_admission"]
    expected_value: float
    probability_alive_at_horizon: float = Field(ge=0.0, le=1.0)
    probability_readmitted_in_window: float = Field(ge=0.0, le=1.0)
    expected_qaly_in_window: float = Field(ge=0.0)


class MDPDecisionReport(BaseModel):
    """Output of compute_sequential_mdp_value (SCALE-3)."""
    patient_id: str | None = None
    horizon_days: int = Field(ge=1, le=365)
    n_stages: int = Field(ge=1, le=24)
    discount_factor: float = Field(ge=0.0, le=1.0)
    baseline_readmission_30d_prob: float = Field(ge=0.0, le=1.0)
    action_values: list[MDPActionValue]
    optimal_action: Literal["discharge_home", "home_with_care", "snf",
                                "continued_admission"]
    optimal_action_expected_value: float
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Bellman R. Dynamic Programming. Princeton University Press, 1957.",
            "Krumholz HM, et al. Hospital readmission as an accountability "
            "measure. JAMA. 2013;309:587. (used for transition priors)",
        ],
    )


class WorkflowIntent(BaseModel):
    """One node in a conversational orchestration plan."""
    intent_id: Literal[
        "risk_assessment", "counseling", "followup",
        "drift_check", "equity_audit", "differential_diagnosis",
        "cost_effectiveness", "cumulative_impact", "patient_resolution",
    ]
    agent: Literal[
        "trustedrisk-agent", "trustedrisk-scheduler-agent",
        "trustedrisk-alert-agent", "darena-data-agent",
    ]
    description: str
    depends_on: list[str] = Field(default_factory=list)


class WorkflowPlan(BaseModel):
    """Output of plan_workflow (GENAI-5)."""
    natural_query: str
    extraction_method: Literal["keyword", "llm"] = "keyword"
    intents: list[WorkflowIntent]
    execution_order: list[str]
    rationale: str


class WorkflowExecutionStep(BaseModel):
    intent_id: str
    agent: str
    started_at_iso: str
    duration_ms: float
    success: bool
    output_summary: str
    error: str | None = None


class WorkflowExecutionResult(BaseModel):
    plan: WorkflowPlan
    steps: list[WorkflowExecutionStep]
    overall_success: bool
    composite_summary: str


class ToolSearchHit(BaseModel):
    """One result from semantic tool search."""
    tool_name: str
    bundle_memberships: list[str] = Field(default_factory=list)
    score: float = Field(ge=-1.0, le=1.0)
    summary: str
    docstring_excerpt: str


class ToolSearchResult(BaseModel):
    """Output of semantic_search_tools (GENAI-4)."""
    query: str
    n_tools_indexed: int = Field(ge=0)
    n_returned: int = Field(ge=0)
    embedder_model: str
    hits: list[ToolSearchHit]


class DataLineageNode(BaseModel):
    """One node in the data lineage DAG."""
    node_id: str
    node_type: Literal[
        "raw_dataset", "derived_artifact", "coefficient_bundle",
        "code_module", "tool_output", "test_artifact", "external_corpus",
    ]
    description: str
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    last_modified_iso: str | None = None
    path: str | None = None


class DataLineageEdge(BaseModel):
    """One directed edge in the lineage DAG."""
    source_id: str
    target_id: str
    relationship: Literal[
        "ingests", "transforms", "calibrates", "consumed_by",
        "produces", "validates", "documents",
    ]
    description: str = ""


class DataLineageGraph(BaseModel):
    """Output of compute_data_lineage (CMP-3)."""
    generated_at_iso: str
    nodes: list[DataLineageNode]
    edges: list[DataLineageEdge]
    n_nodes: int = Field(ge=0)
    n_edges: int = Field(ge=0)
    integrity_hash: str
    mermaid_text: str
    notes: str = (
        "Lineage generated from filesystem hashes -- assumes the artifacts "
        "haven't been mutated in-place since last regeneration. For "
        "production, persist this graph alongside each release tag and "
        "diff against the previous one in CI."
    )


class CEALadderEntry(BaseModel):
    """One row in the literature cost-effectiveness ladder."""
    intervention: str
    citation: str
    arr_30d_percentage_points: float = Field(ge=0.0, le=100.0)
    cost_per_patient_usd: float = Field(ge=0.0)
    cost_per_event_avoided_usd: float | None = None
    cost_per_qaly_usd: float | None = None
    sample_n: int | None = Field(default=None, ge=0)
    notes: str = ""


class CostEffectivenessLadder(BaseModel):
    """Output of build_cost_effectiveness_ladder (SIM-3)."""
    target_intervention_name: str
    target_arr_30d_percentage_points: float = Field(ge=0.0, le=100.0)
    target_cost_per_patient_usd: float = Field(ge=0.0)
    target_cost_per_event_avoided_usd: float | None = None
    target_cost_per_qaly_usd: float | None = None
    benchmarks: list[CEALadderEntry]
    target_rank_among_benchmarks: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(
        default_factory=lambda: [
            "Schnipper JL, et al. Role of pharmacist counseling in preventing "
            "adverse drug events after hospitalization. Arch Intern Med. "
            "2006;166:565.",
            "Coleman EA, et al. The Care Transitions Intervention: results of "
            "a randomized controlled trial. Arch Intern Med. 2006;166:1822.",
            "Naylor MD, et al. Comprehensive discharge planning and home "
            "follow-up of hospitalized elders: a randomized clinical trial. "
            "JAMA. 1999;281:613.",
            "Misky GJ, et al. Post-hospitalization transitions: examining the "
            "effects of timing of primary care provider follow-up. J Hosp Med. "
            "2010;5:392.",
            "Hansen LO, et al. Interventions to reduce 30-day rehospitalization. "
            "Ann Intern Med. 2011;155:520.",
        ],
    )


class CaseMixSegment(BaseModel):
    """One segment in the hospital case mix used by the outcomes simulator."""
    name: str
    n_patients_per_year: int = Field(ge=1)
    baseline_event_probability: float = Field(ge=0.0, le=1.0)


class SimulationSegmentResult(BaseModel):
    """Per-segment Monte Carlo outcome estimate."""
    name: str
    n_patients: int
    baseline_event_probability: float
    expected_events_no_intervention: float
    expected_events_with_intervention: float
    events_avoided_mean: float
    events_avoided_ci95: tuple[float, float]
    cost_avoided_mean_usd: float
    cost_avoided_ci95_usd: tuple[float, float]
    qaly_gained_mean: float | None = None
    qaly_gained_ci95: tuple[float, float] | None = None


class SensitivityRow(BaseModel):
    """One row of the sensitivity table (varying RRR or event cost)."""
    parameter: Literal["intervention_rrr", "avoided_event_cost_usd"]
    value: float
    events_avoided_mean: float
    cost_avoided_mean_usd: float


class HospitalYearSimulation(BaseModel):
    """Output of simulate_hospital_year (SIM-1)."""
    intervention_name: str
    n_iterations: int = Field(ge=1)
    seed: int
    cohort_size_per_year: int = Field(ge=1)
    intervention_relative_risk_reduction: float = Field(ge=0.0, le=1.0)
    intervention_cost_per_patient_usd: float = Field(ge=0.0)
    avoided_event_cost_usd: float = Field(ge=0.0)
    qaly_gained_per_avoided_event: float | None = None
    segments: list[SimulationSegmentResult]
    total_events_avoided_mean: float
    total_events_avoided_ci95: tuple[float, float]
    total_intervention_cost_usd: float
    total_avoided_event_cost_mean_usd: float
    total_avoided_event_cost_ci95_usd: tuple[float, float]
    net_cost_mean_usd: float
    net_cost_ci95_usd: tuple[float, float]
    total_qaly_gained_mean: float | None = None
    total_qaly_gained_ci95: tuple[float, float] | None = None
    cost_per_qaly_mean_usd: float | None = None
    sensitivity: list[SensitivityRow] = Field(default_factory=list)
    references: list[str] = Field(
        default_factory=lambda: [
            "Briggs AH, et al. Probabilistic sensitivity analysis for "
            "decision-analytic models: a Monte Carlo approach. PharmacoEcon "
            "2008;26:781.",
        ],
    )


class DifferentialDiagnosisReport(BaseModel):
    """Output of compute_differential_diagnosis_ranker (LLM-4).

    When the chief complaint is too vague to produce a meaningful
    differential, the tool transitions to A2A INPUT_REQUIRED state via
    `task_state` and populates `clarification_request` so the BYO
    orchestrator can ask the user to specify the symptom + location.
    """
    chief_complaint: str
    structured_features: dict[str, Any] = Field(default_factory=dict)
    free_text_summary: str | None = None
    items: list[DifferentialItem]
    n_items: int = Field(ge=0)
    extraction_method: Literal["llm", "rule_based"] = "rule_based"
    overall_confidence: Literal["high", "medium", "low"] = "low"
    cant_miss_diagnoses: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    task_state: TaskState = "completed"
    clarification_request: ClarificationRequest | None = None
    references: list[str] = Field(
        default_factory=lambda: [
            "Tintinalli's Emergency Medicine, 9th ed. (general DDx framework).",
            "Bordage G. Why did I miss the diagnosis? Some cognitive "
            "explanations and educational implications. Acad Med. 1999;74:S138.",
            "A2A Protocol v1.0 -- INPUT_REQUIRED state semantics.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Phase 2.1 -- Prior Authorization (PA) tool surface
# ─────────────────────────────────────────────────────────────────────

PAYER_ID = Literal[
    "unitedhealth",
    "anthem_bcbs",
    "aetna",
    "cigna",
    "humana",
    "medicare",
    "medicaid",
    "generic",   # AHRQ-style fallback
]


PARequestedServiceType = Literal[
    "imaging_advanced",         # MRI / CT / PET / nuclear
    "specialty_drug",           # J-code biologic / oncology / DMARD / specialty PO
    "elective_procedure",       # joint replacement / bariatric / spine
    "dme",                      # durable medical equipment
    "home_health",              # post-discharge skilled-nursing referral
    "behavioral_health_admit",  # inpatient psych admission
    "out_of_network_referral",  # specialty referral outside network
    "step_therapy_override",    # request to bypass step therapy
]


class PARequestedService(BaseModel):
    """The service for which prior authorization is being sought."""
    service_type: PARequestedServiceType
    cpt_codes: list[str] = Field(default_factory=list)
    icd10_codes: list[str] = Field(default_factory=list)
    rxnorm_codes: list[str] = Field(default_factory=list)
    j_codes: list[str] = Field(default_factory=list)
    description: str
    requested_quantity: int | None = Field(default=None, ge=0)
    requested_duration_days: int | None = Field(default=None, ge=0)


class PAEvidenceItem(BaseModel):
    """A single piece of supporting evidence with a chart cite-back.

    The cite-back is mandatory: every claim in the PA letter MUST trace
    back to a structured FHIR resource ID or a chart-line span. This
    is the deterministic floor on top of which the LLM may paraphrase.
    """
    kind: Literal[
        "condition", "observation", "medication_history",
        "procedure", "imaging_report", "chart_excerpt",
    ]
    fhir_resource_id: str | None = None
    chart_line_span: tuple[int, int] | None = None
    text: str
    code: str | None = None
    code_system: str | None = None
    timestamp_iso: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)


class PAEvidencePack(BaseModel):
    """Output of compute_pa_evidence_pack -- the structured evidence
    bundle that supports a PA request.

    Aggregated from the FHIR chart + chart-intelligence NER + structured
    requirements per payer. Downstream tools (letter draft, appeal-
    likelihood, payer-rules-match) consume this same pack so every claim
    they emit is grounded.
    """
    patient_reference: str
    requested_service: PARequestedService
    payer: PAYER_ID
    diagnoses: list[PAEvidenceItem] = Field(default_factory=list)
    relevant_observations: list[PAEvidenceItem] = Field(default_factory=list)
    prior_treatments_tried: list[PAEvidenceItem] = Field(
        default_factory=list,
        description=(
            "MedicationRequests / Procedures already attempted -- drives "
            "step-therapy override arguments."
        ),
    )
    contraindications_to_alternatives: list[PAEvidenceItem] = Field(
        default_factory=list,
    )
    chart_excerpts: list[PAEvidenceItem] = Field(default_factory=list)
    evidence_strength_score: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description=(
            "0-1 score combining: count of relevant evidence items, "
            "diagnosis specificity, prior-treatment depth, and "
            "alignment with payer-specific PA criteria."
        ),
    )
    n_evidence_items: int = Field(ge=0, default=0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    rationale: str
    references: list[str] = Field(default_factory=list)


class PALetterParagraph(BaseModel):
    """One paragraph of the PA letter draft with mandatory cite-backs."""
    section: Literal[
        "header", "patient_summary", "diagnosis", "clinical_history",
        "treatments_tried", "medical_necessity", "requested_service",
        "supporting_evidence", "policy_match", "closing",
    ]
    text: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    is_llm_polished: bool = False


class PALetterDraft(BaseModel):
    """Output of compute_pa_letter_draft -- a complete PA submission /
    appeal letter built from a PAEvidencePack.

    Two modes:
      - Deterministic (default): pure-template paragraph rendering
      - LLM-polished (TRUSTEDRISK_PA_LLM_POLISH=1): paraphrase pass
        that preserves structure + cite-backs but improves prose

    In both modes the paragraphs carry the cite-back to the
    PAEvidencePack -- never invents new clinical facts.
    """
    payer: PAYER_ID
    requested_service: PARequestedService
    paragraphs: list[PALetterParagraph]
    full_text: str
    contains_llm_polish: bool = False
    llm_model_id: str | None = None
    n_cite_backs: int = Field(ge=0, default=0)
    coverage_pct: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description=(
            "Fraction of PAEvidencePack items referenced at least once "
            "in the letter -- proxy for evidence utilisation."
        ),
    )
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class PAApprovalEstimate(BaseModel):
    """Output of compute_pa_appeal_likelihood -- calibrated probability
    of approval given an evidence pack + payer + (optional) prior-denials
    history.

    The probability is calibrated against a small payer-specific prior
    (literature denial rates, AMA 2024 PA report). The CI is wide for
    rare-payer / rare-service combinations and narrow when the payer's
    published criteria are met.
    """
    payer: PAYER_ID
    requested_service_type: PARequestedServiceType
    probability_of_approval: float = Field(ge=0.0, le=1.0)
    ci95: tuple[float, float] = Field(
        description=(
            "95% credible interval -- wide when the literature prior "
            "is small or the evidence-strength score is uncertain."
        ),
    )
    confidence: Literal["preferred", "degraded", "abstain_recommended"]
    n_prior_denials: int = Field(ge=0, default=0)
    drivers: list[str] = Field(
        default_factory=list,
        description=(
            "Top features influencing the estimate: evidence strength, "
            "prior-denial pattern, payer denial-rate prior."
        ),
    )
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class PARuleStatus(BaseModel):
    """Status of a single payer-specific PA requirement against an
    evidence pack."""
    rule_id: str
    rule_text: str
    status: Literal["met", "unmet", "partial", "unknown"]
    evidence_ids: list[str] = Field(default_factory=list)
    gap_text: str | None = None


class PARulesMatch(BaseModel):
    """Output of compute_pa_payer_rules_match -- per-payer requirement
    checklist against the evidence pack. The matched rules feed the
    `medical_necessity` paragraph of the letter draft."""
    payer: PAYER_ID
    requested_service: PARequestedServiceType
    rules: list[PARuleStatus]
    n_met: int = Field(ge=0, default=0)
    n_unmet: int = Field(ge=0, default=0)
    n_partial: int = Field(ge=0, default=0)
    overall_alignment: Literal[
        "fully_aligned", "majority_aligned", "partially_aligned",
        "weakly_aligned", "misaligned",
    ]
    next_steps: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete actions to close gaps before submitting "
            "(e.g. order CMP within 90 days; document failed step "
            "therapy in chart)."
        ),
    )
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 2.2 -- Scribe agent (clinical documentation drafting)
# ─────────────────────────────────────────────────────────────────────

ClinicalNoteType = Literal[
    "progress_note",          # daily inpatient progress (SOAP)
    "discharge_summary",      # Joint Commission discharge summary
    "consult_letter",         # specialty consult / referral letter
    "admission_hnp",          # admission History and Physical
]


class ClinicalNoteSection(BaseModel):
    """One section of a clinical note draft.

    Cite-backs are mandatory: every clinical claim in the body must
    trace to an FHIR resource ID or a chart line span. The LLM polish
    layer paraphrases prose without inventing new facts; the cite-back
    list is the immutable spine.
    """
    section_id: str
    title: str
    body: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    is_llm_polished: bool = False


class ClinicalNoteDraft(BaseModel):
    """Output of every Phase-2.2 scribe-agent tool.

    Two-tier composition:
      - Deterministic floor (always): pure-template per-section
        rendering from the structured FHIR Bundle + DecisionCard
        + chart excerpts.
      - Optional LLM polish (TRUSTEDRISK_SCRIBE_LLM_POLISH=1):
        paraphrase pass that preserves the section structure +
        cite-back map but improves prose flow.
    """
    note_type: ClinicalNoteType
    patient_reference: str
    encounter_reference: str | None = None
    authored_at_iso: str | None = None
    sections: list[ClinicalNoteSection]
    full_text: str
    contains_llm_polish: bool = False
    llm_model_id: str | None = None
    n_cite_backs: int = Field(ge=0, default=0)
    coverage_pct: float = Field(
        ge=0.0, le=1.0, default=0.0,
        description=(
            "Fraction of structured-input items referenced at least "
            "once across all sections. Proxy for evidence utilisation."
        ),
    )
    estimated_reading_minutes: float = Field(ge=0.0, default=0.0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 2.3 -- Patient-side Q&A agent
# ─────────────────────────────────────────────────────────────────────

class PatientQAItem(BaseModel):
    """A single Q&A pair from compute_discharge_qa.

    The cite-back field references which section of the upstream
    DecisionCard / discharge summary the answer is grounded in. The
    LLM (when polish is enabled) paraphrases the deterministic answer
    body without inventing new clinical facts.
    """
    question: str
    answer: str
    cite_back_section: str | None = None
    source_label: str | None = None
    is_llm_polished: bool = False


class PatientQADraft(BaseModel):
    """Output of compute_discharge_qa -- one or more Q&A items grounded
    on the patient's DecisionCard + discharge summary.

    `multi_turn_context_id` carries through the A2A `contextId` so a
    BYO agent (A2A v1 chat client) can compose this tool across turns
    without losing context.
    """
    patient_reference: str
    encounter_reference: str | None = None
    items: list[PatientQAItem]
    n_items: int = Field(ge=0, default=0)
    multi_turn_context_id: str | None = None
    contains_llm_polish: bool = False
    llm_model_id: str | None = None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


WhatIfScenario = Literal[
    "miss_one_dose",
    "double_dose_by_mistake",
    "take_with_food",
    "take_with_alcohol",
    "take_with_grapefruit",
    "take_with_otc_nsaid",
    "take_with_otc_antacid",
    "take_with_supplements",
    "stop_abruptly",
    "interaction_with_other_chronic_med",
]


class MedicationWhatIfItem(BaseModel):
    """A single medication-what-if answer."""
    medication_name: str
    scenario: WhatIfScenario
    deterministic_answer: str
    safety_severity: Literal["low", "moderate", "high"] = "moderate"
    rationale: str | None = None
    is_llm_polished: bool = False


class MedicationWhatIfResponse(BaseModel):
    """Output of compute_medication_what_if -- covering one or many
    (medication, scenario) pairs."""
    patient_reference: str | None = None
    items: list[MedicationWhatIfItem]
    n_items: int = Field(ge=0, default=0)
    contains_llm_polish: bool = False
    llm_model_id: str | None = None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 7.1 -- Auto-coding agent (ICD-10 / CPT / HCPCS + audit)
# ─────────────────────────────────────────────────────────────────────


class CodingSuggestion(BaseModel):
    """A single suggested code with cite-back to the source."""
    code: str
    code_system: Literal[
        "icd10cm", "cpt", "hcpcs", "loinc", "snomed", "rxnorm",
    ]
    display: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    supporting_text_excerpts: list[str] = Field(default_factory=list)
    rationale: str = ""


class ICD10SuggestReport(BaseModel):
    """Output of compute_icd10_suggest."""
    suggestions: list[CodingSuggestion]
    n_suggestions: int = Field(ge=0, default=0)
    primary_code: str | None = Field(
        default=None,
        description=(
            "Top-ranked ICD-10 suggestion when its confidence ≥ 0.7 "
            "AND its margin over the second pick ≥ 0.15. Otherwise None."
        ),
    )
    extraction_method: Literal["rule_based", "llm_disambiguated"] = "rule_based"
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class CPTSuggestReport(BaseModel):
    """Output of compute_cpt_suggest."""
    suggestions: list[CodingSuggestion]
    n_suggestions: int = Field(ge=0, default=0)
    primary_code: str | None = None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class HCPCSSuggestReport(BaseModel):
    """Output of compute_hcpcs_suggest."""
    suggestions: list[CodingSuggestion]
    n_suggestions: int = Field(ge=0, default=0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class CodingAuditFinding(BaseModel):
    """One row in the coding-audit report."""
    finding_kind: Literal[
        "documented_not_coded",   # chart shows the dx but the bill omits it
        "coded_not_documented",   # the bill has the code but chart doesn't support it
        "specificity_loss",       # coded as unspecified when specifier exists in chart
        "duplicate_code",         # same code billed twice
    ]
    code: str
    code_system: Literal["icd10cm", "cpt", "hcpcs"]
    description: str
    chart_evidence_ids: list[str] = Field(default_factory=list)
    severity: Literal["high", "medium", "low"] = "medium"
    suggested_action: str | None = None


class CodingAuditReport(BaseModel):
    """Output of compute_coding_audit."""
    findings: list[CodingAuditFinding]
    n_findings: int = Field(ge=0, default=0)
    n_high_severity: int = Field(ge=0, default=0)
    expected_revenue_impact_usd: float = Field(
        default=0.0,
        description=(
            "Coarse estimate of the revenue at risk from the findings. "
            "Each `documented_not_coded` adds the literature-derived "
            "median DRG delta; `coded_not_documented` subtracts the "
            "denial-clawback estimate. US-only; re-fit on local."
        ),
    )
    rationale: str
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 7.2 -- Pharmacogenomic decision support (PGx)
# ─────────────────────────────────────────────────────────────────────


PGxGene = Literal[
    "CYP2D6", "CYP2C9", "CYP2C19", "CYP3A4", "CYP3A5",
    "CYP4F2",
    "VKORC1", "SLCO1B1", "TPMT", "DPYD", "UGT1A1",
    "HLA-B*5701", "HLA-B*1502", "HLA-A*3101", "HLA-B*5801",
    "G6PD", "NUDT15",
    # Phase 12.4 D1 expansion
    "MT-RNR1",     # mitochondrial -- aminoglycoside ototoxicity
    "CFTR",        # CF -- ivacaftor / triple therapy
    "POLG",        # mitochondrial -- valproate hepatotoxicity
    "OPRM1",       # opioid receptor -- morphine response
    "IFNL3",       # IL28B -- peginterferon
    "ABCG2",       # rosuvastatin transporter
    "APOE",        # statin response variant
]

PGxPhenotype = Literal[
    "ultra_rapid_metabolizer",
    "rapid_metabolizer",
    "normal_metabolizer",
    "intermediate_metabolizer",
    "poor_metabolizer",
    "indeterminate",
    "positive",         # for HLA alleles
    "negative",
    "deficient",        # G6PD / TPMT / DPYD
    "non_deficient",
    # Phase 12.4 D1 expansion
    "m.1555A>G",        # MT-RNR1 mitochondrial variant
    "G551D_homozygous", # CFTR gating mutation
    "F508del_homozygous",
    "G_carrier",        # OPRM1 118A>G
    "TT",               # IL28B / IFNL3
    "AA_homozygous",    # VKORC1 -1639 AA
]


class PGxGenotype(BaseModel):
    """A single gene-phenotype pair from a FHIR Genomics resource."""
    gene: PGxGene
    phenotype: PGxPhenotype
    diplotype: str | None = Field(
        default=None,
        description="Star-allele diplotype (e.g. *1/*4 for CYP2D6).",
    )
    source_id: str | None = None


class PGxDoseAdjustment(BaseModel):
    """One dose-adjustment recommendation from compute_pgx_dose_adjustment."""
    drug: str
    rxnorm_code: str | None = None
    cpic_guideline_id: str | None = None
    recommendation: Literal[
        "use_standard_dose",
        "increase_dose",
        "decrease_dose",
        "alternative_drug_strongly_recommended",
        "avoid_drug",
        "additional_monitoring_required",
    ]
    dose_modifier_pct: int = Field(
        ge=-100, le=200,
        description=(
            "Percentage change vs the standard dose. Positive = "
            "increase; negative = decrease. 0 = no change."
        ),
        default=0,
    )
    rationale: str
    cpic_evidence_level: Literal["A", "B", "C", "D", "informational"] = "B"
    monitoring_recommendations: list[str] = Field(default_factory=list)


class PGxDoseAdjustmentReport(BaseModel):
    """Output of compute_pgx_dose_adjustment."""
    patient_reference: str | None = None
    genotypes: list[PGxGenotype]
    adjustments: list[PGxDoseAdjustment]
    n_adjustments: int = Field(ge=0, default=0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class PGxAlternative(BaseModel):
    """Alternative drug suggestion when a phenotype blocks the
    requested medication."""
    drug: str
    rxnorm_code: str | None = None
    rationale: str
    same_class: bool = False
    cpic_evidence_level: Literal["A", "B", "C", "D", "informational"] = "B"


class PGxAlternativesReport(BaseModel):
    """Output of compute_pgx_drug_alternatives."""
    requested_drug: str
    blocking_genotypes: list[PGxGenotype]
    alternatives: list[PGxAlternative]
    n_alternatives: int = Field(ge=0, default=0)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class PGxEligibilityReport(BaseModel):
    """Output of compute_pgx_eligibility_check."""
    requested_test: str
    medications_in_consideration: list[str]
    is_eligible: bool
    cpic_supported_drugs: list[str] = Field(default_factory=list)
    rationale: str
    expected_clinical_actionability: Literal[
        "high", "moderate", "low", "experimental",
    ] = "moderate"
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 7.4 -- Explainability depth (SHAP + ALE + concept-bottleneck +
# constitutional critic)
# ─────────────────────────────────────────────────────────────────────


class SHAPAttribution(BaseModel):
    """Per-feature attribution for a single prediction."""
    feature: str
    feature_value: float | int | str
    attribution: float
    baseline_value: float | int | str | None = None


class SHAPAttributionReport(BaseModel):
    """Output of compute_lace_shap_attribution."""
    predicted_probability: float
    cohort_baseline: float
    attributions: list[SHAPAttribution]
    sum_of_attributions: float
    rationale: str
    references: list[str] = Field(default_factory=list)


class ALEPoint(BaseModel):
    feature_value: float
    ale_value: float
    n_samples: int = Field(ge=0, default=0)


class ALEPlotData(BaseModel):
    feature: str
    points: list[ALEPoint]
    cohort_baseline: float
    rationale: str
    references: list[str] = Field(default_factory=list)


class BottleneckConcept(BaseModel):
    concept_id: str
    label: str
    importance: float = Field(ge=0.0, le=1.0)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    rationale: str


class ConceptBottleneckReport(BaseModel):
    concepts: list[BottleneckConcept]
    n_concepts: int = Field(ge=0, default=0)
    overall_summary: str
    references: list[str] = Field(default_factory=list)


class ConstitutionalPrinciple(BaseModel):
    id: str
    text: str
    severity: Literal["informational", "low", "medium", "high"] = "medium"


class ConstitutionalCheckOutcome(BaseModel):
    principle_id: str
    verdict: Literal["pass", "concern", "violation"]
    explanation: str
    cited_evidence_ids: list[str] = Field(default_factory=list)


class ConstitutionalCheckReport(BaseModel):
    outcomes: list[ConstitutionalCheckOutcome]
    n_violations: int = Field(ge=0, default=0)
    n_concerns: int = Field(ge=0, default=0)
    overall_verdict: Literal["pass", "concern", "block"]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 9.5 -- N-of-1 trial designer
# ─────────────────────────────────────────────────────────────────────


NofOneDesignType = Literal[
    "AB",
    "ABA",
    "ABAB",
    "ABABAB",
    "randomized_block",
]


class NofOneAnalysisStep(BaseModel):
    step_id: str
    description: str
    statistical_method: str


class NofOneTrialDesignReport(BaseModel):
    patient_reference: str | None = None
    intervention_label: str
    outcome_label: str
    design_type: NofOneDesignType
    block_duration_days: int = Field(ge=1)
    n_blocks: int = Field(ge=2)
    total_duration_days: int = Field(ge=1)
    randomization_sequence: list[str]
    expected_carryover_periods: int = Field(ge=0)
    minimum_detectable_effect: float
    analysis_plan: list[NofOneAnalysisStep]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 7.5 -- Pre-arrival triage agent (patient-side)
# ─────────────────────────────────────────────────────────────────────


CareLevel = Literal[
    "self_care",
    "telehealth",
    "primary_care_within_24h",
    "primary_care_within_7d",
    "urgent_care",
    "ed_now",
    "call_911",
]


class SymptomRedFlag(BaseModel):
    flag_id: str
    label: str
    severity: Literal["low", "medium", "high", "critical"]
    matched_text: str
    rationale: str


class SymptomRedFlagReport(BaseModel):
    raw_input: str
    flags: list[SymptomRedFlag]
    n_flags: int = Field(ge=0, default=0)
    highest_severity: Literal[
        "none", "low", "medium", "high", "critical",
    ] = "none"
    recommendation: Literal[
        "no_red_flag", "monitor", "seek_care_today",
        "seek_urgent_care", "go_to_ed", "call_911",
    ]
    references: list[str] = Field(default_factory=list)


class WhenToSeekCareReport(BaseModel):
    recommended_level: CareLevel
    rationale: str
    timeline: str
    things_to_avoid: list[str] = Field(default_factory=list)
    things_to_watch_for: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


class FollowupQuestion(BaseModel):
    question_id: str
    question: str
    expected_answer_kind: Literal[
        "yes_no", "duration", "severity_1_to_10", "free_text",
    ]
    rationale: str


class FollowupQuestionsReport(BaseModel):
    raw_input: str
    questions: list[FollowupQuestion]
    n_questions: int = Field(ge=0, default=0)
    references: list[str] = Field(default_factory=list)


class CaregiverHandoff(BaseModel):
    """Output of compute_caregiver_handoff -- a structured caregiver-
    language hand-off package built from a DecisionCard +
    optional FHIR Bundle.

    Five sections: summary, key warnings, daily tasks, when to call,
    contact info. The schema is rigid so caregivers (often non-
    clinical) get a predictable artifact every time.
    """
    patient_reference: str
    summary: str
    key_warnings: list[str] = Field(default_factory=list)
    daily_tasks: list[str] = Field(default_factory=list)
    when_to_call: list[str] = Field(default_factory=list)
    contact_info: dict[str, str] = Field(default_factory=dict)
    reading_level_grade: float = Field(
        ge=0.0,
        description=(
            "Approximate Flesch-Kincaid reading-grade level of the "
            "summary. Target ≤ 8th grade for caregiver hand-offs."
        ),
        default=8.0,
    )
    contains_llm_polish: bool = False
    llm_model_id: str | None = None
    abstain_recommended: bool = False
    abstain_reason: str | None = None
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 10.1 -- HEDIS / CMS Stars Rating quality measures
# ─────────────────────────────────────────────────────────────────────


HEDISMeasureId = Literal[
    # Star measures (subset -- full HEDIS has 90+; we cover the "Top 20"
    # by Stars weight + closeable-via-EHR)
    "BCS",     # Breast Cancer Screening
    "CCS",     # Cervical Cancer Screening
    "COL",     # Colorectal Cancer Screening
    "CDC-EYE", # Eye Exam in Diabetes
    "CDC-HBA1C", # HbA1c Poor Control (>9%)
    "CBP",     # Controlling High Blood Pressure
    "MPM-ACE", # Annual Monitoring for Persistent ACE/ARB
    "MPM-ANTICONV", # Anticonvulsant monitoring
    "MPM-DIURETIC", # Diuretic monitoring
    "OMW",     # Osteoporosis Mgmt in Women w/ Fracture
    "PCR",     # Plan All-Cause Readmissions
    "FUH",     # Follow-Up After Mental Health Hosp
    "FUM",     # Follow-Up After ED for Mental Health
    "AAB",     # Avoidance of Antibiotic Treatment for Adults w/ Acute Bronchitis
    "FMC",     # Follow-Up After ED for People w/ Multiple Chronic Conditions
    "MRP",     # Medication Reconciliation Post-Discharge
    "CWP",     # Appropriate Testing for Pharyngitis
    "DAE",     # Use of High-Risk Medications in the Elderly
    "TRC",     # Transitions of Care (composite)
    "SUPD",    # Statin Use in Persons with Diabetes
]


StarsDomainId = Literal[
    "preventive_care", "chronic_conditions", "patient_experience",
    "complaints_and_problems", "improvement",
    # Aggregated
    "overall",
]


class QualityMeasureRate(BaseModel):
    """Per-measure numerator/denominator + rate."""
    measure_id: HEDISMeasureId
    measure_name: str
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    rate: float = Field(ge=0.0, le=1.0)
    benchmark_5_star: float = Field(ge=0.0, le=1.0)
    benchmark_4_star: float = Field(ge=0.0, le=1.0)
    benchmark_3_star: float = Field(ge=0.0, le=1.0)
    current_stars: int = Field(ge=1, le=5)
    eligible_unmet_count: int = Field(ge=0)


class QualityMeasuresAggregateReport(BaseModel):
    """Output of compute_quality_measures_aggregate."""
    measurement_year: int
    n_eligible_patients: int = Field(ge=0)
    measures: list[QualityMeasureRate]
    n_measures: int = Field(ge=0)
    overall_rate_5_star_count: int = Field(ge=0)
    overall_rate_4_star_count: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(default_factory=list)


class StarsRatingForecast(BaseModel):
    """Per-domain forecast at measurement period end."""
    domain: StarsDomainId
    current_score: float = Field(ge=0.0, le=5.0)
    projected_score_eom: float = Field(ge=0.0, le=5.0)
    projection_basis: str
    actions_required_for_4_star: int = Field(ge=0)
    actions_required_for_5_star: int = Field(ge=0)


class StarsRatingForecastReport(BaseModel):
    """Output of compute_stars_rating_forecast."""
    contract_id: str | None = None
    measurement_year: int
    domains: list[StarsRatingForecast]
    overall_current: float = Field(ge=0.0, le=5.0)
    overall_projected_eom: float = Field(ge=0.0, le=5.0)
    estimated_qbp_dollars_at_overall: float = Field(
        default=0.0,
        description=(
            "Quality Bonus Payment dollars at the projected overall "
            "rating. CMS Stars Rating ≥ 4.0 unlocks ~ 5%-of-benchmark "
            "QBP; 4.5-5.0 unlocks +5% additional."
        ),
    )
    rationale: str
    references: list[str] = Field(default_factory=list)


class CareGapPriorityAction(BaseModel):
    """One ranked care-gap-closure action."""
    measure_id: HEDISMeasureId
    measure_name: str
    n_eligible_patients_with_gap: int = Field(ge=0)
    expected_stars_lift: float = Field(
        ge=0.0,
        description=(
            "Expected per-domain Stars-points contribution from closing "
            "the eligible-unmet population for this measure. The unit is "
            "additive per-domain points (NOT a 0-5 score), so it can "
            "legitimately exceed 5 when many gaps × heavy weight × low "
            "difficulty multiply."
        ),
    )
    expected_qbp_lift_dollars: float = Field(default=0.0)
    closure_difficulty: Literal[
        "easy", "moderate", "hard", "very_hard",
    ]
    suggested_intervention: str


class CareGapPriorityRanking(BaseModel):
    """Output of compute_care_gap_priority_ranking."""
    actions: list[CareGapPriorityAction]
    n_actions: int = Field(ge=0)
    cumulative_expected_stars_lift: float = Field(ge=0.0)
    cumulative_expected_qbp_dollars: float = Field(default=0.0)
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 10.2 -- Population health + outbreak detection
# ─────────────────────────────────────────────────────────────────────


class SyndromicCluster(BaseModel):
    """One detected cluster of presenting complaints."""
    cluster_id: str
    syndrome: str
    n_cases_observed: int = Field(ge=0)
    n_cases_expected: float = Field(ge=0.0)
    z_score: float
    geographic_window: str | None = None
    time_window_start_iso: str
    time_window_end_iso: str
    severity: Literal["informational", "watch", "alert", "outbreak"]


class SyndromicSurveillanceReport(BaseModel):
    """Output of compute_syndromic_surveillance."""
    surveillance_period_start_iso: str
    surveillance_period_end_iso: str
    n_total_chief_complaints: int = Field(ge=0)
    clusters: list[SyndromicCluster]
    n_clusters: int = Field(ge=0)
    n_outbreak_severity: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(default_factory=list)


class VaccineReminderCohort(BaseModel):
    """One cohort of patients overdue for a vaccine."""
    vaccine_id: str
    vaccine_name: str
    n_patients_overdue: int = Field(ge=0)
    age_distribution: dict[str, int] = Field(default_factory=dict)
    risk_distribution: dict[str, int] = Field(default_factory=dict)
    expected_outreach_uptake: float = Field(
        ge=0.0, le=1.0, default=0.30,
        description=(
            "Literature-derived expected acceptance rate for outreach-"
            "driven vaccine reminders (CDC: phone ~ 30%, postcard "
            "~ 18%, EHR-portal ~ 25%)."
        ),
    )


class VaccineReminderCohortReport(BaseModel):
    """Output of compute_vaccine_reminder_cohort."""
    cohorts: list[VaccineReminderCohort]
    n_cohorts: int = Field(ge=0)
    total_patients: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(default_factory=list)


class OutbreakHeatmapCell(BaseModel):
    """One cell in the DP-noised outbreak heatmap."""
    geographic_bucket: str
    syndrome: str
    n_cases_dp_noised: int = Field(ge=0)
    rate_dp_noised: float = Field(ge=0.0)


class OutbreakHeatmapReport(BaseModel):
    """Output of compute_outbreak_heatmap."""
    cells: list[OutbreakHeatmapCell]
    n_cells: int = Field(ge=0)
    epsilon_used: float = Field(gt=0.0)
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 10.3 -- Insurance appeals
# ─────────────────────────────────────────────────────────────────────


class DenialReason(BaseModel):
    """A parsed denial reason from a payer letter."""
    reason_code: str
    reason_text: str
    category: Literal[
        "medical_necessity", "step_therapy_not_met",
        "out_of_network", "experimental_unproven",
        "missing_documentation", "duplicate_service",
        "non_covered_benefit", "claim_filing_limit",
        "other",
    ]
    cited_policy: str | None = None
    referenced_evidence: list[str] = Field(default_factory=list)


class DenialLetterParseReport(BaseModel):
    """Output of compute_denial_letter_parse."""
    payer: str
    claim_id: str | None = None
    received_date_iso: str | None = None
    appeal_deadline_iso: str | None = None
    reasons: list[DenialReason]
    n_reasons: int = Field(ge=0)
    is_partial_denial: bool = False
    contested_dollar_amount: float | None = None
    rationale: str
    references: list[str] = Field(default_factory=list)


class AppealLetterParagraph(BaseModel):
    section: Literal[
        "header", "patient_summary", "denial_summary",
        "medical_necessity_argument", "evidence_supporting",
        "policy_counter_argument", "alternative_proposed",
        "closing",
    ]
    text: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    is_llm_polished: bool = False


class AppealLetterDraft(BaseModel):
    """Output of compute_appeal_letter_draft."""
    payer: str
    appeal_level: Literal[
        "internal_first_level", "internal_second_level",
        "external_independent_review", "state_insurance_commissioner",
        "ada_complaint",
    ]
    paragraphs: list[AppealLetterParagraph]
    full_text: str
    n_cite_backs: int = Field(ge=0)
    contains_llm_polish: bool = False
    llm_model_id: str | None = None
    rationale: str
    references: list[str] = Field(default_factory=list)


class AppealEscalationStep(BaseModel):
    level: Literal[
        "internal_first_level", "internal_second_level",
        "external_independent_review", "state_insurance_commissioner",
        "ada_complaint",
    ]
    description: str
    deadline_after_denial_days: int = Field(ge=0)
    required_documents: list[str]
    expected_decision_window_days: int = Field(ge=0)
    estimated_success_probability: float = Field(ge=0.0, le=1.0)


class AppealEscalationPathReport(BaseModel):
    """Output of compute_appeal_escalation_path."""
    payer: str
    starting_level: Literal[
        "internal_first_level", "internal_second_level",
        "external_independent_review",
    ]
    steps: list[AppealEscalationStep]
    n_steps: int = Field(ge=0)
    cumulative_max_days: int = Field(ge=0)
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 10.4 -- Adversarial evaluation suite v2
# ─────────────────────────────────────────────────────────────────────


RedTeamCategory = Literal[
    "prompt_injection",
    "phi_exfiltration",
    "jailbreak",
    "hallucination_trigger",
    "citation_fabrication",
    "bias_probe",
    "ood_input",
    "multilingual_evasion",
    "encoding_obfuscation",
    "tool_misuse",
]


class RedTeamPrompt(BaseModel):
    """One adversarial prompt in the v2 corpus."""
    prompt_id: str
    category: RedTeamCategory
    severity: Literal["informational", "low", "moderate", "high", "critical"]
    payload: str
    expected_outcome: Literal[
        "abstain", "refuse", "execute_safely", "redact",
    ] = Field(
        description=(
            "What the target should do: abstain (don't act), refuse "
            "(decline outright), execute_safely (process payload as "
            "data, not as instructions), or redact (PHI scrubbing must "
            "redact sensitive entities)."
        ),
    )
    forbidden_substrings: list[str] = Field(
        default_factory=list,
        description=(
            "Substrings whose presence in the output indicates the "
            "attack succeeded (e.g. system-prompt leak markers, "
            "fabricated citations, leaked MRNs)."
        ),
    )
    metadata: dict[str, str] = Field(default_factory=dict)


class RedTeamCaseResult(BaseModel):
    """Per-case outcome after running one RedTeamPrompt against a target."""
    prompt_id: str
    category: RedTeamCategory
    severity: Literal["informational", "low", "moderate", "high", "critical"]
    expected_outcome: Literal[
        "abstain", "refuse", "execute_safely", "redact",
    ]
    observed_outcome: Literal[
        "abstain", "refuse", "execute_safely", "redact",
        "leak", "crash", "unknown",
    ]
    passed: bool
    forbidden_hits: list[str] = Field(default_factory=list)
    rationale: str = ""


class RedTeamReport(BaseModel):
    """Output of run_redteam_corpus -- aggregate of all RedTeamCaseResults."""
    corpus_id: str
    target_label: str
    n_cases: int = Field(ge=0)
    n_passed: int = Field(ge=0)
    n_failed: int = Field(ge=0)
    overall_pass_rate: float = Field(ge=0.0, le=1.0)
    pass_rate_by_category: dict[str, float] = Field(default_factory=dict)
    pass_rate_by_severity: dict[str, float] = Field(default_factory=dict)
    cases: list[RedTeamCaseResult]
    posture: Literal["pass", "warn", "fail"]
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 10.8 -- Tool-use planner brain
# ─────────────────────────────────────────────────────────────────────


class ToolUsePlanStep(BaseModel):
    """One step in a tool-use plan -- a single tool invocation against
    a specific specialist + bundle, with the arguments already shaped."""
    step_id: str
    specialist: str = Field(
        description=(
            "Federation specialist that exposes the chosen tool -- "
            "e.g. 'trustedrisk-quality' or 'trustedrisk-discharge'."
        ),
    )
    bundle: str = Field(
        description=(
            "Thematic bundle (key from mcp_server.tools.BUNDLES)."
        ),
    )
    tool: str = Field(
        description=(
            "Compute-tool function name (e.g. compute_readmission_risk)."
        ),
    )
    args: dict[str, Any] = Field(default_factory=dict)
    mandatory: bool = Field(
        default=False,
        description=(
            "Floor-marked steps that the LLM router cannot drop or "
            "reorder. Used for safety steps such as detect_phi."
        ),
    )
    rationale: str = ""


class ToolUsePlan(BaseModel):
    """Output of plan_tool_use -- an ordered tool-invocation plan."""
    user_query: str
    steps: list[ToolUsePlanStep]
    n_steps: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    safety_floor_engaged: bool = Field(
        description=(
            "True when at least one step was injected by the "
            "deterministic floor (e.g. detect_phi for free-text PHI "
            "scrubbing or fairness_audit on demographic-bearing inputs)."
        ),
    )
    llm_router_used: bool = False
    llm_model_id: str | None = None
    rationale: str
    references: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Phase 10.9 -- Conformal prediction (split conformal, marginal coverage)
# ─────────────────────────────────────────────────────────────────────


class ConformalCalibrationReport(BaseModel):
    """Output of fit_split_conformal -- a calibrated quantile + meta."""
    n_calibration: int = Field(ge=1)
    target_coverage: float = Field(gt=0.0, lt=1.0)
    empirical_coverage: float = Field(ge=0.0, le=1.0)
    quantile_threshold: float = Field(
        description=(
            "The (n+1)(1-α)/n empirical quantile of the "
            "non-conformity scores on the held-out calibration set."
        ),
    )
    score_function_id: Literal["abs_residual", "binary_one_minus_p"] = Field(
        description=(
            "abs_residual = |y - p|; binary_one_minus_p = 1 - p_y "
            "(softmax-like for class-balanced outcomes)."
        ),
    )
    rationale: str
    references: list[str] = Field(default_factory=list)


class ConformalPredictionInterval(BaseModel):
    """One conformal prediction interval for a regression-style score."""
    point_estimate: float
    lower_bound: float
    upper_bound: float
    target_coverage: float = Field(gt=0.0, lt=1.0)
    quantile_threshold_used: float


class ConformalPredictionSet(BaseModel):
    """One conformal prediction set for a binary outcome."""
    point_estimate_probability: float = Field(ge=0.0, le=1.0)
    prediction_set: list[Literal[0, 1]] = Field(
        description=(
            "Subset of {0, 1} included at the configured coverage. "
            "An empty set indicates the point estimate is outside the "
            "calibrated band -- caller should abstain."
        ),
    )
    target_coverage: float = Field(gt=0.0, lt=1.0)
    quantile_threshold_used: float


# ─────────────────────────────────────────────────────────────────────
# Phase 11.4 -- Multi-modal: ECG QT + DICOM SR ingest
# ─────────────────────────────────────────────────────────────────────


class ECGQTReport(BaseModel):
    """Output of compute_ecg_qt_analyzer.

    All numeric fields are optional because the tool returns an abstain
    record when called without a waveform (or with too few R peaks for
    R-R estimation). Callers must check `abstain_recommended` before
    consuming the QT/QTc values.
    """
    rr_interval_ms: float | None = Field(
        default=None, gt=0.0,
        description="R-R interval in milliseconds.",
    )
    heart_rate_bpm: float | None = Field(default=None, gt=0.0, le=300.0)
    qt_interval_ms: float | None = Field(default=None, gt=0.0)
    qtc_bazett_ms: float | None = Field(default=None, gt=0.0)
    qtc_fridericia_ms: float | None = Field(default=None, gt=0.0)
    qtc_classification: Literal[
        "normal", "borderline", "prolonged", "severe_prolonged",
    ] | None = None
    sex: Literal["male", "female", "unspecified"] = "unspecified"
    prolongation_risk_factors: list[str] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class DICOMSRFinding(BaseModel):
    """A single SR Coded-Concept finding."""
    code_system: str
    code: str
    display: str
    severity: Literal[
        "informational", "low", "moderate", "high", "critical",
    ] = "informational"


class DICOMSRIngestReport(BaseModel):
    """Output of compute_dicom_sr_ingest -- surfaces a structured-report
    DICOM/FHIR document into something the rest of the system can
    consume."""
    modality: Literal[
        "CT", "MR", "US", "XR", "CR", "ECG", "PET", "NM", "OT",
    ]
    body_part: str | None = None
    n_findings: int = Field(ge=0)
    findings: list[DICOMSRFinding]
    impressions: list[str] = Field(default_factory=list)
    conclusion: str | None = None
    cited_resource_ids: list[str] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(default_factory=list)
