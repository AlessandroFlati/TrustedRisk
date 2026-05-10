"""Phase 10.8 -- Tool-use planner brain.

Given a free-text user query, return a structured `ToolUsePlan` that
selects (specialist, bundle, tool, args) from the 89-tool / 30-bundle
TrustedRisk surface.

Two layers -- deterministic floor + optional LLM router:

  1. **Deterministic floor** -- keyword + regex classifier with hand-
     curated intent -> tool routing rules. Always runs and may inject
     mandatory safety steps (PHI scrub on free-text inputs, fairness
     audit when demographics are mentioned). The floor is the *contract*
     -- it bounds what the LLM router can change.

  2. **Optional LLM router** -- pluggable callable
        (user_query: str, floor_plan: ToolUsePlan, catalog: dict[str, list[str]])
            -> list[ToolUsePlanStep]
     that may add steps, reorder non-mandatory steps, or shape args. The
     LLM CANNOT drop steps marked `mandatory=True`.

Pure-deterministic when `llm_router=None`. CI exercises only the floor;
production deployments wire in a real model.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

from shared.schemas import ToolUsePlan, ToolUsePlanStep


# ─────────────────────────────────────────────────────────────────────
# Intent -> routing rule registry
# ─────────────────────────────────────────────────────────────────────


_DEMOGRAPHIC_TRIGGERS = re.compile(
    r"\b(black|white|hispanic|asian|latino|female|male|elderly|"
    r"pediatric|child|race|ethnicity|gender|insurance|medicaid|"
    r"medicare)\b",
    re.IGNORECASE,
)

_FREE_TEXT_PHI_TRIGGERS = re.compile(
    r"\b(mrn|ssn|dob|phone|email|address|zip|patient\s+name|"
    r"social[\s-]?security|medical\s+record\s+number)\b",
    re.IGNORECASE,
)


# Each rule: (regex, specialist, bundle, tool, default_args, mandatory)
_INTENT_RULES: list[tuple[re.Pattern, str, str, str, dict[str, Any], bool]] = [
    # Discharge planning
    (re.compile(r"\b(discharge|readmission|safe to go home)\b", re.IGNORECASE),
     "trustedrisk-discharge", "core_discharge",
     "compute_readmission_risk", {}, False),

    # Decision utility
    (re.compile(r"\b(disposition|recommended action|admit or discharge)\b",
                  re.IGNORECASE),
     "trustedrisk-discharge", "core_discharge",
     "compute_decision_utility", {}, False),

    # Med reconciliation
    (re.compile(r"\b(med(ication)? reconciliation|med list compare|"
                  r"home meds vs)\b", re.IGNORECASE),
     "trustedrisk-discharge", "core_discharge",
     "compute_medication_reconciliation", {}, False),

    # Polypharmacy / DDI
    (re.compile(r"\b(drug[\s-]drug interaction|ddi|polypharmacy)\b",
                  re.IGNORECASE),
     "trustedrisk-discharge", "core_discharge",
     "detect_polypharmacy_concerns", {}, False),

    # ED triage
    (re.compile(r"\b(ed triage|esi level|emergency severity|"
                  r"triage this patient)\b", re.IGNORECASE),
     "trustedrisk-acute", "ed_acute", "compute_admission_triage", {}, False),

    # NEWS2 / deterioration
    (re.compile(r"\b(news2|deterioration|rapid response|early warning)\b",
                  re.IGNORECASE),
     "trustedrisk-acute", "ed_acute",
     "compute_clinical_deterioration_score", {}, False),

    # Stroke
    (re.compile(r"\b(stroke|nihss|tpa|alteplase|thrombolysis)\b",
                  re.IGNORECASE),
     "trustedrisk-acute", "stroke_acs",
     "compute_stroke_thrombolysis_eligibility", {}, False),

    # Chest pain HEART
    (re.compile(r"\b(chest pain|heart score|acs|stemi)\b", re.IGNORECASE),
     "trustedrisk-acute", "stroke_acs", "compute_heart_score", {}, False),

    # Pediatric early warning
    (re.compile(r"\b(pews|pediatric early warning|infant|pediatric vitals)\b",
                  re.IGNORECASE),
     "trustedrisk-pediatric", "pediatric",
     "compute_pediatric_early_warning", {}, False),

    # Pediatric dosing
    (re.compile(r"\b(weight[\s-]?based dos|pediatric dos|mg/kg)\b",
                  re.IGNORECASE),
     "trustedrisk-pediatric", "pediatric",
     "compute_weight_based_dosing", {}, False),

    # Mental health crisis
    (re.compile(r"\b(suicide risk|c-?ssrs|self[\s-]harm)\b",
                  re.IGNORECASE),
     "trustedrisk-mental-health", "mental_health",
     "compute_suicide_risk_assessment", {}, False),

    # Antimicrobial
    (re.compile(r"\b(empiric antibiotics?|antibiogram|sepsis empiric|"
                  r"de-?escalation)\b", re.IGNORECASE),
     "trustedrisk-discharge", "antimicrobial",
     "compute_empiric_antibiotic_selection", {}, False),

    # Oncology RECIST
    (re.compile(r"\b(recist|chemo dose|oncology cycle|tumor response)\b",
                  re.IGNORECASE),
     "trustedrisk-discharge", "oncology",
     "compute_oncology_treatment_response", {}, False),

    # Maternal MEOWS
    (re.compile(r"\b(meows|maternal early warning|preeclampsia|hellp)\b",
                  re.IGNORECASE),
     "trustedrisk-acute", "obstetric_geriatric",
     "compute_maternal_early_warning", {}, False),

    # Geriatric falls / delirium
    (re.compile(r"\b(morse falls|cam delirium|fall risk)\b", re.IGNORECASE),
     "trustedrisk-acute", "obstetric_geriatric",
     "compute_falls_risk_morse", {}, False),

    # Trauma / MTP
    (re.compile(r"\b(iss|trauma severity|mtp|massive transfusion)\b",
                  re.IGNORECASE),
     "trustedrisk-acute", "trauma_critical",
     "compute_trauma_severity_score", {}, False),

    # Endocrine / DKA / glucose
    (re.compile(r"\b(dka|diabetic ketoacidosis|inpatient glycemic)\b",
                  re.IGNORECASE),
     "trustedrisk-acute", "endocrine_acute",
     "compute_dka_severity", {}, False),

    # Imaging / contrast
    (re.compile(r"\b(ct scan|mri|imaging appropriateness|contrast safety|"
                  r"acr criteria)\b", re.IGNORECASE),
     "trustedrisk-acute", "imaging",
     "compute_imaging_appropriateness", {}, False),

    # AKI / dialysis
    (re.compile(r"\b(aki|kdigo|acute kidney injury|dialysis initiation)\b",
                  re.IGNORECASE),
     "trustedrisk-acute", "nephrology",
     "compute_aki_kdigo_stage", {}, False),

    # Cost-effectiveness
    (re.compile(r"\b(icer|cost[\s-]effective|qaly|evoi|"
                  r"value of intervention)\b", re.IGNORECASE),
     "trustedrisk-population", "economics",
     "compute_expected_value_of_intervention", {}, False),

    # SHARP patient resolution
    (re.compile(r"\b(resolve patient|find patient|the gentleman|"
                  r"who is the patient)\b", re.IGNORECASE),
     "trustedrisk-evidence", "context_resolution",
     "compute_resolve_patient_from_query", {}, False),

    # Differential diagnosis
    (re.compile(r"\b(differential diagnosis|ddx|rule out)\b", re.IGNORECASE),
     "trustedrisk-evidence", "diagnosis",
     "compute_differential_diagnosis_ranker", {}, False),

    # Patient counseling
    (re.compile(r"\b(discharge counseling|patient[\s-]language|6th-grade)\b",
                  re.IGNORECASE),
     "trustedrisk-discharge", "patient_facing",
     "compute_discharge_counseling", {}, False),

    # Translate counseling -- anchored on the verb 'translate' + target lang
    (re.compile(r"\btranslate\b[^.]{0,80}?\b(spanish|chinese|vietnamese|"
                  r"arabic|french|portuguese|german|italian|english|"
                  r"español|中文)\b|\bmultilingual\b",
                  re.IGNORECASE),
     "trustedrisk-patient", "patient_facing",
     "compute_translate_discharge_counseling", {}, False),

    # FAQs / what-if
    (re.compile(r"\b(what if I (miss|take|stop)|what happens if I)\b",
                  re.IGNORECASE),
     "trustedrisk-patient", "patient_qa",
     "compute_medication_what_if", {}, False),

    # Caregiver hand-off
    (re.compile(r"\b(caregiver|hand-?off|family member summary)\b",
                  re.IGNORECASE),
     "trustedrisk-patient", "patient_qa",
     "compute_caregiver_handoff", {}, False),

    # Charlson / Elixhauser
    (re.compile(r"\b(charlson|elixhauser|comorbidity index)\b",
                  re.IGNORECASE),
     "trustedrisk-discharge", "data_normalization",
     "compute_charlson_elixhauser_index", {}, False),

    # LOINC / RxNorm normalization
    (re.compile(r"\b(loinc|rxnorm|umls)\b", re.IGNORECASE),
     "trustedrisk-discharge", "data_normalization",
     "compute_normalize_observations", {}, False),

    # Order set
    (re.compile(r"\b(discharge order set|order set|order bundle)\b",
                  re.IGNORECASE),
     "trustedrisk-discharge", "clinical_workflow",
     "compute_order_set", {}, False),

    # PubMed / external knowledge
    (re.compile(r"\b(pubmed|literature search|recent papers|cochrane)\b",
                  re.IGNORECASE),
     "trustedrisk-evidence", "external_knowledge",
     "compute_pubmed_search", {}, False),

    # Clinical NER
    (re.compile(r"\b(clinical ner|extract entities|named entity)\b",
                  re.IGNORECASE),
     "trustedrisk-evidence", "chart_intelligence",
     "compute_clinical_ner", {}, False),

    # Prior Authorization
    (re.compile(r"\b(prior auth|pa letter|pa appeal|payer rules)\b",
                  re.IGNORECASE),
     "trustedrisk-pa", "prior_authorization",
     "compute_pa_letter_draft", {}, False),

    # SCRIBE: progress note
    (re.compile(r"\b(progress note|soap note|daily note)\b", re.IGNORECASE),
     "trustedrisk-scribe", "clinical_documentation",
     "compute_progress_note_draft", {}, False),

    # SCRIBE: discharge summary
    (re.compile(r"\b(discharge summary|joint commission)\b", re.IGNORECASE),
     "trustedrisk-scribe", "clinical_documentation",
     "compute_discharge_summary_draft", {}, False),

    # SCRIBE: H&P
    (re.compile(r"\b(h&p|history and physical|admission note)\b",
                  re.IGNORECASE),
     "trustedrisk-scribe", "clinical_documentation",
     "compute_admission_hnp_draft", {}, False),

    # Auto-coding
    (re.compile(r"\b(icd-?10|cpt code|hcpcs|coding audit)\b", re.IGNORECASE),
     "trustedrisk-coder", "auto_coding",
     "compute_icd10_suggest", {}, False),

    # PGx
    (re.compile(r"\b(pharmacogenomic|cpic|pgx|cyp2c19|cyp2d6|warfarin dose)\b",
                  re.IGNORECASE),
     "trustedrisk-pgx", "pharmacogenomics",
     "compute_pgx_dose_adjustment", {}, False),

    # Pre-arrival triage
    (re.compile(r"\b(red flag|urgent care|when to seek care|"
                  r"when should I go to the er)\b", re.IGNORECASE),
     "trustedrisk-preadmit", "preadmit_triage",
     "compute_when_to_seek_care", {}, False),

    # N-of-1 trial
    (re.compile(r"\b(n[-\s]?of[-\s]?1|single[\s-]patient trial)\b",
                  re.IGNORECASE),
     "trustedrisk-evidence", "research_design",
     "compute_n_of_1_trial_design", {}, False),

    # HEDIS Stars (Phase 10.1)
    (re.compile(r"\b(hedis|stars rating|star rating|qbp|"
                  r"quality bonus payment)\b", re.IGNORECASE),
     "trustedrisk-quality", "quality_stars",
     "compute_quality_measures_aggregate", {}, False),

    # Care-gap ranking (Phase 10.1)
    (re.compile(r"\b(care[\s-]gap|gap[\s-]closure|prioritise.*gaps?)\b",
                  re.IGNORECASE),
     "trustedrisk-quality", "quality_stars",
     "compute_care_gap_priority_ranking", {}, False),

    # Pop health (Phase 10.2)
    (re.compile(r"\b(syndromic|outbreak|cluster detection|"
                  r"essence|nssp)\b", re.IGNORECASE),
     "trustedrisk-pophealth", "population_health",
     "compute_syndromic_surveillance", {}, False),
    (re.compile(r"\b(vaccine reminder|immunisation|immunization "
                  r"reminder|acip)\b", re.IGNORECASE),
     "trustedrisk-pophealth", "population_health",
     "compute_vaccine_reminder_cohort", {}, False),

    # Insurance appeals (Phase 10.3)
    (re.compile(r"\b(denial letter|claim denied|parse denial)\b",
                  re.IGNORECASE),
     "trustedrisk-appeals", "insurance_appeals",
     "compute_denial_letter_parse", {}, False),
    (re.compile(r"\b(appeal letter|appeal draft|insurance appeal)\b",
                  re.IGNORECASE),
     "trustedrisk-appeals", "insurance_appeals",
     "compute_appeal_letter_draft", {}, False),
    (re.compile(r"\b(escalation path|external review|state insurance "
                  r"commissioner|iro)\b", re.IGNORECASE),
     "trustedrisk-appeals", "insurance_appeals",
     "compute_appeal_escalation_path", {}, False),
]


# ─────────────────────────────────────────────────────────────────────
# Floor
# ─────────────────────────────────────────────────────────────────────


def _build_floor(query: str) -> tuple[list[ToolUsePlanStep], bool]:
    """Run the deterministic floor.

    Returns (steps, safety_floor_engaged).
    """
    steps: list[ToolUsePlanStep] = []
    safety_engaged = False

    # 1. PHI scrub when free-text PHI markers are present.
    if _FREE_TEXT_PHI_TRIGGERS.search(query):
        steps.append(ToolUsePlanStep(
            step_id=f"floor-phi-{len(steps)+1:03d}",
            specialist="trustedrisk-discharge",
            bundle="core_discharge",
            tool="detect_phi",
            args={"text": query},
            mandatory=True,
            rationale=(
                "PHI markers detected in free-text query; scrubbing is a "
                "mandatory deterministic-floor step before any downstream "
                "tool reads the input."
            ),
        ))
        safety_engaged = True

    # 2. Intent-routed clinical step(s).
    seen_tools: set[str] = set()
    for pat, specialist, bundle, tool, default_args, mandatory in _INTENT_RULES:
        if not pat.search(query) or tool in seen_tools:
            continue
        steps.append(ToolUsePlanStep(
            step_id=f"floor-int-{len(steps)+1:03d}",
            specialist=specialist, bundle=bundle, tool=tool,
            args=dict(default_args), mandatory=mandatory,
            rationale=(
                f"Keyword match on intent rule for tool {tool!r}."
            ),
        ))
        seen_tools.add(tool)

    # 3. Fairness audit floor when demographics surface.
    if _DEMOGRAPHIC_TRIGGERS.search(query):
        steps.append(ToolUsePlanStep(
            step_id=f"floor-fair-{len(steps)+1:03d}",
            specialist="trustedrisk-population",
            bundle="economics",
            tool="compute_fairness_audit",
            args={},
            mandatory=True,
            rationale=(
                "Demographic terms detected; fairness audit is a "
                "mandatory floor step for any recommendation that "
                "depends on demographic-bearing input."
            ),
        ))
        safety_engaged = True

    return steps, safety_engaged


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


LLMRouter = Callable[
    [str, list[ToolUsePlanStep], dict[str, list[str]]],
    Awaitable[list[ToolUsePlanStep]],
]


def _bundle_catalog() -> dict[str, list[str]]:
    """Return a copy of BUNDLES from the runtime registry. Lazy import
    keeps the module decoupled from the FastMCP graph."""
    from mcp_server.tools import BUNDLES   # type: ignore
    return {k: list(v) for k, v in BUNDLES.items()}


def _maybe_extend_with_tfidf_retrieval(
    user_query: str,
    floor_steps: list[ToolUsePlanStep],
    top_k: int,
) -> list[ToolUsePlanStep]:
    """Phase 14.14 Q3 -- surface TF-IDF candidates as advisory steps.

    Only fires when the deterministic floor produced no intent-clinical
    step (`floor-int-*`). Mandatory safety steps (PHI scrub, fairness)
    don't suppress the retrieval since they don't carry clinical intent.
    """
    has_intent = any(
        s.step_id.startswith("floor-int-") for s in floor_steps
    )
    if has_intent:
        return floor_steps
    from .tool_retrieval_tfidf import retrieve_tools_by_query
    res = retrieve_tools_by_query(user_query, top_k=top_k)
    if not res.hits:
        return floor_steps
    extended = list(floor_steps)
    seen_tools = {s.tool for s in extended}
    for hit in res.hits:
        if hit.tool_name in seen_tools:
            continue
        bundle = (
            hit.bundle_memberships[0] if hit.bundle_memberships
            else "core_discharge"
        )
        extended.append(ToolUsePlanStep(
            step_id=f"floor-tfidf-{len(extended)+1:03d}",
            specialist="trustedrisk-discharge",
            bundle=bundle,
            tool=hit.tool_name,
            args={},
            mandatory=False,
            rationale=(
                f"TF-IDF retrieval candidate (cosine={hit.score:.3f}); "
                f"surfaced as advisory step because no intent rule "
                f"matched the query."
            ),
        ))
        seen_tools.add(hit.tool_name)
    return extended


async def plan_tool_use(
    user_query: str,
    *,
    llm_router: LLMRouter | None = None,
    llm_model_id: str | None = None,
    enable_tfidf_retrieval: bool = False,
    tfidf_top_k: int = 3,
) -> ToolUsePlan:
    """Produce a ToolUsePlan for a free-text user request.

    Args:
        user_query: free-text query.
        llm_router: optional async callable that refines the floor plan.
            Cannot drop steps marked `mandatory=True`.
        llm_model_id: free-text model id propagated into the report.
        enable_tfidf_retrieval: when True and no intent-clinical step
            fires, surface up to ``tfidf_top_k`` advisory steps via the
            pure-Python TF-IDF index (Phase 14.14 Q3).
        tfidf_top_k: cap on advisory steps from TF-IDF retrieval.

    Returns:
        ToolUsePlan.
    """
    floor_steps, safety_engaged = _build_floor(user_query)
    catalog = _bundle_catalog()

    if enable_tfidf_retrieval:
        floor_steps = _maybe_extend_with_tfidf_retrieval(
            user_query, floor_steps, tfidf_top_k,
        )

    refined_steps = floor_steps
    llm_used = False
    if llm_router is not None:
        candidate = await llm_router(user_query, list(floor_steps), catalog)
        if not isinstance(candidate, list):
            raise ValueError(
                "llm_router must return a list[ToolUsePlanStep]"
            )
        # Re-inject any mandatory step the router dropped
        floor_mandatory = {s.tool for s in floor_steps if s.mandatory}
        candidate_tools = {s.tool for s in candidate}
        missing = floor_mandatory - candidate_tools
        if missing:
            for s in floor_steps:
                if s.tool in missing:
                    candidate.append(s)
        refined_steps = candidate
        llm_used = True

    confidence = round(min(1.0, 0.55 + 0.10 * len(floor_steps)), 3)

    rationale = (
        f"Floor produced {len(floor_steps)} step(s); "
        f"safety_floor_engaged = {safety_engaged}; "
        f"llm_router_used = {llm_used}. "
        f"Final plan has {len(refined_steps)} step(s)."
    )

    return ToolUsePlan(
        user_query=user_query,
        steps=refined_steps,
        n_steps=len(refined_steps),
        confidence=confidence,
        safety_floor_engaged=safety_engaged,
        llm_router_used=llm_used,
        llm_model_id=llm_model_id if llm_used else None,
        rationale=rationale,
        references=[
            "TrustedRisk Phase 10.8 -- tool-use planner brain (S1).",
            "OWASP LLM Top 10 -- LLM02 insecure output (2024).",
        ],
    )
