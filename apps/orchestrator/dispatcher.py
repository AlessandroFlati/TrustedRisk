"""TrustedRisk Orchestrator dispatcher.

Single entry-point that takes a clinical free-form prompt and decides
which underlying capability to invoke. Two layers, in order:

1. **Deterministic keyword router** (always available, sub-ms): scans
   the prompt for the longest matching keyword across the prebuilt
   workflow registry first, then across SPECIALIST_ROUTES. Demo macro
   pipelines are first-class entries, so prompts like "complete sepsis
   pipeline" route directly to the composer macro without an LLM call.

2. **LLM-driven** (when an API key is configured): a small Gemini /
   Anthropic / OpenAI call ranks the full catalog (specialists +
   workflow templates, including macro and parametric variants)
   against the prompt and returns the chosen target plus a one-line
   rationale. Routed via LiteLLM through ADK. Only consulted when the
   keyword router finds no match -- which is the case for free-form
   prose that doesn't name a known clinical phrase.

Both layers ultimately invoke the same underlying tools (or workflow
orchestrator) the per-specialist handlers and the composer already use,
so we get one consistent execution path independent of whether an LLM
is available. Falls back to a discovery summary when neither layer
matches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from apps._shared.specialist_routes import SPECIALIST_ROUTES, Route
from apps.composer.orchestrator import execute_workflow
from apps.composer.workflows import REGISTRY as WORKFLOW_REGISTRY


@dataclass
class DispatchResult:
    target_kind: str            # "workflow" | "specialist" | "discovery" | "fanout"
    target_label: str           # human-readable label for the chosen target
    target_slug: str | None     # specialist slug or workflow id
    rationale: str              # one-liner for the response narration
    output: Any | None = None   # raw tool / workflow output
    output_dump: dict | None = None
    error: str | None = None
    sub_results: list["DispatchResult"] | None = None  # populated for fanout


# ------------------------------------------------------------------ keyword

_WORKFLOW_KEYWORDS = (
    # Macro pipelines (composer-level): listed first so the longest-match
    # router picks the full pipeline over a single-leg workflow when the
    # prompt explicitly asks for the complete arc.
    ("complete_sepsis_pipeline", ("complete sepsis pipeline",
                                      "full sepsis pipeline",
                                      "sepsis pipeline",
                                      "complete sepsis arc",
                                      "full sepsis arc")),
    ("discharge_full",      ("full discharge bundle",
                                 "complete discharge bundle",
                                 "full discharge handoff",
                                 "complete discharge handoff",
                                 "full discharge arc",
                                 "complete discharge arc")),
    ("complete_chf_admission", ("complete chf admission",
                                     "full chf admission",
                                     "complete chf arc",
                                     "full chf arc")),
    ("chf_admission",       ("chf admission", "chf inpatient",
                                 "heart failure admission",
                                 "admit for chf", "admit for heart failure")),
    ("sepsis_workup",       ("sepsis workup", "septic patient",
                                 "febrile and hypotensive",
                                 "febrile guy with hypotension",
                                 "febrile patient with hypotension",
                                 "fever and hypotension", "septic shock")),
    ("discharge_planning",  ("discharge planning", "discharge plan",
                                 "ready to go home", "discharge bundle",
                                 "going home", "what is the plan",
                                 "what's the plan", "send him home",
                                 "send her home", "frequent admissions",
                                 "frequent readmission",
                                 "readmission risk", "readmission score",
                                 "30-day readmission",
                                 "run the readmission",
                                 "compute readmission",
                                 "lace index", "lace score",
                                 "lace assessment",
                                 "leaving us tomorrow",
                                 "leaving tomorrow",
                                 "what should we set up")),
    ("outpatient_med_review", ("outpatient med review", "med review",
                                  "polypharmacy review",
                                  "yearly med review")),
    ("stroke_alert",        ("stroke alert", "reperfusion decision",
                                 "code stroke")),
    # Vertical A additions
    ("trauma_resus_decision", ("polytrauma", "trauma activation",
                                  "trauma resus", "level 1 trauma",
                                  "high energy mechanism")),
    ("acute_chest_pain",    ("acute chest pain", "chest pain workup",
                                 "rule out acs", "heart score workup")),
    ("dka_management",      ("dka management", "ketoacidosis", "diabetic ketoacidosis",
                                 "ph 7.1", "anion gap acidosis")),
    ("aki_workup",          ("aki workup", "acute kidney injury",
                                 "kdigo workup", "creatinine doubled",
                                 "rising creatinine")),
    ("respiratory_failure_workup", ("respiratory failure", "hypoxemic failure",
                                       "hypercapnic failure", "acute hypoxia",
                                       "spo2 dropping")),
    # Vertical B
    ("inpatient_deterioration_response", ("rapid response", "rrt activation",
                                              "deterioration on the floor",
                                              "news2 escalation", "floor escalation")),
    ("icu_step_down",       ("step down", "icu step-down", "transfer to floor",
                                 "off the unit", "icu to floor")),
    ("inpatient_glycemic_control_workflow", ("glycemic control",
                                                  "inpatient hyperglycemia",
                                                  "insulin titration",
                                                  "sliding scale review")),
    ("periop_complications_response", ("post-op complication", "post op complication",
                                           "post-op chest pain", "post operative crisis",
                                           "rcri")),
    ("inpatient_falls_intervention", ("inpatient fall", "morse falls",
                                          "fall risk", "delirium screen",
                                          "cam-icu", "confused on the floor")),
    # Vertical C
    ("post_discharge_followup", ("post-discharge call", "day 7 call",
                                     "followup call", "phone followup",
                                     "after the hospital")),
    ("caregiver_handoff_prep", ("caregiver handoff", "family handoff",
                                    "going home with caregiver",
                                    "primary caregiver")),
    ("transitional_care_management", ("tcm visit", "transitional care",
                                         "99495", "99496",
                                         "post-discharge clinic")),
    ("chronic_disease_followup", ("chronic disease followup",
                                       "chronic care visit",
                                       "panel visit", "annual visit")),
    ("palliative_transition", ("palliative", "hospice transition",
                                   "comfort care", "end of life",
                                   "goals of care")),
    # Vertical D
    ("pgx_prescribing_check", ("pgx prescribing", "pharmacogenomic check",
                                   "starting clopidogrel", "cyp test",
                                   "cyp2c19", "vkorc1")),
    ("ddi_audit",          ("ddi audit", "drug interaction audit",
                                "polypharmacy audit", "med safety audit")),
    ("anticoagulant_review", ("warfarin review", "anticoagulant review",
                                  "doac review", "inr swing",
                                  "bleeding on warfarin")),
    ("antibiotic_stewardship", ("stewardship round", "antibiotic stewardship",
                                    "day 3 antibiotic", "de-escalate",
                                    "narrow spectrum")),
    ("opioid_safety_review", ("opioid safety", "narcotic review",
                                  "chronic pain regimen", "opioid + benzo",
                                  "naloxone education")),
    ("pediatric_dosing_review", ("pediatric dose", "child dose",
                                      "weight-based dose", "pediatric prescribing")),
    # Vertical E
    ("pediatric_acute_workup", ("pediatric acute", "sick child",
                                    "pediatric workup", "pews score",
                                    "child in ed")),
    ("mental_health_crisis", ("mental health crisis", "suicidal ideation",
                                   "psych emergency", "c-ssrs",
                                   "thinking about ending things",
                                   "psychiatric crisis")),
    ("maternal_obstetric_emergency", ("maternal emergency", "obstetric emergency",
                                          "preeclampsia", "pregnant with",
                                          "meows", "pregnant patient")),
    ("oncology_cycle_review", ("oncology cycle", "chemo cycle",
                                    "recist", "next cycle",
                                    "cancer followup")),
    ("geriatric_assessment", ("geriatric assessment", "comprehensive geriatric",
                                  "elderly review", "geriatric review",
                                  "frailty assessment")),
    ("rheumatology_followup", ("rheumatology", "ra followup",
                                    "biologic followup", "dmard review",
                                    "autoimmune followup")),
    ("transplant_immunosuppression_review", ("transplant review",
                                                  "tacrolimus level",
                                                  "immunosuppression",
                                                  "post-transplant")),
    ("infectious_disease_consult", ("id consult", "infectious disease consult",
                                        "complex infection", "id workup")),
    # Vertical F
    ("pre_op_optimization", ("pre-op optimization", "pre-op clinic",
                                  "pre-op evaluation", "preadmit clinic",
                                  "elective surgery prep")),
    ("periop_risk_stratification", ("periop risk", "anesthesia risk",
                                        "rcri ariscat caprini",
                                        "anesthesia evaluation",
                                        "surgical risk score")),
    ("post_op_recovery",   ("post-op recovery", "pacu recovery",
                                "post-op day", "recovery review",
                                "post-op rounds")),
    ("surgical_consent_workup", ("surgical consent", "informed consent",
                                      "shared decision", "pre-op consent",
                                      "consent conversation")),
    # Vertical G
    ("chart_to_codes",     ("chart to codes", "auto-code", "auto code",
                                "icd-10 + cpt", "billing codes from chart",
                                "coding pipeline")),
    ("clinical_documentation_polish", ("documentation polish",
                                            "end-of-shift docs",
                                            "polish my notes",
                                            "documentation pass")),
    ("progress_note_workflow", ("progress note", "daily round note",
                                     "soap note", "round documentation")),
    ("consult_letter_workflow", ("consult letter draft", "draft consult",
                                       "letter to specialist",
                                       "specialist letter")),
    ("discharge_summary_workflow", ("discharge summary draft",
                                         "draft discharge summary",
                                         "discharge note draft")),
    # Vertical H
    ("prior_auth_pipeline", ("prior auth", "prior authorization", "pa pipeline",
                                  "auth submission", "pa request")),
    ("denial_appeal_pipeline", ("denial appeal", "appeal pipeline",
                                     "they denied us", "denied claim",
                                     "fight the denial")),
    ("cost_effectiveness_review", ("cost-effectiveness", "qaly review",
                                        "evoi review", "shared decision cost",
                                        "is it worth it")),
    ("coverage_determination", ("coverage determination", "value defensible",
                                     "pre-submission self-check",
                                     "is this covered")),
    # Vertical I
    ("population_outreach", ("population outreach", "outreach pipeline",
                                  "syndromic surveillance pipeline",
                                  "outbreak heatmap", "vaccine outreach")),
    ("care_gap_closure_pipeline", ("care gap closure", "close the gap",
                                        "outreach for gap", "hedis closure")),
    ("hedis_quality_improvement", ("stars rating", "stars forecast",
                                        "hedis improvement",
                                        "quality bonus", "ma contract")),
    # Vertical J - horizontal expansion (J51..J58)
    ("pre_hospital_handoff", ("ems incoming", "field call", "ems alert",
                                  "pre-hospital handoff",
                                  "ambulance inbound")),
    ("vte_prophylaxis_review", ("vte prophylaxis", "dvt prophylaxis",
                                     "caprini score", "anticoag prophylaxis",
                                     "lmwh dose")),
    ("postpartum_followup", ("postpartum visit", "6-week postpartum",
                                  "post-partum check", "postnatal followup")),
    ("behavioral_health_step_down", ("mat followup", "mat handoff",
                                          "post-crisis followup",
                                          "buprenorphine followup",
                                          "step down behavioral")),
    ("vaccine_schedule_check", ("vaccine catch-up", "acip schedule",
                                     "vaccine gap", "immunization audit",
                                     "vaccine registry")),
    ("well_child_visit", ("well child", "well-child visit",
                              "pediatric routine", "annual pediatric")),
    ("ssi_prevention_bundle", ("surgical site infection",
                                    "ssi prevention", "pre-op abx",
                                    "perioperative antibiotic prophylaxis")),
    ("out_of_network_referral", ("out-of-network referral",
                                      "oon referral", "off-panel referral",
                                      "out of network specialist")),
)


def _match_workflow(prompt: str) -> str | None:
    """Return the workflow whose LONGEST keyword matches the prompt.

    Longest-match is required for the demo macros to win over their
    single-leg components: "complete sepsis pipeline" contains the
    substring "sepsis" but must route to `complete_sepsis_pipeline`,
    not to `sepsis_workup`. We score by the length of the matched
    keyword (a proxy for specificity) and return the highest-scoring
    workflow id.
    """
    p = prompt.lower()
    best: tuple[int, str] | None = None
    for wid, kws in _WORKFLOW_KEYWORDS:
        for k in kws:
            if k in p:
                if best is None or len(k) > best[0]:
                    best = (len(k), wid)
                break
    return None if best is None else best[1]


def _match_specialist_route(prompt: str) -> tuple[str, Route] | None:
    """Scan every specialist's route list for the longest-keyword match."""
    p = prompt.lower()
    best: tuple[int, str, Route] | None = None
    for slug, routes in SPECIALIST_ROUTES.items():
        for r in routes:
            for kw in r.keywords:
                if kw in p:
                    score = len(kw)
                    if best is None or score > best[0]:
                        best = (score, slug, r)
                    break
    if best is None:
        return None
    return best[1], best[2]


# ------------------------------------------------------------------ optional LLM


def _llm_available() -> bool:
    return bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


async def _llm_dispatch(prompt: str) -> list[tuple[str, str, str]] | None:
    """Ask the configured LLM to pick one or more targets.

    Returns a ranked list of (kind, target, rationale) tuples (1 - 3
    items). Multiple picks are intended for ambiguous prompts: the
    orchestrator runs all of them in parallel and merges the artifacts.
    Returns None if the LLM call fails or no key is set.
    """
    if not _llm_available():
        return None
    try:
        from litellm import acompletion
    except ImportError:
        return None

    catalog_lines: list[str] = ["WORKFLOWS (kind=workflow):"]
    for wid, wf in WORKFLOW_REGISTRY.items():
        catalog_lines.append(f"  - {wid} -- {wf.title}")
    catalog_lines.append("")
    catalog_lines.append("SPECIALIST TOOLS (kind=specialist):")
    for slug, routes in SPECIALIST_ROUTES.items():
        for r in routes:
            catalog_lines.append(
                f"  - {slug} / {r.tool_name} -- {r.skill_label}"
            )

    system = (
        "You are the TrustedRisk clinical dispatcher. Pick 1 to 3 targets "
        "from the catalog that together best answer the user's prompt. "
        "Use a single target when the prompt clearly maps to one tool; "
        "use 2-3 when the prompt is ambiguous, multi-faceted, or would "
        "benefit from a second-opinion ensemble. Order by relevance, "
        "most relevant first.\n\n"
        "Reply with one target per line, no preamble, no markdown, no "
        "code fence. Each line is exactly:\n"
        "  KIND<PIPE>TARGET<PIPE>RATIONALE\n"
        "where:\n"
        "- KIND is the literal word 'workflow' or 'specialist'.\n"
        "- TARGET is the workflow id (e.g. sepsis_workup) when KIND is "
        "workflow, or 'slug/tool_name' (e.g. trustedrisk-acute/"
        "compute_clinical_deterioration_score) when KIND is specialist.\n"
        "- RATIONALE is one short clinical sentence (<= 25 words).\n"
        "<PIPE> is the literal pipe character |. Do not use it inside "
        "RATIONALE. Do not invent kinds or targets that are not in the "
        "catalog.\n\n"
        "Examples:\n"
        "USER: febrile patient with hypotension, walk me through workup\n"
        "REPLY:\n"
        "workflow|sepsis_workup|Febrile and hypotensive presentation suggests sepsis bundle.\n\n"
        "USER: insurance denied her biologic, write a push-back\n"
        "REPLY:\n"
        "specialist|trustedrisk-appeals/_chained_appeal_letter|Denial of medication needs an appeal letter draft.\n"
        "specialist|trustedrisk-appeals/compute_appeal_escalation_path|Provide the escalation roadmap if first appeal fails.\n\n"
        "USER: 4-year-old with cough and rapid breathing\n"
        "REPLY:\n"
        "specialist|trustedrisk-pediatric/compute_pediatric_early_warning|Pediatric respiratory distress needs PEWS scoring.\n"
        "specialist|trustedrisk-preadmit/compute_symptom_red_flag_check|Cross-check parental concern against general red-flag rules.\n"
    )
    user = f"USER: {prompt}\n\nCATALOG:\n" + "\n".join(catalog_lines)

    model = os.environ.get("ADK_MODEL", "gemini/gemini-2.0-flash")
    # Total LLM budget: 2.5 s. PO's chat client has a soft timeout on
    # external agent responses, and adding a retry on top of a slow
    # generation pushed us past it. One attempt with a hard wall-clock
    # cap keeps us under the budget; on failure we fall back cleanly to
    # the keyword router before the user sees any timeout.
    import asyncio as _asyncio
    try:
        resp = await _asyncio.wait_for(
            acompletion(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=300,
            ),
            timeout=2.5,
        )
        content = (resp["choices"][0]["message"]["content"] or "").strip()
    except (Exception, _asyncio.TimeoutError):
        return None
    if not content:
        return None

    picks: list[tuple[str, str, str]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith("reply:"):
            line = line[len("reply:"):].strip()
            if not line:
                continue
        if "|" not in line:
            continue
        parts = line.split("|", 2)
        if len(parts) != 3:
            continue
        kind, target, rationale = (p.strip() for p in parts)
        if kind not in ("workflow", "specialist"):
            continue
        target = target.replace("/", ":")
        picks.append((kind, target, rationale))
        if len(picks) >= 3:
            break
    return picks or None


# ------------------------------------------------------------------ execution


_FHIR_DERIVED_KEYS = (
    "fhir_bundle", "admission_meds", "discharge_meds", "medications",
    "patient_demographics", "chief_complaint", "vital_signs",
    "vital_signs_obs", "vital_signs_dict", "age",
    "age_months", "infection_source", "patient_factors",
    "current_medications", "immunizations",
)

# Workflow inputs that are list-shaped or dict-shaped by convention.
# The dispatcher picks a typed default ([] / {} / "") when the caller
# did not supply a value, otherwise downstream tools crash on type
# errors instead of producing a clean abstain (e.g. a Pydantic
# `list[str] | None` field rejects "" silently). Note: `vital_signs` is
# intentionally NOT defaulted here -- the FHIR overlay may emit either a
# list of dicts (deterioration tool) or a single dict (admission triage)
# depending on workflow shape, and forcing an empty default would cause
# the wrong tool to abstain spuriously when the bundle has the data.
_LIST_DEFAULT_KEYS = (
    "followup_plan", "patient_instructions",
    "medications", "admission_meds", "discharge_meds",
    "current_medications", "injuries", "immunizations",
)
_DICT_DEFAULT_KEYS = (
    "patient_demographics", "fhir_bundle", "decision_card",
    "nihss_item_scores", "structured_features", "patient_factors",
)


def _typed_default(key: str) -> Any:
    if key in _LIST_DEFAULT_KEYS:
        return []
    if key in _DICT_DEFAULT_KEYS:
        return {}
    return ""


async def _overlay_real_fhir(
    inputs: dict, metadata: dict,
    provenance_out: dict | None = None,
) -> tuple[bool, str | None]:
    """Fetch the live patient bundle and populate FHIR-derived inputs.

    Callers pass an empty or partially-filled inputs dict and a metadata
    dict that must contain `fhir_server_url`. This function fetches the
    live bundle and writes `fhir_bundle`, `admission_meds`,
    `discharge_meds`, `medications`, and `patient_demographics` into
    `inputs`. If the fetch fails, returns (False, reason) so the caller
    can abstain cleanly.

    Returns (ok, reason). On ok=True, `inputs` has been populated with
    live values. On ok=False, FHIR-derived keys are set to empty
    sentinels so no downstream tool can accidentally run on stale data.
    """
    # Zero out FHIR-derived keys up front so that if the fetch fails
    # the caller gets typed sentinels rather than stale values.
    _LIST_FHIR_KEYS = ("admission_meds", "discharge_meds", "medications",
                          "vital_signs", "vital_signs_obs",
                          "current_medications",
                          "immunizations")
    _DICT_FHIR_KEYS = ("patient_demographics", "patient_factors",
                          "vital_signs_dict")
    _SCALAR_NONE_KEYS = ("fhir_bundle", "chief_complaint",
                                "infection_source", "age", "age_months")
    for k in _FHIR_DERIVED_KEYS:
        if k in inputs:
            if k in _LIST_FHIR_KEYS:
                inputs[k] = []
            elif k in _DICT_FHIR_KEYS:
                inputs[k] = {}
            else:
                inputs[k] = None

    try:
        from mcp_server.fhir.client import (
            fetch_patient_bundle, resolve_patient_id,
        )
        from mcp_server.tools.medication_reconciliation import (
            _extract_meds_from_bundle,
        )
    except ImportError as exc:
        return False, f"mcp_server import unavailable: {exc}"
    try:
        pid = await resolve_patient_id(metadata.get("patient_id"))
        bundle = await fetch_patient_bundle(pid)
    except Exception as exc:
        return False, f"FHIR fetch failed: {type(exc).__name__}: {exc}"
    if not isinstance(bundle, dict):
        return False, "FHIR fetch returned non-dict bundle"

    # Populate the FHIR-derived inputs unconditionally. Macro workflows
    # only declare `["patient_id"]` as required_inputs, so the previous
    # gated approach (`if k in inputs:`) silently dropped the bundle and
    # med lists by the time the sub-workflow ran. The downstream tools
    # are responsible for abstaining when their specific field is empty.
    inputs["fhir_bundle"] = bundle

    adm = _extract_meds_from_bundle(bundle, phase="admission")
    dis = _extract_meds_from_bundle(bundle, phase="discharge")
    inputs["admission_meds"] = adm
    inputs["discharge_meds"] = dis
    inputs["medications"] = dis or adm
    # Alias: many workflows reference `${input.current_medications}` while
    # only `medications` is overlay-populated. Treating the two as
    # equivalent matches the meaning ("the patient's currently active
    # med list") and prevents missing_medication_list abstain on the
    # detect_polypharmacy_concerns / DDI / opioid steps.
    inputs["current_medications"] = inputs["medications"]

    demo = _patient_demographics_from_bundle(bundle)
    if demo:
        inputs["patient_demographics"] = demo

    # Workflow-input overlays: extract chief_complaint, vital_signs, age,
    # infection_source, patient_factors, immunizations from the bundle.
    # These are populated unconditionally (not gated on `if k in inputs`)
    # because macro workflows declare only `["patient_id"]` as
    # required_inputs -- if we gated, the sub-workflows wouldn't see the
    # FHIR-derived fields when the macro hops down a level. Tools that
    # don't need a field are unaffected; tools that need it run on real
    # data instead of typed-default sentinels.
    cc = _extract_chief_complaint_from_bundle(bundle)
    if cc:
        inputs["chief_complaint"] = cc
    vs = _extract_vital_signs_from_bundle(bundle)
    if vs:
        inputs["vital_signs"] = vs
        # Workflows expose two narrower aliases for the same data:
        #   - `vital_signs_obs` is consumed by NEWS2 / PEWS / MEOWS, all
        #     of which expect a list[dict] of {type, value, unit, observed_at}.
        #   - `vital_signs_dict` is consumed by compute_admission_triage,
        #     which expects a flat dict keyed by type. Collapse the most
        #     recent observation per type into a single dict.
        inputs["vital_signs_obs"] = vs
        collapsed: dict[str, Any] = {}
        for entry in vs:
            t = entry.get("type")
            if isinstance(t, str) and t:
                collapsed[t] = entry.get("value")
        if collapsed:
            inputs["vital_signs_dict"] = collapsed
            # Per-tool scalar aliases. PEWS / qSOFA components / GRACE /
            # APACHE / preeclampsia all expect individual scalar kwargs
            # (heart_rate, respiratory_rate, spo2, systolic_bp, ...) rather
            # than a vital_signs container. Backend.call filters by
            # signature so a tool that doesn't declare the kwarg simply
            # ignores it; tools that do receive the bundle-derived value.
            _SCALAR_VS_ALIASES = {
                "heart_rate":      ("heart_rate",),
                "respiratory_rate": ("respiratory_rate",),
                "spo2":             ("spo2", "preop_spo2_pct"),
                "systolic_bp":      ("systolic_bp", "systolic_bp_mmHg"),
                "diastolic_bp":     ("diastolic_bp",),
                "temperature":      ("temperature", "temperature_c",
                                       "temperature_celsius"),
            }
            for vs_key, aliases in _SCALAR_VS_ALIASES.items():
                if vs_key in collapsed:
                    val = collapsed[vs_key]
                    for alias in aliases:
                        inputs.setdefault(alias, val)
    demo_for_age = demo if isinstance(demo, dict) else {}
    a = demo_for_age.get("age") if isinstance(demo_for_age, dict) else None
    if isinstance(a, int):
        inputs["age"] = a
        # Workflows like transitional_care_management declare
        # `patient_age` and `patient_sex` as their input names. Populate
        # both alias forms so a workflow declaring either receives the
        # bundle-derived value rather than the empty-string default
        # (which downstream tools detect as an HL7 unknown sentinel).
        inputs["patient_age"] = a
    s = demo_for_age.get("sex") if isinstance(demo_for_age, dict) else None
    if isinstance(s, str) and s:
        inputs["sex"] = s
        inputs["patient_sex"] = s
    am = _extract_age_months_from_bundle(bundle)
    if am is not None:
        inputs["age_months"] = am
    src = _extract_infection_source_from_bundle(bundle)
    if src:
        inputs["infection_source"] = src
    pf = _extract_patient_factors_from_bundle(bundle)
    if pf:
        inputs["patient_factors"] = pf
    imms = _extract_immunizations_from_bundle(bundle)
    if imms:
        inputs["immunizations"] = imms

    # LOINC-mapped lab values — populates the discrete arg names that
    # AKI / DKA / preeclampsia / contrast / UGIB / hepatitis tools expect
    # (creatinine_baseline_mg_dl, ph, bicarbonate_meq_l, glucose_mg_dl,
    # platelets_per_ul, bilirubin_mg_dl, ast_u_l, alt_u_l, hemoglobin_g_dl,
    # blood_urea_mmol_l). Each value is the most recent observation of
    # that LOINC; baseline is the oldest. Absent codes simply leave the
    # corresponding key unset, so the tool falls back to its own abstain
    # path when the data isn't there.
    labs = _extract_clinical_labs_from_bundle(bundle)
    for k, v in labs.items():
        if v is not None:
            inputs[k] = v
    ga = _extract_gestational_age_weeks_from_bundle(bundle)
    if ga is not None:
        inputs["gestational_age_weeks"] = ga

    # Concatenated DocumentReference plaintext: feeds compute_icd10_suggest,
    # compute_cpt_suggest, compute_coding_audit when no chart_text was
    # passed explicitly. Only DocumentReference bodies the bundle already
    # carries -- no fabrication.
    try:
        from mcp_server.tools._scribe_helpers import extract_full_chart_text
        chart_text = extract_full_chart_text(bundle)
        if chart_text:
            inputs["chart_text"] = chart_text
            # compute_cpt_suggest takes `procedure_text`; the same chart
            # narrative is the most-relevant source unless the caller
            # supplies a tighter procedure-only excerpt.
            inputs.setdefault("procedure_text", chart_text)
    except ImportError:
        pass

    # Provenance counts: an upstream chat-LLM rendering the artifact
    # can read these numbers to verify the engine operated on real
    # FHIR resources (and not on a Jane-Doe demo template). The counts
    # double as a transparency contract for regulatory audits.
    if provenance_out is not None:
        from collections import Counter
        c: Counter = Counter()
        for e in bundle.get("entry") or []:
            res = e.get("resource") if isinstance(e, dict) else None
            if isinstance(res, dict):
                c[res.get("resourceType", "?")] += 1
        import datetime as _dt
        provenance_out.update({
            "fhir_server_url": metadata.get("fhir_server_url"),
            "patient_id": metadata.get("patient_id"),
            "fetched_at_iso": _dt.datetime.now(_dt.timezone.utc)
                .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_total_resources": sum(c.values()),
            "resource_counts": dict(c),
        })
    return True, None


def _patient_demographics_from_bundle(bundle: dict) -> dict:
    """Pull a {age, sex, race, ethnicity, insurance_type} dict.

    Race / ethnicity live in US Core extensions on the Patient resource;
    insurance_type is derived from Coverage.payor.display when a Coverage
    resource is present. Absent fields are simply omitted (downstream
    audit tools tolerate partial demographics).
    """
    import datetime as _dt
    out: dict = {}
    entries = bundle.get("entry") or []
    patient = next(
        (e.get("resource") for e in entries
         if isinstance(e, dict)
         and isinstance(e.get("resource"), dict)
         and e["resource"].get("resourceType") == "Patient"),
        None,
    )
    if not isinstance(patient, dict):
        return out
    bd = patient.get("birthDate")
    if isinstance(bd, str):
        try:
            born = _dt.date.fromisoformat(bd[:10])
            out["age"] = (_dt.date.today() - born).days // 365
        except ValueError:
            pass
    g = patient.get("gender")
    if isinstance(g, str):
        out["sex"] = g
    for ext in patient.get("extension") or []:
        url = (ext or {}).get("url", "")
        if "us-core-race" in url:
            for sub in ext.get("extension") or []:
                if sub.get("url") == "ombCategory":
                    coding = (sub.get("valueCoding") or {}).get("display") \
                        or (sub.get("valueCoding") or {}).get("code")
                    if isinstance(coding, str):
                        out["race"] = coding.lower()
        elif "us-core-ethnicity" in url:
            for sub in ext.get("extension") or []:
                if sub.get("url") == "ombCategory":
                    disp = (sub.get("valueCoding") or {}).get("display")
                    if isinstance(disp, str):
                        # "Not Hispanic or Latino" must NOT match the
                        # plain "hispanic" substring -- check the explicit
                        # negation prefix first.
                        d = disp.lower()
                        if "not hispanic" in d or "not latino" in d:
                            out["ethnicity"] = "non_hispanic"
                        elif "hispanic" in d or "latino" in d:
                            out["ethnicity"] = "hispanic"
                        else:
                            out["ethnicity"] = "non_hispanic"
    coverage = next(
        (e.get("resource") for e in entries
         if isinstance(e, dict)
         and isinstance(e.get("resource"), dict)
         and e["resource"].get("resourceType") == "Coverage"),
        None,
    )
    if isinstance(coverage, dict):
        payors = coverage.get("payor") or []
        if payors and isinstance(payors[0], dict):
            disp = (payors[0].get("display") or "").lower()
            if "medicaid" in disp:
                out["insurance_type"] = "medicaid"
            elif "medicare" in disp:
                out["insurance_type"] = "medicare"
            elif disp:
                out["insurance_type"] = "commercial"
    return out


# ─── FHIR -> workflow-input extractors ────────────────────────────────


_LOINC_VITAL_MAP: dict[str, str] = {
    "8867-4": "heart_rate",
    "9279-1": "respiratory_rate",
    "59408-5": "spo2",
    "2708-6": "spo2",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8310-5": "temperature",
    "8716-3": "temperature",
    "85354-9": "blood_pressure_panel",  # composite, expanded below
    "9269-2": "consciousness",  # GCS total
    "9270-0": "consciousness",  # AVPU
}


def _extract_chief_complaint_from_bundle(bundle: dict) -> str:
    """Pull a chief_complaint string from the most recent Encounter.

    Prefers an in-progress encounter, then the most recently started.
    Reads `Encounter.reasonCode[*].coding[*].display` and `text`.
    """
    entries = bundle.get("entry") or []
    encounters: list[dict] = [
        e.get("resource") for e in entries
        if isinstance(e, dict)
        and isinstance(e.get("resource"), dict)
        and e["resource"].get("resourceType") == "Encounter"
    ]
    if not encounters:
        return ""
    in_progress = [e for e in encounters
                       if e.get("status") == "in-progress"]
    candidates = in_progress or encounters

    def _start(enc: dict) -> str:
        return ((enc.get("period") or {}).get("start") or "")

    candidates = sorted(candidates, key=_start, reverse=True)
    enc = candidates[0]
    bits: list[str] = []
    for r in (enc.get("reasonCode") or []):
        text = (r.get("text") or "").strip()
        if text:
            bits.append(text)
            continue
        for c in (r.get("coding") or []):
            disp = (c.get("display") or "").strip()
            if disp:
                bits.append(disp)
                break
    return " and ".join(bits).lower()


def _extract_vital_signs_from_bundle(bundle: dict) -> list[dict]:
    """Return the latest vital-sign Observation set as a list of dicts.

    Each dict has {type, value, unit, observed_at} keys, matching the
    `VitalSign` Pydantic schema consumed by `compute_clinical_deterioration_score`.
    Composite blood-pressure panels are expanded into separate
    systolic_bp / diastolic_bp entries.
    """
    entries = bundle.get("entry") or []
    out: list[dict] = []
    for e in entries:
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Observation":
            continue
        code = r.get("code") or {}
        loinc = next(
            (c.get("code") for c in (code.get("coding") or [])
             if c.get("system") == "http://loinc.org"),
            None,
        )
        if not loinc:
            continue
        kind = _LOINC_VITAL_MAP.get(loinc)
        if not kind:
            continue
        observed = r.get("effectiveDateTime") or r.get("issued") or ""
        if kind == "blood_pressure_panel":
            for comp in (r.get("component") or []):
                cc = comp.get("code") or {}
                cloinc = next(
                    (cc2.get("code") for cc2 in (cc.get("coding") or [])
                     if cc2.get("system") == "http://loinc.org"),
                    None,
                )
                ckind = _LOINC_VITAL_MAP.get(cloinc or "")
                if not ckind:
                    continue
                vq = comp.get("valueQuantity") or {}
                val = vq.get("value")
                if val is None:
                    continue
                out.append({
                    "type": ckind,
                    "value": float(val),
                    "unit": vq.get("unit"),
                    "observed_at": observed,
                })
            continue
        vq = r.get("valueQuantity") or {}
        val = vq.get("value")
        if val is None:
            v_str = r.get("valueString")
            if isinstance(v_str, str) and v_str:
                out.append({
                    "type": kind, "value": v_str,
                    "unit": None, "observed_at": observed,
                })
            continue
        out.append({
            "type": kind,
            "value": float(val),
            "unit": vq.get("unit"),
            "observed_at": observed,
        })
    return out


def _extract_age_months_from_bundle(bundle: dict) -> int | None:
    """Compute pediatric age in months from Patient.birthDate."""
    import datetime as _dt
    entries = bundle.get("entry") or []
    patient = next(
        (e.get("resource") for e in entries
         if isinstance(e, dict)
         and isinstance(e.get("resource"), dict)
         and e["resource"].get("resourceType") == "Patient"),
        None,
    )
    if not isinstance(patient, dict):
        return None
    bd = patient.get("birthDate")
    if not isinstance(bd, str):
        return None
    try:
        born = _dt.date.fromisoformat(bd[:10])
    except ValueError:
        return None
    today = _dt.date.today()
    months = (today.year - born.year) * 12 + (today.month - born.month)
    if today.day < born.day:
        months -= 1
    return max(0, months)


def _extract_infection_source_from_bundle(bundle: dict) -> str:
    """Heuristically infer an infection_source from Conditions / Encounter
    reasonCodes. Returns one of: urinary / pneumonia / skin_soft_tissue /
    intra_abdominal / cns / bloodstream / unknown / "" (no signal).
    """
    keywords: list[tuple[tuple[str, ...], str]] = [
        (("urinary", "uti", "pyelo", "cystitis"), "urinary"),
        (("pneumonia", "lower respiratory"), "pneumonia"),
        (("cellulitis", "skin", "soft tissue", "abscess"),
         "skin_soft_tissue"),
        (("peritonitis", "intra-abdominal", "diverticulitis",
          "cholangitis", "cholecystitis"),
         "intra_abdominal"),
        (("meningitis", "encephalitis", "ventriculitis"), "cns"),
        (("bacteremia", "bloodstream", "endocarditis"), "bloodstream"),
    ]
    haystack: list[str] = []
    for e in bundle.get("entry") or []:
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict):
            continue
        if r.get("resourceType") in ("Condition", "Encounter"):
            for cc in (r.get("code"), *(r.get("reasonCode") or [])):
                if not isinstance(cc, dict):
                    continue
                txt = (cc.get("text") or "").lower()
                if txt:
                    haystack.append(txt)
                for c in (cc.get("coding") or []):
                    disp = (c.get("display") or "").lower()
                    if disp:
                        haystack.append(disp)
    blob = " | ".join(haystack)
    for kws, label in keywords:
        if any(k in blob for k in kws):
            return label
    return ""


def _extract_patient_factors_from_bundle(bundle: dict) -> dict:
    """Build the `patient_factors` dict consumed by
    compute_empiric_antibiotic_selection (allergies, eGFR, MRSA risk, etc.).
    """
    factors: dict = {}
    for e in bundle.get("entry") or []:
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict):
            continue
        rt = r.get("resourceType")
        if rt == "AllergyIntolerance":
            code = r.get("code") or {}
            txt = (code.get("text") or "").lower()
            for c in (code.get("coding") or []):
                txt = (txt + " " + (c.get("display") or "")).lower()
            if "penicillin" in txt or "pcn" in txt or "amoxicillin" in txt:
                factors["allergy_penicillin"] = True
            if "carbapenem" in txt or "meropenem" in txt:
                factors["allergy_carbapenem"] = True
            if "sulfa" in txt or "sulfonamide" in txt:
                factors["allergy_sulfa"] = True
        elif rt == "Observation":
            code = r.get("code") or {}
            loinc = next(
                (c.get("code") for c in (code.get("coding") or [])
                 if c.get("system") == "http://loinc.org"),
                None,
            )
            if loinc == "62238-1":  # eGFR
                vq = r.get("valueQuantity") or {}
                val = vq.get("value")
                if val is not None:
                    factors["egfr_ml_min"] = float(val)
        elif rt == "Condition":
            code = r.get("code") or {}
            txt = ((code.get("text") or "")
                   + " "
                   + " ".join(c.get("display") or ""
                                for c in (code.get("coding") or []))
                   ).lower()
            if "immunocompromised" in txt or "neutropen" in txt:
                factors["immunocompromised"] = True
            if "diabetes" in txt:
                factors.setdefault("comorbidities", []).append("dm")
    demo = _patient_demographics_from_bundle(bundle)
    if "age" in demo:
        factors["age"] = demo["age"]
    return factors


def _extract_immunizations_from_bundle(bundle: dict) -> list[dict]:
    """Return the patient's Immunization history as a flat list of dicts
    with {cvx_code, display, occurrence_date}.
    """
    out: list[dict] = []
    for e in bundle.get("entry") or []:
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Immunization":
            continue
        vc = r.get("vaccineCode") or {}
        cvx = next(
            (c.get("code") for c in (vc.get("coding") or [])
             if c.get("system") == "http://hl7.org/fhir/sid/cvx"),
            None,
        )
        disp = (vc.get("text")
                or next((c.get("display") for c in (vc.get("coding") or [])
                         if c.get("display")), ""))
        out.append({
            "cvx_code": cvx,
            "display": disp,
            "occurrence_date": r.get("occurrenceDateTime"),
        })
    return out


# LOINC -> kwarg name(s) of the lab value AKI / DKA / contrast / UGIB /
# hepatitis tools expect. Each entry maps a LOINC code to a list of
# (kwarg, role) tuples where role is "current" (most recent observation)
# or "baseline" (oldest observation). Tools that ask for a single value
# get the "current" reading; tools that ask for a baseline + current
# pair (AKI staging) get both. Multiple kwargs may be populated from
# the same LOINC when the same data fits two argument names.
_LOINC_LAB_MAP: dict[str, list[tuple[str, str]]] = {
    # Creatinine — AKI staging needs both baseline and current.
    "2160-0": [("creatinine_current_mg_dl", "current"),
                ("creatinine_baseline_mg_dl", "baseline"),
                ("creatinine_mg_dl", "current"),
                ("serum_creatinine_mg_dl", "current"),
                ("preop_creatinine_mg_dl", "current")],
    # eGFR (CKD-EPI) — already used by patient_factors but also fed into
    # contrast safety / chemo dose adjustment / RCRI when available.
    "62238-1": [("egfr_ml_min", "current")],
    # Arterial pH — DKA + APACHE.
    "11558-4": [("ph", "current"), ("arterial_ph", "current")],
    "2744-1":  [("ph", "current"), ("arterial_ph", "current")],
    # Bicarbonate (HCO3) — DKA + dialysis decision.
    "1963-8":  [("bicarbonate_meq_l", "current")],
    "1959-6":  [("bicarbonate_meq_l", "current")],
    # Anion gap — DKA.
    "1863-0":  [("anion_gap", "current")],
    # Serum glucose — DKA + endocrine.
    "2345-7":  [("glucose_mg_dl", "current")],
    "2339-0":  [("glucose_mg_dl", "current")],
    # Potassium — DKA + dialysis + APACHE.
    "2823-3":  [("potassium_meq_l", "current"),
                 ("serum_potassium_mmol_l", "current")],
    # Sodium — APACHE.
    "2951-2":  [("serum_sodium_mmol_l", "current")],
    # Platelets — preeclampsia / SOFA / Maddrey / GBS / IPSS-R.
    "777-3":   [("platelets_per_ul", "current"),
                 ("platelets_thousands_per_uL", "current_thousands"),
                 ("platelet_count_per_ul", "current")],
    # Total bilirubin — preeclampsia / SOFA / Maddrey.
    "1975-2":  [("bilirubin_mg_dl", "current"),
                 ("total_bilirubin_mg_dl", "current"),
                 ("serum_bilirubin_mg_dl", "current")],
    # AST.
    "1920-8":  [("ast_u_l", "current"), ("ast_ul", "current")],
    # ALT.
    "1742-6":  [("alt_u_l", "current"), ("alt_ul", "current")],
    # Hemoglobin — GBS / IPSS-R / preop ariscat.
    "718-7":   [("hemoglobin_g_dl", "current"),
                 ("preop_hemoglobin_g_dl", "current")],
    # Hematocrit — APACHE.
    "4544-3":  [("hematocrit_pct", "current")],
    # WBC.
    "6690-2":  [("wbc_thousands_per_uL", "current_thousands"),
                 ("wbc_per_ul_thousand", "current_thousands")],
    # ANC.
    "751-8":   [("anc_per_ul", "current"),
                 ("anc_thousands_per_uL", "current_thousands"),
                 ("absolute_neutrophil_count_per_ul", "current"),
                 ("neutrophil_count_per_ul", "current")],
    # BUN (mg/dL) — GBS expects mmol/L; conversion handled below.
    "3094-0":  [("bun_mg_dl", "current"), ("blood_urea_mmol_l", "current_bun_mmol")],
    # Lactate.
    "32693-4": [("lactate_mmol_l", "current"), ("lactate_mg_dl", "current")],
    # Albumin.
    "1751-7":  [("albumin_g_dl", "current"), ("serum_albumin_g_dl", "current")],
    # Beta-2 microglobulin (myeloma ISS).
    "1952-5":  [("serum_beta2_microglobulin_mg_l", "current"),
                 ("beta2_microglobulin_mg_l", "current")],
    # INR.
    "6301-6":  [("inr", "current")],
    # Prothrombin time (sec).
    "5902-2":  [("patient_pt_seconds", "current"),
                 ("prothrombin_time_sec", "current")],
}


def _extract_clinical_labs_from_bundle(bundle: dict) -> dict:
    """Return a flat dict of the most-recent lab values keyed by the
    discrete kwarg name that compute_* tools accept.

    For codes that map to "baseline" + "current" pairs (creatinine via
    LOINC 2160-0), the oldest observation populates the baseline kwarg
    and the most recent populates the current kwarg. When only one
    observation is present, baseline = current.

    Values are NOT inferred or fabricated — only LOINC-coded
    Observation.valueQuantity readings are used. Tools whose required
    kwarg ends up unset will follow their own abstain path.
    """
    import datetime as _dt
    by_code: dict[str, list[tuple[float, str]]] = {}
    for entry in bundle.get("entry") or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") != "Observation":
            continue
        code = r.get("code") or {}
        loinc = next(
            (c.get("code") for c in (code.get("coding") or [])
             if c.get("system") == "http://loinc.org"),
            None,
        )
        if not loinc or loinc not in _LOINC_LAB_MAP:
            continue
        vq = r.get("valueQuantity") or {}
        val = vq.get("value")
        if val is None:
            continue
        observed = (
            r.get("effectiveDateTime") or r.get("issued") or "1970-01-01"
        )
        try:
            value = float(val)
        except (TypeError, ValueError):
            continue
        by_code.setdefault(loinc, []).append((value, observed))

    out: dict = {}
    for loinc, observations in by_code.items():
        # Sort by observed timestamp (string sort works for ISO-8601).
        observations.sort(key=lambda x: x[1])
        oldest_value = observations[0][0]
        latest_value = observations[-1][0]
        for kwarg, role in _LOINC_LAB_MAP[loinc]:
            if role == "baseline":
                out[kwarg] = oldest_value
            elif role == "current":
                out[kwarg] = latest_value
            elif role == "current_thousands":
                # Tools that want platelets / WBC / ANC in thousands/uL
                # take the value as-is when the source LOINC reports in
                # thousands; if the source unit is per-uL, divide by 1000.
                out[kwarg] = (latest_value / 1000.0
                                if latest_value > 1000 else latest_value)
            elif role == "current_bun_mmol":
                # Convert mg/dL -> mmol/L (Glasgow-Blatchford expects
                # mmol/L; BUN reported in mg/dL on US LOINC).
                out[kwarg] = round(latest_value * 0.357, 2)
    return out


def _extract_gestational_age_weeks_from_bundle(bundle: dict) -> int | None:
    """Find a gestational-age-weeks integer from Conditions / Encounters.

    Looks at any resource whose code text or description mentions
    'gestational age' / 'gestation' / 'weeks pregnancy' and returns the
    first integer found. Used to populate
    `compute_preeclampsia_assessment.gestational_age_weeks`.
    """
    import re
    pattern = re.compile(r"(\d{1,2})\s*(?:weeks?|wks?|w)", re.IGNORECASE)
    for e in bundle.get("entry") or []:
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict):
            continue
        rt = r.get("resourceType")
        if rt not in ("Condition", "Observation", "Encounter"):
            continue
        # Check obvious text fields.
        haystack: list[str] = []
        code = r.get("code") or {}
        haystack.append((code.get("text") or "").lower())
        for c in code.get("coding") or []:
            haystack.append((c.get("display") or "").lower())
        for n in r.get("note") or []:
            haystack.append((n.get("text") or "").lower())
        # Numeric Observation: valueQuantity in 'wk' / 'weeks'.
        if rt == "Observation":
            txt = (
                (code.get("text") or "") + " "
                + " ".join(c.get("display") or ""
                              for c in code.get("coding") or [])
            ).lower()
            if "gestation" in txt:
                vq = r.get("valueQuantity") or {}
                val = vq.get("value")
                unit = (vq.get("unit") or "").lower()
                if val is not None and ("wk" in unit or "week" in unit):
                    try:
                        return int(float(val))
                    except (TypeError, ValueError):
                        pass
        for s in haystack:
            if "gestation" not in s and "pregnan" not in s:
                continue
            m = pattern.search(s)
            if m:
                try:
                    return int(m.group(1))
                except ValueError:
                    continue
    return None


async def _run_workflow(workflow_id: str, metadata: dict | None) -> DispatchResult:
    workflow = WORKFLOW_REGISTRY.get(workflow_id)
    if workflow is None:
        return DispatchResult(
            target_kind="workflow", target_label=workflow_id,
            target_slug=workflow_id,
            rationale=f"Unknown workflow id {workflow_id!r}.",
            error="unknown_workflow",
        )
    # Require a real FHIR context. The dispatcher never falls back to
    # hardcoded demo fixtures; without a live bundle the workflow would
    # produce output for a fabricated patient, which is clinically unsafe.
    if not isinstance(metadata, dict) or not metadata.get("fhir_server_url"):
        return DispatchResult(
            target_kind="workflow", target_label=workflow.title,
            target_slug=workflow_id,
            rationale=(
                f"Abstained from running {workflow.title!r}: the dispatcher "
                f"requires a real FHIR context (fhir_server_url in message "
                f"metadata or live SHARP headers). No demo fixtures are used "
                f"as a fallback."
            ),
            error="no_fhir_context: dispatcher requires a real FHIR context "
                  "(chat-mode metadata or live SHARP headers)",
            output_dump={"abstain_recommended": True},
        )
    inputs: dict[str, Any] = {}
    # Always-propagate metadata keys: tool inputs that any downstream
    # specialist may legitimately consume.
    _ALWAYS = ("patient_id", "encounter_id")
    for k, v in metadata.items():
        if v is None:
            continue
        if k in _ALWAYS:
            inputs[k] = v
    # Alias `patient_reference` <- `patient_id` so scribe tools (HnP,
    # progress note, discharge summary, consult letter) that name the
    # arg `patient_reference` receive the same value the dispatcher
    # propagates as `patient_id`.
    if "patient_id" in inputs and "patient_reference" not in inputs:
        inputs["patient_reference"] = inputs["patient_id"]
    # Drop any stale values for FHIR-derived inputs and rebuild from the
    # live bundle. If the fetch fails, abstain rather than run on an empty
    # context.
    bundle_provenance: dict = {}
    ok, reason = await _overlay_real_fhir(
        inputs, metadata, provenance_out=bundle_provenance,
    )
    if not ok:
        return DispatchResult(
            target_kind="workflow", target_label=workflow.title,
            target_slug=workflow_id,
            rationale=(
                f"Abstained from running {workflow.title!r}: a real "
                f"FHIR-context was advertised but the live bundle "
                f"could not be retrieved. {reason}"
            ),
            error=f"abstain:fhir_unavailable: {reason}",
        )
    for k in workflow.required_inputs:
        inputs.setdefault(k, _typed_default(k))
    try:
        execution = await execute_workflow(workflow, inputs)
    except Exception as exc:
        return DispatchResult(
            target_kind="workflow", target_label=workflow.title,
            target_slug=workflow_id,
            rationale=f"Workflow {workflow_id!r} crashed before completion.",
            error=f"{type(exc).__name__}: {exc}",
        )
    n_steps = len(execution.steps)
    ok_steps = sum(1 for s in execution.steps if not s.error)

    # Detect inner-step abstain flags so the rationale exposed to the
    # chat-LLM cannot silently report a numeric value from an abstained
    # tool as if it were a calibrated estimate. This matters in
    # particular for compute_readmission_risk, which now abstains in
    # the LACE calibration-plateau OOD band but still populates
    # probability_mean for transparency. Macro hops have a nested
    # output_dump shape {inner_step_id: inner_dump}; we recurse one
    # level so a sub-workflow that abstained on `risk` still surfaces.
    # Optional steps are a separate axis from abstain. A step the
    # workflow author marked `optional=True` is one whose output the
    # downstream chain doesn't depend on -- so its abstain remains
    # visible (transparency) but does not flip the workflow-level
    # abstain_recommended flag. Only required steps that abstain
    # warrant the "the engine refused to commit" top-level signal.
    step_is_optional: dict[str, bool] = {
        s.id: s.optional for s in workflow.steps
    }
    # (step_id, abstain_reason, optional)
    abstained_steps: list[tuple[str, str, bool]] = []
    for s in execution.steps:
        od = s.output_dump if isinstance(s.output_dump, dict) else None
        if not od:
            continue
        is_opt = bool(step_is_optional.get(s.step_id, False))
        if od.get("abstain_recommended"):
            reason = od.get("abstain_reason") or "abstain (no reason given)"
            abstained_steps.append((s.step_id, str(reason), is_opt))
            continue
        for inner_id, inner in od.items():
            if isinstance(inner, dict) and inner.get("abstain_recommended"):
                reason = inner.get("abstain_reason") or "abstain"
                # A macro hop is optional if its outer step is optional;
                # we do not have visibility into the inner sub-workflow's
                # step.optional from here, so we conservatively inherit
                # the outer step's optional flag.
                abstained_steps.append(
                    (f"{s.step_id}/{inner_id}", str(reason), is_opt)
                )

    n_required_abstain = sum(1 for _, _, opt in abstained_steps if not opt)

    output_dump = {
        "workflow_id": workflow_id,
        "workflow_title": workflow.title,
        "duration_ms": execution.duration_ms,
        # Top-level abstain flips ONLY when at least one *required* step
        # abstained (or the workflow engine itself reported an abstain).
        # Optional-step abstains remain enumerated below for transparency.
        "abstain_recommended": execution.abstain_recommended or bool(
            n_required_abstain),
        "abstain_reason": execution.abstain_reason,
        "abstained_steps": [
            {"step_id": sid, "reason": reason, "optional": opt}
            for sid, reason, opt in abstained_steps
        ],
        "steps": [
            {
                "step_id": s.step_id, "specialist": s.specialist,
                "tool_name": s.tool_name, "duration_ms": s.duration_ms,
                "skipped": s.skipped, "error": s.error,
                "output_dump": s.output_dump,
            }
            for s in execution.steps
        ],
    }
    if bundle_provenance:
        # Surface the bundle counts so the upstream chat-LLM (and any
        # downstream auditor) can verify the engine consumed real FHIR
        # resources rather than a fabricated template.
        output_dump["_bundle_provenance"] = bundle_provenance

    abstain_tag = _abstain_tag_from_dump(output_dump)
    return DispatchResult(
        target_kind="workflow", target_label=workflow.title,
        target_slug=workflow_id,
        rationale=(
            f"Recognised the user's prompt as a {workflow.title} request "
            f"and ran all {n_steps} steps ({ok_steps} ok).{abstain_tag}"
        ),
        output=execution,
        output_dump=output_dump,
    )


async def _run_specialist_route(slug: str, route: Route,
                                  metadata: dict | None) -> DispatchResult:
    # Require a real FHIR context. The dispatcher never falls back to
    # hardcoded demo fixtures in route.inputs; without a live bundle the
    # tool would produce output for a fabricated patient.
    if not isinstance(metadata, dict) or not metadata.get("fhir_server_url"):
        return DispatchResult(
            target_kind="specialist",
            target_label=f"{slug} -> {route.skill_label}",
            target_slug=f"{slug}:{route.tool_name}",
            rationale=(
                f"Abstained from consulting {slug} on {route.skill_label}: "
                f"the dispatcher requires a real FHIR context "
                f"(fhir_server_url in message metadata or live SHARP headers). "
                f"No demo fixtures are used as a fallback."
            ),
            error="no_fhir_context: dispatcher requires a real FHIR context "
                  "(chat-mode metadata or live SHARP headers)",
            output_dump={"abstain_recommended": True},
        )
    inputs: dict[str, Any] = {}
    _ALWAYS = ("patient_id", "encounter_id")
    for k, v in metadata.items():
        if v is None:
            continue
        if k in inputs or k in _ALWAYS:
            inputs[k] = v
    # Alias `patient_reference` <- `patient_id` for scribe tools (see
    # comment in _run_workflow).
    if "patient_id" in inputs and "patient_reference" not in inputs:
        inputs["patient_reference"] = inputs["patient_id"]
    # Drop any stale values for FHIR-derived inputs and rebuild from the
    # live bundle. If the fetch fails, abstain.
    bundle_provenance: dict = {}
    ok, reason = await _overlay_real_fhir(
        inputs, metadata, provenance_out=bundle_provenance,
    )
    if not ok:
        return DispatchResult(
            target_kind="specialist",
            target_label=f"{slug} -> {route.skill_label}",
            target_slug=f"{slug}:{route.tool_name}",
            rationale=(
                f"Abstained from consulting {slug} on "
                f"{route.skill_label}: a real FHIR-context was "
                f"advertised but the live bundle could not be "
                f"retrieved. {reason}"
            ),
            error=f"abstain:fhir_unavailable: {reason}",
        )
    from apps.composer.backends import default_backend
    backend = default_backend()
    try:
        output = await backend.call(slug, route.tool_name, route.callable, inputs)
    except TypeError as exc:
        # Specialist routes declare `inputs={}` so the dispatcher's FHIR
        # overlay can fill them; tools whose required parameters are not
        # bundle-derivable (NIHSS item_scores, PGx genotypes, raw_input
        # for preadmit triage, ...) reach the call with the args still
        # missing, and the underlying compute_* function raises a
        # `TypeError: ... missing N required positional argument(s)`. The
        # spec-correct behaviour is `abstain_recommended=True` with a
        # structured reason -- not a hard crash that surfaces a Python
        # traceback to the chat-LLM. We translate the TypeError into a
        # clean abstain so the upstream UI shows the same warning shape
        # used for every other "data not supplied" path.
        msg = str(exc)
        if "missing" in msg and "argument" in msg:
            missing_arg = msg.split("argument")[0].rsplit(":", 1)[-1].strip(
                " ',"
            ) or "required_inputs"
            return DispatchResult(
                target_kind="specialist",
                target_label=f"{slug} -> {route.skill_label}",
                target_slug=f"{slug}:{route.tool_name}",
                rationale=(
                    f"Abstained from consulting {slug} on "
                    f"{route.skill_label}: this tool requires explicit "
                    f"clinician-supplied inputs ({missing_arg}) that "
                    f"are not derivable from the patient's FHIR bundle."
                ),
                output_dump={
                    "abstain_recommended": True,
                    "abstain_reason": f"missing_clinician_supplied_inputs:{msg}",
                },
            )
        return DispatchResult(
            target_kind="specialist",
            target_label=f"{slug} -> {route.skill_label}",
            target_slug=f"{slug}:{route.tool_name}",
            rationale=(
                f"Routed to {slug} ({route.skill_label}) but the call "
                f"raised TypeError."
            ),
            error=f"TypeError: {exc}",
        )
    except Exception as exc:
        return DispatchResult(
            target_kind="specialist",
            target_label=f"{slug} -> {route.skill_label}",
            target_slug=f"{slug}:{route.tool_name}",
            rationale=(
                f"Routed to {slug} ({route.skill_label}) but the call "
                f"raised {type(exc).__name__}."
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
    dump = output.model_dump() if hasattr(output, "model_dump") else output
    summary_bits: list[str] = []
    if isinstance(dump, dict):
        for f in route.summary_fields:
            v = dump.get(f)
            if v is not None:
                summary_bits.append(f"{f}={v!r}")
    summary = "; ".join(summary_bits) or "see attached artifact for details"
    final_dump = dump if isinstance(dump, dict) else None
    if final_dump is not None and bundle_provenance:
        final_dump = {**final_dump, "_bundle_provenance": bundle_provenance}
    abstain_tag = _abstain_tag_from_dump(final_dump)
    return DispatchResult(
        target_kind="specialist",
        target_label=f"{slug} -> {route.skill_label}",
        target_slug=f"{slug}:{route.tool_name}",
        rationale=(
            f"Consulted {slug} for {route.skill_label}: {summary}."
            f"{abstain_tag}"
        ),
        output=output, output_dump=final_dump,
    )


# ------------------------------------------------------------------ public API


def _abstain_tag_from_dump(output_dump: Any) -> str:
    """Build the chat-LLM-visible abstain warning from a result's dump.

    Two abstain shapes are recognised:
    - workflow shape: `output_dump["abstained_steps"]` is a list of
      `{step_id, reason, optional}` (set by `_run_workflow`).
    - single-tool shape: `output_dump["abstain_recommended"]` is True
      and `output_dump["abstain_reason"]` is the explanation (set by
      tools that abstain directly).

    Required-step abstains receive the strong "DO NOT use as final
    estimate" warning; optional-step abstains are surfaced with a
    softer "skipped optional step" wording so the chat-LLM can present
    the workflow as completed-with-a-gap rather than as refused-to-run.

    Returns "" when no abstain is present.
    """
    if not isinstance(output_dump, dict):
        return ""
    steps = output_dump.get("abstained_steps") or []
    if isinstance(steps, list) and steps:
        required_bits: list[str] = []
        optional_bits: list[str] = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("step_id") or "?")
            reason = str(s.get("reason") or "abstain")
            short_reason = reason.split(":", 1)[0]
            if s.get("optional"):
                optional_bits.append(f"{sid} -> {short_reason}")
            else:
                required_bits.append(f"{sid} -> {short_reason}")
        parts: list[str] = []
        if required_bits:
            parts.append(
                f" ABSTAINED on {len(required_bits)} required step(s): "
                + "; ".join(required_bits)
                + ". The numeric outputs of these steps are NOT "
                "calibrated for this patient and MUST NOT be presented "
                "as final estimates."
            )
        if optional_bits:
            parts.append(
                f" Skipped {len(optional_bits)} optional step(s) "
                "for transparency (the workflow can still complete "
                "without them): " + "; ".join(optional_bits) + "."
            )
        return "".join(parts)
    if output_dump.get("abstain_recommended"):
        reason = str(output_dump.get("abstain_reason") or "abstain")
        return (
            f" ABSTAINED: {reason.split(':', 1)[0]}. The numeric output "
            "of this tool is NOT calibrated for this patient and MUST "
            "NOT be presented as a final estimate."
        )
    return ""


async def _run_pick(pick: tuple[str, str, str],
                       metadata: dict | None) -> DispatchResult | None:
    kind, target, rationale = pick
    if kind == "workflow":
        res = await _run_workflow(target, metadata)
    else:
        bits = target.split(":")
        if len(bits) < 2:
            return None
        slug = bits[-2] if len(bits) > 2 else bits[0]
        tool_name = bits[-1]
        route = next(
            (r for r in SPECIALIST_ROUTES.get(slug, [])
             if r.tool_name == tool_name),
            None,
        )
        if route is None:
            return None
        res = await _run_specialist_route(slug, route, metadata)
    if rationale and len(rationale) > 8:
        # Preserve any abstain warning attached by the inner runner so
        # the chat-LLM cannot present an abstained numeric output as a
        # calibrated result. The LLM's pick rationale describes why the
        # target was chosen; the abstain tag describes why the answer
        # is unsafe to use - both must reach the user.
        res.rationale = rationale + _abstain_tag_from_dump(res.output_dump)
    return res


async def dispatch(prompt: str, metadata: dict | None = None) -> DispatchResult:
    """Single entry-point used by the orchestrator's A2A handler.

    Order:
      1. Deterministic keyword router (sub-millisecond). When the prompt
         contains a specific clinical phrase or a demo macro keyword,
         routing is unambiguous and the LLM would add latency without
         improving the pick. The keyword table is precision-tuned for
         the catalog, and longest-match guarantees that the most
         specific workflow wins.
      2. LLM-driven ranker (1 - 3 picks, fan-out merge). Only consulted
         when the keyword router has nothing to say -- which is the
         case for free-form prose that doesn't name a known clinical
         phrase. This is where the LLM's flexibility actually helps.
      3. Discovery fallback when neither layer matches.

    PO's chat client times out external agent responses on a tight
    budget, and the LLM has a 2 - 3 s cold-start floor. Doing the
    cheap path first keeps demo prompts well under that budget while
    preserving LLM intelligence for the long tail of free-form
    requests.
    """
    if not (prompt or "").strip():
        return _discovery()

    import asyncio as _asyncio

    # Layer 1: keyword router (fan-out supported).
    wid = _match_workflow(prompt)
    sm = _match_specialist_route(prompt)
    coros: list = []
    if wid is not None:
        coros.append(_run_workflow(wid, metadata))
    if sm is not None:
        slug, route = sm
        coros.append(_run_specialist_route(slug, route, metadata))
    if len(coros) >= 2:
        sub_results = list(await _asyncio.gather(*coros))
        return _merge_fanout(sub_results)
    if coros:
        return await coros[0]

    # Layer 2: LLM picks 1-3 targets in priority order.
    llm_picks = await _llm_dispatch(prompt)
    if llm_picks:
        sub_results: list[DispatchResult] = []
        for r in await _asyncio.gather(
            *[_run_pick(p, metadata) for p in llm_picks],
            return_exceptions=False,
        ):
            if r is not None:
                sub_results.append(r)
        if len(sub_results) == 1:
            return sub_results[0]
        if sub_results:
            return _merge_fanout(sub_results)

    # Layer 3: discovery.
    return _discovery()


def _merge_fanout(results: list[DispatchResult]) -> DispatchResult:
    """Synthesise a single DispatchResult from a 2-3 wide fan-out."""
    labels = " + ".join(r.target_label for r in results)
    rationales = " // ".join(
        f"{r.target_label}: {r.rationale.rstrip('.')}"
        for r in results
    )
    return DispatchResult(
        target_kind="fanout",
        target_label=labels,
        target_slug=";".join(r.target_slug or "" for r in results),
        rationale=(
            f"Fan-out across {len(results)} capabilities for a richer "
            f"second-opinion answer. {rationales}."
        ),
        output_dump={
            "n_sub_results": len(results),
            "sub_results": [
                {
                    "target_kind": r.target_kind,
                    "target_label": r.target_label,
                    "target_slug": r.target_slug,
                    "rationale": r.rationale,
                    "error": r.error,
                    "output_dump": r.output_dump,
                }
                for r in results
            ],
        },
        sub_results=results,
    )


def _discovery() -> DispatchResult:
    n_workflows = len(WORKFLOW_REGISTRY)
    n_specialists = len(SPECIALIST_ROUTES)
    n_routes = sum(len(v) for v in SPECIALIST_ROUTES.values())
    return DispatchResult(
        target_kind="discovery", target_label="catalog",
        target_slug=None,
        rationale=(
            f"I'm the TrustedRisk Care Engine. I route clinical prompts "
            f"to {n_specialists} sub-agents covering {n_routes} skills, "
            f"or compose them through {n_workflows} care-arc workflows "
            f"(base + macro + payer-/severity-parametric variants). "
            f"Try a clinical scenario (e.g. 'run the readmission risk', "
            f"'sepsis workup', 'denial appeal for biologic') and I'll "
            f"narrate which sub-agent or workflow I'm dispatching to."
        ),
        output_dump={
            "workflows": sorted(WORKFLOW_REGISTRY.keys()),
            "specialists": sorted(SPECIALIST_ROUTES.keys()),
            "n_workflows": n_workflows,
            "n_specialists": n_specialists,
            "n_specialist_routes": n_routes,
            "llm_available": _llm_available(),
        },
    )
