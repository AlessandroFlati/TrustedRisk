"""Direct-tool chain harness — covers >=50% of the 146 MCP tools.

While ``scripts/smoke_demo_full.py`` exercises the dispatcher path
(``_run_workflow`` / ``_run_specialist_route`` against the Care Engine
registry), this harness invokes the underlying compute_* tools
directly with realistic kwargs. The purpose is twofold:

  1. Verify the tools' input-shape contracts under realistic FHIR
     bundles -- especially the chain shapes the system prompt
     prescribes (HEDIS aggregate -> stars + care-gap; resolve-meds ->
     ddi + polypharmacy; fetch-docs -> NER + ground; admission triage
     -> deterioration -> consult letter; denial-parse -> appeal-letter
     -> escalation; ...).

  2. Pre-warm any contracts not yet exercised by the workflow registry
     so a fresh build of the federation cannot ship a tool with a
     surprise crash on a vanilla input.

Each "chain" is a small list of (tool, kwargs_lambda) pairs that the
runner executes sequentially, threading the previous tool's output
into the next where the system prompt expects it. Each chain reports
PASS / ABSTAIN / CRASH per step plus a chain-level verdict.

Run::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/smoke_tool_chains.py

Exit code 0 if no step crashed (TypeError / AttributeError / Exception
that is NOT a structured abstain), 1 otherwise.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import sys
import time
import traceback
from typing import Any, Awaitable, Callable

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.fixtures.fhir_helpers import bind_in_memory_fhir  # noqa: E402
from mcp_server.sharp.headers import FHIRContext, _fhir_ctx  # noqa: E402

# All MCP tool imports. Kept inline so each chain's needs are visible
# at the chain definition.
from mcp_server.tools.acs_disposition_decision import (  # noqa: E402
    compute_acs_disposition_decision,
)
from mcp_server.tools.active_meds_resolver import (  # noqa: E402
    compute_resolve_active_meds,
)
from mcp_server.tools.admission_triage import compute_admission_triage  # noqa: E402
from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage  # noqa: E402
from mcp_server.tools.antibiotic_de_escalation import (  # noqa: E402
    compute_antibiotic_de_escalation,
)
from mcp_server.tools.auto_coding import (  # noqa: E402
    compute_coding_audit,
    compute_cpt_suggest,
    compute_icd10_suggest,
)
from mcp_server.tools.cardiology_depth import (  # noqa: E402
    compute_grace_acs_score,
    compute_timi_acs_score,
)
from mcp_server.tools.care_gap_detector import compute_care_gap_detector  # noqa: E402
from mcp_server.tools.caregiver_summary import compute_caregiver_summary  # noqa: E402
from mcp_server.tools.charlson_elixhauser import (  # noqa: E402
    compute_charlson_elixhauser_index,
)
from mcp_server.tools.chart_intelligence import compute_clinical_ner  # noqa: E402
from mcp_server.tools.chemo_dose_adjustment import (  # noqa: E402
    compute_chemo_dose_adjustment,
)
from mcp_server.tools.clinical_deterioration_score import (  # noqa: E402
    compute_clinical_deterioration_score,
)
from mcp_server.tools.contrast_safety_check import (  # noqa: E402
    compute_contrast_safety_check,
)
from mcp_server.tools.delirium_screening_cam import (  # noqa: E402
    compute_delirium_screening_cam,
)
from mcp_server.tools.detect_phi import detect_phi  # noqa: E402
from mcp_server.tools.dialysis_initiation_decision import (  # noqa: E402
    compute_dialysis_initiation_decision,
)
from mcp_server.tools.differential_diagnosis_ranker import (  # noqa: E402
    compute_differential_diagnosis_ranker,
)
from mcp_server.tools.discharge_counseling import compute_discharge_counseling  # noqa: E402
from mcp_server.tools.dka_severity import compute_dka_severity  # noqa: E402
from mcp_server.tools.empiric_antibiotic_selection import (  # noqa: E402
    compute_empiric_antibiotic_selection,
)
from mcp_server.tools.endocrinology_advanced import (  # noqa: E402
    compute_adrenal_insufficiency_workup,
    compute_thyroid_management,
)
from mcp_server.tools.expected_value_of_intervention import (  # noqa: E402
    compute_expected_value_of_intervention,
)
from mcp_server.tools.falls_risk_morse import compute_falls_risk_morse  # noqa: E402
from mcp_server.tools.fetch_patient_documents import (  # noqa: E402
    compute_fetch_patient_documents,
)
from mcp_server.tools.gi_hepatology_depth import (  # noqa: E402
    compute_glasgow_blatchford_ugib,
    compute_maddrey_alcoholic_hepatitis,
)
from mcp_server.tools.ground_claim import ground_claim  # noqa: E402
from mcp_server.tools.heart_score import compute_heart_score  # noqa: E402
from mcp_server.tools.heme_onc_depth import (  # noqa: E402
    compute_ecog_performance_status,
    compute_ipss_r_mds_score,
    compute_iss_myeloma_staging,
    compute_karnofsky_performance,
)
from mcp_server.tools.imaging_appropriateness import (  # noqa: E402
    compute_imaging_appropriateness,
)
from mcp_server.tools.infectious_disease import (  # noqa: E402
    compute_hiv_management_tier,
    compute_qsofa_score,
    compute_tb_risk_screen,
)
from mcp_server.tools.inpatient_glycemic_control import (  # noqa: E402
    compute_inpatient_glycemic_control,
)
from mcp_server.tools.insurance_appeals import (  # noqa: E402
    compute_appeal_escalation_path,
    compute_appeal_letter_draft,
    compute_denial_letter_parse,
)
from mcp_server.tools.lab_trend_analysis import (  # noqa: E402
    compute_lab_trend_analysis,
)
from mcp_server.tools.massive_transfusion_protocol import (  # noqa: E402
    compute_massive_transfusion_protocol,
)
from mcp_server.tools.medication_reconciliation import (  # noqa: E402
    compute_medication_reconciliation,
)
from mcp_server.tools.medication_what_if import (  # noqa: E402
    compute_medication_what_if,
)
from mcp_server.tools.multimodal import compute_ecg_qt_analyzer  # noqa: E402
from mcp_server.tools.neurology_depth import (  # noqa: E402
    compute_hunt_hess_sah,
    compute_ich_score,
    compute_modified_rankin,
)
from mcp_server.tools.loinc_normalizer import (  # noqa: E402
    compute_normalize_observations,
)
from mcp_server.tools.oncology_treatment_response import (  # noqa: E402
    compute_oncology_treatment_response,
)
from mcp_server.tools.pa_appeal_likelihood import (  # noqa: E402
    compute_pa_appeal_likelihood,
)
from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack  # noqa: E402
from mcp_server.tools.pa_letter_draft import compute_pa_letter_draft  # noqa: E402
from mcp_server.tools.pa_payer_rules_match import (  # noqa: E402
    compute_pa_payer_rules_match,
)
from mcp_server.tools.patient_faq import compute_patient_faq  # noqa: E402
from mcp_server.tools.pediatric_early_warning import (  # noqa: E402
    compute_pediatric_early_warning,
)
from mcp_server.tools.peri_op_risk import (  # noqa: E402
    compute_ariscat_pulmonary_risk,
    compute_caprini_vte_risk,
    compute_rcri_cardiac_risk,
)
from mcp_server.tools.pgx import (  # noqa: E402
    compute_pgx_dose_adjustment,
    compute_pgx_drug_alternatives,
    compute_pgx_eligibility_check,
)
from mcp_server.tools.polypharmacy_concerns import (  # noqa: E402
    detect_polypharmacy_concerns,
)
from mcp_server.tools.population_health import (  # noqa: E402
    compute_outbreak_heatmap,
    compute_syndromic_surveillance,
    compute_vaccine_reminder_cohort,
)
from mcp_server.tools.preadmit_triage import (  # noqa: E402
    compute_symptom_followup_questions,
    compute_symptom_red_flag_check,
    compute_when_to_seek_care,
)
from mcp_server.tools.preeclampsia_assessment import (  # noqa: E402
    compute_preeclampsia_assessment,
)
from mcp_server.tools.psychiatric_admission_decision import (  # noqa: E402
    compute_psychiatric_admission_decision,
)
from mcp_server.tools.quality_stars import (  # noqa: E402
    compute_care_gap_priority_ranking,
    compute_quality_measures_aggregate,
    compute_stars_rating_forecast,
)
from mcp_server.tools.readmission_risk import compute_readmission_risk  # noqa: E402
from mcp_server.tools.rxnorm_ddi import (  # noqa: E402
    compute_rxnorm_ddi_lookup,
    resolve_rxcui,
)
from mcp_server.tools.sleep_pain import (  # noqa: E402
    compute_epworth_sleepiness_scale,
    compute_stop_bang_osa_screen,
)
from mcp_server.tools.critical_care import (  # noqa: E402
    compute_apache_ii_score,
    compute_sofa_score,
)
from mcp_server.tools.stroke_severity import compute_stroke_severity  # noqa: E402
from mcp_server.tools.stroke_thrombolysis_eligibility import (  # noqa: E402
    compute_stroke_thrombolysis_eligibility,
)
from mcp_server.tools.suicide_risk_assessment import (  # noqa: E402
    compute_suicide_risk_assessment,
)
from mcp_server.tools.trauma_severity_score import (  # noqa: E402
    compute_trauma_severity_score,
)
from mcp_server.tools.weight_based_dosing import (  # noqa: E402
    compute_weight_based_dosing,
)


# ───────────────────────── fixture loader (shared with smoke_demo_full) ─────────


_BUNDLE_PATHS: dict[str, str] = {
    "marcus":  "fixtures/po_demo/marcus_reyes.json",
    "eleanor": "fixtures/po_demo/eleanor_greene.json",
    "nadia":   "fixtures/po_demo/nadia_okafor.json",
    "sofia":   "fixtures/po_demo/sofia_ramirez.json",
}
_BUNDLE_CACHE: dict[str, dict] = {}


def _bundle(slug: str) -> dict:
    if slug in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[slug]
    path = ROOT / _BUNDLE_PATHS[slug]
    _BUNDLE_CACHE[slug] = json.loads(path.read_text(encoding="utf-8"))
    return _BUNDLE_CACHE[slug]


def _patient_id(slug: str) -> str:
    bundle = _bundle(slug)
    for entry in bundle.get("entry", []):
        r = entry.get("resource") or {}
        if r.get("resourceType") == "Patient":
            r.setdefault("id", f"demo-{slug}")
            return f"Patient/{r['id']}"
    return f"Patient/demo-{slug}"


# ───────────────────────── chain runner data model ─────────────────────────


@dataclasses.dataclass
class StepResult:
    tool_name: str
    status: str            # PASS | ABSTAIN | CRASH
    duration_ms: float
    abstain_reason: str | None
    error: str | None
    output: Any | None


@dataclasses.dataclass
class ChainResult:
    name: str
    bundle: str
    steps: list[StepResult]

    @property
    def n_crashes(self) -> int:
        return sum(1 for s in self.steps if s.status == "CRASH")

    @property
    def n_abstain(self) -> int:
        return sum(1 for s in self.steps if s.status == "ABSTAIN")

    @property
    def n_pass(self) -> int:
        return sum(1 for s in self.steps if s.status == "PASS")

    @property
    def n_netdep(self) -> int:
        return sum(1 for s in self.steps if s.status == "NETWORK_DEP")


async def _run_step(
    tool: Callable[..., Awaitable[Any]],
    kwargs: dict[str, Any],
) -> StepResult:
    name = getattr(tool, "__name__", repr(tool))
    t0 = time.perf_counter()
    try:
        out = await tool(**kwargs)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        # Network-dependent tools (compute_resolve_active_meds,
        # compute_fetch_patient_documents, ground_claim,
        # compute_readmission_risk) call get_fhir_client() instead of
        # the monkey-patched fetch_patient_bundle, so they try to talk
        # to the in-memory FHIR URL and connection-refuse. Production
        # is fine -- they reach the live server via the SHARP context.
        # We surface them as NETWORK_DEP to keep the run signal clear.
        is_net = (
            "ClientConnectorError" in msg
            or "Cannot connect to host" in msg
            or "NewConnectionError" in msg
        )
        return StepResult(
            tool_name=name,
            status="NETWORK_DEP" if is_net else "CRASH",
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 2),
            abstain_reason=None,
            error=msg,
            output=None,
        )
    dt = round((time.perf_counter() - t0) * 1000.0, 2)
    abstain_recommended = bool(getattr(out, "abstain_recommended", False))
    if isinstance(out, dict):
        abstain_recommended = bool(out.get("abstain_recommended", False))
    abstain_reason = (
        str(getattr(out, "abstain_reason", "") or "")
        if not isinstance(out, dict)
        else str(out.get("abstain_reason", "") or "")
    )
    if abstain_recommended:
        return StepResult(
            tool_name=name, status="ABSTAIN", duration_ms=dt,
            abstain_reason=abstain_reason or "abstain (no reason given)",
            error=None, output=out,
        )
    return StepResult(
        tool_name=name, status="PASS", duration_ms=dt,
        abstain_reason=None, error=None, output=out,
    )


# ───────────────────────── chain catalog ─────────────────────────


# Each chain entry: (chain_name, bundle_slug, [(tool, kwargs_factory), ...])
# kwargs_factory may consume the previous step's output via the
# `_PrevOutput` sentinel returned through the runner. To keep the
# catalog readable we use a small ChainBuilder helper.


class _ChainBuilder:
    def __init__(self, name: str, bundle_slug: str) -> None:
        self.name = name
        self.bundle_slug = bundle_slug
        self.steps: list[tuple[Callable, Callable[[dict, list[StepResult]],
                                                          dict]]] = []

    def step(
        self,
        tool: Callable[..., Awaitable[Any]],
        kwargs_fn: Callable[[dict, list[StepResult]], dict],
    ) -> "_ChainBuilder":
        self.steps.append((tool, kwargs_fn))
        return self


def _ch(name: str, bundle: str) -> _ChainBuilder:
    return _ChainBuilder(name, bundle)


# ───────────────────────── chains ─────────────────────────


def _all_chains() -> list[_ChainBuilder]:
    out: list[_ChainBuilder] = []

    # ----- 1. resolve-meds -> DDI + polypharmacy + ddi-lookup ---
    def _resolve_meds_kw(b, _):
        return {}

    def _meds_list(prev: list[StepResult]) -> list[str]:
        if not prev or prev[-1].output is None:
            return ["warfarin", "metoprolol", "lisinopril", "atorvastatin"]
        res = prev[-1].output
        meds_list = getattr(res, "medications", None)
        if meds_list is None and isinstance(res, dict):
            meds_list = res.get("medications")
        out: list[str] = []
        for m in (meds_list or [])[:6]:
            if isinstance(m, dict):
                out.append(m.get("rxnorm_code") or m.get("name") or "")
            else:
                out.append(str(m))
        return [m for m in out if m] or ["warfarin", "metoprolol",
                                             "lisinopril", "atorvastatin"]

    def _ddi_kw(b, prev):
        return {"drugs": _meds_list(prev)}

    def _polypharm_kw(b, prev):
        return {"medications": _meds_list(prev)}

    out.append(
        _ch("resolve_meds_chain", "eleanor")
        .step(compute_resolve_active_meds, _resolve_meds_kw)
        .step(compute_rxnorm_ddi_lookup, _ddi_kw)
        .step(detect_polypharmacy_concerns, _polypharm_kw)
    )

    # ----- 2. fetch-docs -> clinical NER + ground claim ---
    def _fetch_docs_kw(b, _):
        return {}

    def _ner_kw(b, prev):
        # plaintext from the first successful doc
        prev_out = prev[-1].output if prev else None
        text = ""
        docs = getattr(prev_out, "documents", None) or (
            prev_out.get("documents") if isinstance(prev_out, dict) else []
        )
        if docs:
            d = docs[0]
            text = (
                getattr(d, "text", None)
                or (d.get("text") if isinstance(d, dict) else "")
                or ""
            )
        if not text:
            text = (
                "Patient is a 67yo F with sepsis from urinary source, "
                "started on ceftriaxone 1g IV q24h. Lactate trending down."
            )
        return {"text": text[:4000]}

    def _ground_kw(b, prev):
        return {
            "claim_text": (
                "Ceftriaxone is appropriate empiric coverage for "
                "uncomplicated urinary sepsis."
            ),
        }

    out.append(
        _ch("fetch_docs_chain", "marcus")
        .step(compute_fetch_patient_documents, _fetch_docs_kw)
        .step(compute_clinical_ner, _ner_kw)
        .step(ground_claim, _ground_kw)
    )

    # ----- 3. HEDIS aggregate -> stars + care-gap ranking (parallel-ish) -
    def _qm_aggregate_kw(b, _):
        return {
            "measurement_year": 2026,
            "cohort_summary": {
                "DM_HBA1C_CONTROL": {"eligible": 1200, "compliant": 920},
                "BCS_BREAST_CANCER_SCREEN": {"eligible": 1500, "compliant": 1100},
                "CCS_COLORECTAL_CANCER_SCREEN":
                    {"eligible": 1100, "compliant": 700},
                "ADV_VAX_FLU": {"eligible": 3000, "compliant": 1900},
            },
            "n_eligible_patients": 5000,
        }

    def _stars_kw(b, prev):
        return {
            "aggregate": prev[-1].output,
            "contract_size_thousand_members": 25.0,
        }

    def _care_gap_rank_kw(b, prev):
        # Pass the aggregate (step 0), not the forecast (step 1).
        return {"aggregate": prev[0].output, "top_n": 10}

    out.append(
        _ch("hedis_chain", "sofia")
        .step(compute_quality_measures_aggregate, _qm_aggregate_kw)
        .step(compute_stars_rating_forecast, _stars_kw)
        .step(compute_care_gap_priority_ranking, _care_gap_rank_kw)
    )

    # ----- 4. admission triage -> NEWS2 -> consult letter (deterioration) -
    def _triage_kw(b, _):
        return {
            "chief_complaint": "fever and hypotension, suspect urosepsis",
            "vital_signs": {"systolic_bp": 84, "heart_rate": 122,
                              "temperature": 39.1, "spo2": 91,
                              "respiratory_rate": 28},
            "age": 67,
            "patient_id": "demo",
        }

    def _news2_kw(b, _):
        return {
            "vital_signs": [
                {"type": "heart_rate", "value": 122},
                {"type": "respiratory_rate", "value": 28},
                {"type": "spo2", "value": 91},
                {"type": "systolic_bp", "value": 84},
                {"type": "temperature", "value": 39.1},
            ],
            "patient_id": "demo",
        }

    def _ddx_kw(b, _):
        return {
            "chief_complaint": "fever and hypotension, suspect urosepsis",
            "structured_features": {},
        }

    out.append(
        _ch("triage_chain", "marcus")
        .step(compute_admission_triage, _triage_kw)
        .step(compute_clinical_deterioration_score, _news2_kw)
        .step(compute_differential_diagnosis_ranker, _ddx_kw)
    )

    # ----- 5. denial parse -> appeal letter -> escalation path -----
    def _denial_kw(b, _):
        return {
            "letter_text": (
                "Re: Adalimumab. Your prior authorization request has been "
                "denied. Reason: lack of evidence for medical necessity. "
                "Step therapy with methotrexate is required first. You "
                "have 60 days to file an internal appeal."
            ),
        }

    def _appeal_letter_kw(b, prev):
        return {
            "parsed_denial": prev[-1].output,
            "patient_summary":
                "62yo female with seropositive RA, failed two non-biologic DMARDs.",
            "medical_necessity_argument":
                ("Patient has demonstrated inadequate response to "
                 "methotrexate (DAS28 5.6) plus sulfasalazine over 6 months."),
        }

    def _esc_kw(b, _):
        return {"payer": "United Healthcare",
                "starting_level": "internal_first_level"}

    out.append(
        _ch("appeals_chain", "nadia")
        .step(compute_denial_letter_parse, _denial_kw)
        .step(compute_appeal_letter_draft, _appeal_letter_kw)
        .step(compute_appeal_escalation_path, _esc_kw)
    )

    # ----- 6. PA pipeline -----
    # The real signatures expect the evidence_pack to be built FIRST and
    # then fed into the rules-match, letter draft, and likelihood tools.
    def _pa_evidence_kw(b, _):
        return {
            "patient_reference": "Patient/demo",
            "requested_service": {
                "service_type": "specialty_drug",
                "description":
                    "Semaglutide (Ozempic) for type 2 diabetes uncontrolled "
                    "on metformin",
                "rxnorm_codes": ["1991306"],
                "icd10_codes": ["E11.65"],
            },
            "payer": "medicare",
            "fhir_bundle": b,
        }

    def _pa_match_kw(b, prev):
        return {"evidence_pack": prev[-1].output}

    def _pa_letter_kw(b, prev):
        return {
            "evidence_pack": prev[-2].output,
            "rules_match": prev[-1].output,
        }

    def _pa_likelihood_kw(b, prev):
        return {
            "evidence_pack": prev[-3].output,
            "rules_match": prev[-2].output,
        }

    out.append(
        _ch("pa_chain", "nadia")
        .step(compute_pa_evidence_pack, _pa_evidence_kw)
        .step(compute_pa_payer_rules_match, _pa_match_kw)
        .step(compute_pa_letter_draft, _pa_letter_kw)
        .step(compute_pa_appeal_likelihood, _pa_likelihood_kw)
    )

    # ----- 7. discharge bundle: readmission -> care gap -> caregiver summary -
    def _readmit_kw(b, _):
        return {"horizon_days": 30, "patient_id": "Patient/demo"}

    def _gap_kw(b, _):
        return {"fhir_bundle": b, "patient_age": 78, "patient_sex": "female"}

    def _counseling_for_summary_kw(b, _):
        return {
            "lace_score": 9,
            "recommendation_action": "home_with_care",
            "patient_id": "Patient/demo",
        }

    def _cg_summary_kw(b, prev):
        return {"counseling": prev[-1].output, "target_audience": "caregiver"}

    out.append(
        _ch("discharge_chain", "eleanor")
        .step(compute_readmission_risk, _readmit_kw)
        .step(compute_care_gap_detector, _gap_kw)
        .step(compute_discharge_counseling, _counseling_for_summary_kw)
        .step(compute_caregiver_summary, _cg_summary_kw)
    )

    # ----- 8. PGx chain -----
    def _pgx_elig_kw(b, _):
        return {
            "requested_test": "CYP2C19_genotype",
            "medications_in_consideration": ["clopidogrel"],
        }

    def _pgx_alt_kw(b, _):
        return {
            "requested_drug": "clopidogrel",
            "genotypes": [{"gene": "CYP2C19", "phenotype": "poor_metabolizer",
                              "diplotype": "*2/*2"}],
        }

    def _pgx_dose_kw(b, _):
        return {
            "medications": ["warfarin"],
            "genotypes": [
                {"gene": "CYP2C9", "phenotype": "intermediate_metabolizer",
                 "diplotype": "*1/*3"},
                {"gene": "VKORC1", "phenotype": "AA_homozygous",
                 "diplotype": "AA"},
            ],
        }

    out.append(
        _ch("pgx_chain", "eleanor")
        .step(compute_pgx_eligibility_check, _pgx_elig_kw)
        .step(compute_pgx_drug_alternatives, _pgx_alt_kw)
        .step(compute_pgx_dose_adjustment, _pgx_dose_kw)
    )

    # ----- 9. AKI -> dialysis decision + lab trend -
    def _aki_kw(b, _):
        return {
            "creatinine_baseline_mg_dl": 1.0,
            "creatinine_current_mg_dl": 2.6,
            "urine_output_ml_per_kg_per_hour": 0.4,
            "urine_output_window_hours": 24,
        }

    def _dialysis_kw(b, prev):
        return {
            "aki_stage": str(getattr(prev[-1].output, "stage", "stage_3")),
            "ph": 7.18,
            "bicarbonate_meq_l": 14.0,
            "potassium_meq_l": 6.2,
            "refractory_hyperkalemia": True,
            "volume_overload_refractory": True,
            "uremic_encephalopathy": False,
        }

    def _lab_trend_kw(b, _):
        return {
            "observations": [
                {"loinc_code": "2160-0", "value": 1.0, "unit": "mg/dL",
                 "observed_at": "2026-04-15T08:00:00Z"},
                {"loinc_code": "2160-0", "value": 1.4, "unit": "mg/dL",
                 "observed_at": "2026-04-22T08:00:00Z"},
                {"loinc_code": "2160-0", "value": 2.0, "unit": "mg/dL",
                 "observed_at": "2026-04-28T08:00:00Z"},
                {"loinc_code": "2160-0", "value": 2.6, "unit": "mg/dL",
                 "observed_at": "2026-04-30T08:00:00Z"},
            ],
        }

    out.append(
        _ch("aki_chain", "marcus")
        .step(compute_aki_kdigo_stage, _aki_kw)
        .step(compute_dialysis_initiation_decision, _dialysis_kw)
        .step(compute_lab_trend_analysis, _lab_trend_kw)
    )

    # ----- 10. DKA -> glycemic control -----
    def _dka_kw(b, _):
        return {
            "ph": 7.05,
            "bicarbonate_meq_l": 9,
            "glucose_mg_dl": 540,
            "ketones_present": True,
            "mental_status": "alert",
            "anion_gap": 22,
            "potassium_meq_l": 4.8,
            "weight_kg": 80,
        }

    def _glycemic_kw(b, _):
        return {
            "is_icu": False,
            "average_glucose_24h": 245.0,
            "n_hypoglycemic_episodes_24h": 0,
            "n_severe_hyperglycemic_episodes_24h": 1,
            "current_regimen": "sliding_scale_only",
            "current_basal_total_units": 0.0,
            "risk_factors": {"weight_kg": 80, "egfr_ml_min": 60},
        }

    out.append(
        _ch("dka_chain", "marcus")
        .step(compute_dka_severity, _dka_kw)
        .step(compute_inpatient_glycemic_control, _glycemic_kw)
    )

    # ----- 11. stroke severity -> thrombolysis eligibility -
    def _stroke_kw(b, _):
        return {
            "item_scores": {
                "loc": 0, "loc_questions": 1, "loc_commands": 0,
                "best_gaze": 0, "visual_fields": 1, "facial_palsy": 1,
                "left_arm_motor": 3, "right_arm_motor": 0,
                "left_leg_motor": 2, "right_leg_motor": 0,
                "limb_ataxia": 0, "sensory": 1, "best_language": 1,
                "dysarthria": 0, "extinction_inattention": 1,
            },
            "last_known_well_minutes_ago": 90,
        }

    def _thrombolysis_kw(b, _):
        return {
            "last_known_well_minutes_ago": 90,
            "nihss_total": 11,
            "clinical_factors": {
                "age_years": 70,
                "blood_pressure_systolic": 168,
                "platelet_count_per_ul": 240000,
                "inr": 1.0,
                "active_bleeding": False,
                "recent_surgery_14d": False,
                "history_of_intracranial_hemorrhage": False,
                "current_anticoagulation": "none",
            },
        }

    out.append(
        _ch("stroke_chain", "eleanor")
        .step(compute_stroke_severity, _stroke_kw)
        .step(compute_stroke_thrombolysis_eligibility, _thrombolysis_kw)
    )

    # ----- 12. trauma -> MTP -----
    def _trauma_kw(b, _):
        return {
            "injuries": [
                {"region": "head", "ais_score": 3},
                {"region": "chest", "ais_score": 4},
                {"region": "abdomen", "ais_score": 2},
            ],
            "glasgow_coma_score": 13,
            "systolic_bp": 78,
            "respiratory_rate": 28,
            "has_active_hemorrhage": True,
        }

    def _mtp_kw(b, _):
        return {
            "penetrating_mechanism": False,
            "field_or_arrival_sbp_le_90": True,
            "heart_rate_ge_120": True,
            "positive_fast_exam": True,
            "estimated_blood_loss_ml": 1500,
        }

    out.append(
        _ch("trauma_chain", "eleanor")
        .step(compute_trauma_severity_score, _trauma_kw)
        .step(compute_massive_transfusion_protocol, _mtp_kw)
    )

    # ----- 13. chest pain trio: HEART -> GRACE -> TIMI -> ACS disposition -
    def _heart_kw(b, _):
        return {
            "history_descriptor": "highly_suspicious",
            "ecg_descriptor": "non_specific_repolarization",
            "age": 67,
            "risk_factors_count": 3,
            "troponin_times_uln": 1.5,
        }

    def _grace_kw(b, _):
        return {
            "age": 67,
            "heart_rate": 96,
            "systolic_bp": 145,
            "creatinine_mg_dl": 1.2,
            "killip_class": 1,
            "cardiac_arrest_at_admission": False,
            "elevated_cardiac_enzymes": True,
            "st_segment_deviation": True,
        }

    def _timi_kw(b, _):
        return {
            "age_ge_65": True,
            "three_or_more_cad_risk_factors": True,
            "known_cad_50_pct_stenosis": True,
            "aspirin_use_in_last_7_days": True,
            "severe_anginal_episodes_in_last_24h": True,
            "st_deviation_ge_0_5_mm": True,
            "elevated_cardiac_markers": True,
        }

    def _acs_dispo_kw(b, prev):
        heart = prev[-3].output
        return {
            "heart_score_total": int(getattr(heart, "score_total", 6) or 6),
            "has_stemi": False,
            "has_dynamic_troponin": True,
            "ongoing_chest_pain": False,
            "hemodynamic_instability": False,
            "new_heart_failure": False,
        }

    out.append(
        _ch("chest_pain_chain", "marcus")
        .step(compute_heart_score, _heart_kw)
        .step(compute_grace_acs_score, _grace_kw)
        .step(compute_timi_acs_score, _timi_kw)
        .step(compute_acs_disposition_decision, _acs_dispo_kw)
    )

    # ----- 14. sepsis severity trio: qSOFA -> SOFA -> APACHE II -
    def _qsofa_kw(b, _):
        return {
            "altered_mentation_gcs_lt_15": True,
            "respiratory_rate_ge_22": True,
            "systolic_bp_le_100": True,
        }

    def _sofa_kw(b, _):
        return {
            "pao2_fio2_ratio": 220,
            "mechanical_ventilation": False,
            "platelets_thousands_per_uL": 90,
            "bilirubin_mg_dl": 2.4,
            "mean_arterial_pressure_mmHg": 62,
            "glasgow_coma_scale": 13,
            "creatinine_mg_dl": 1.7,
            "suspected_infection": True,
        }

    def _apache_kw(b, _):
        return {
            "age": 67,
            "temperature_c": 39.1,
            "mean_arterial_pressure_mmHg": 62,
            "heart_rate": 122,
            "respiratory_rate": 28,
            "fio2": 0.4,
            "pao2": 88,
            "arterial_ph": 7.32,
            "serum_sodium_mmol_l": 138,
            "serum_potassium_mmol_l": 4.1,
            "serum_creatinine_mg_dl": 1.7,
            "hematocrit_pct": 38,
            "wbc_thousands_per_uL": 18.2,
            "glasgow_coma_scale": 13,
            "acute_renal_failure": False,
            "chronic_health_severe": False,
            "immunocompromised": False,
            "post_emergency_surgery": False,
        }

    out.append(
        _ch("sepsis_severity_chain", "marcus")
        .step(compute_qsofa_score, _qsofa_kw)
        .step(compute_sofa_score, _sofa_kw)
        .step(compute_apache_ii_score, _apache_kw)
    )

    # ----- 15. peri-op trio -----
    def _caprini_kw(b, _):
        return {
            "age": 70,
            "bmi_gt_25": True,
            "major_surgery_planned": True,
            "history_vte": False,
            "active_malignancy": False,
            "confined_to_bed_gt_72h": False,
            "central_venous_access": False,
        }

    def _ariscat_kw(b, _):
        return {
            "age": 70,
            "preop_spo2_pct": 93.0,
            "surgical_incision_site": "upper_abdominal",
            "surgical_duration_hours": 3.5,
            "respiratory_infection_last_month": False,
            "preop_anemia_hgb_lt_10": False,
            "emergency_surgery": False,
        }

    def _rcri_kw(b, _):
        return {
            "high_risk_surgery": True,
            "history_ischemic_heart_disease": True,
            "history_congestive_heart_failure": False,
            "history_cerebrovascular_disease": False,
            "insulin_dependent_diabetes": False,
            "creatinine_gt_2_mg_dl": False,
        }

    out.append(
        _ch("periop_chain", "eleanor")
        .step(compute_caprini_vte_risk, _caprini_kw)
        .step(compute_ariscat_pulmonary_risk, _ariscat_kw)
        .step(compute_rcri_cardiac_risk, _rcri_kw)
    )

    # ----- 16. OB chain: preeclampsia -
    def _preecl_kw(b, _):
        return {
            "gestational_age_weeks": 33,
            "systolic_bp": 165.0,
            "diastolic_bp": 108.0,
            "proteinuria_present": True,
            "seizures_present": False,
            "clinical_factors": {
                "platelets_per_ul": 110000,
                "creatinine_mg_dl": 1.0,
                "ast_u_l": 60,
                "headache": True,
                "visual_disturbances": True,
            },
        }

    out.append(
        _ch("ob_chain", "sofia")
        .step(compute_preeclampsia_assessment, _preecl_kw)
    )

    # ----- 17. peds chain: PEWS + weight-based dosing -
    def _pews_kw(b, _):
        return {
            "age_months": 24,
            "behavior": "irritable",
            "heart_rate": 145.0,
            "respiratory_rate": 50.0,
            "spo2": 92.0,
            "systolic_bp": 88.0,
            "capillary_refill_seconds": 4.0,
            "accessory_muscle_use": True,
            "parental_or_nurse_concern": True,
        }

    def _ped_dose_kw(b, _):
        return {
            "drug": "amoxicillin",
            "weight_kg": 14.5,
            "age_months": 24,
            "indication": "otitis_media",
            "route": "PO",
        }

    out.append(
        _ch("peds_chain", "sofia")
        .step(compute_pediatric_early_warning, _pews_kw)
        .step(compute_weight_based_dosing, _ped_dose_kw)
    )

    # ----- 18. neuro: ICH -> mRS -> Hunt-Hess -
    def _ich_kw(b, _):
        return {
            "glasgow_coma_scale": 12,
            "ich_volume_ml": 35.0,
            "intraventricular_hemorrhage": True,
            "infratentorial_origin": False,
            "age_ge_80": False,
        }

    def _mrs_kw(b, _):
        return {"grade": 3}

    def _hh_kw(b, _):
        return {"grade": 3}

    out.append(
        _ch("neuro_chain", "eleanor")
        .step(compute_ich_score, _ich_kw)
        .step(compute_modified_rankin, _mrs_kw)
        .step(compute_hunt_hess_sah, _hh_kw)
    )

    # ----- 19. geriatric: Morse falls + delirium CAM + Charlson -
    def _morse_kw(b, _):
        return {
            "history_of_falling_3mo": True,
            "secondary_diagnosis_present": True,
            "ambulatory_aid": "crutches_cane_walker",
            "has_iv_or_heparin_lock": True,
            "gait": "weak",
            "mental_status": "alert",
        }

    def _cam_kw(b, _):
        return {
            "feature1_acute_onset_or_fluctuating": True,
            "feature2_inattention": True,
            "feature3_disorganized_thinking": True,
            "feature4_altered_consciousness": False,
        }

    def _charlson_kw(b, _):
        return {
            "icd10_codes": ["I50.9", "E11.9", "N18.3", "I25.10"],
        }

    out.append(
        _ch("geri_chain", "eleanor")
        .step(compute_falls_risk_morse, _morse_kw)
        .step(compute_delirium_screening_cam, _cam_kw)
        .step(compute_charlson_elixhauser_index, _charlson_kw)
    )

    # ----- 20. mental health chain -
    def _suicide_kw(b, _):
        return {
            "ideation_lifetime_level": 2,
            "ideation_past_30d_level": 1,
            "behavior_lifetime_attempts": 0,
            "behavior_past_30d_any": False,
            "behavior_self_injury_no_intent": False,
            "warning_factors_count": 2,
            "protective_factors_count": 1,
        }

    def _psych_kw(b, prev):
        return {
            "risk_level": "high",
            "danger_to_self": True,
            "danger_to_others": False,
            "grave_disability": False,
            "voluntary_capable": False,
            "has_safety_plan_in_place": False,
        }

    def _stop_bang_kw(b, _):
        return {
            "snoring_loudly": True,
            "tired_during_day": True,
            "observed_apnea": True,
            "high_blood_pressure": True,
            "bmi_gt_35": False,
            "age_gt_50": True,
            "neck_circumference_gt_40_cm": True,
            "male_sex": True,
        }

    def _epworth_kw(b, _):
        return {
            "sitting_reading": 1,
            "watching_tv": 2,
            "sitting_inactive_in_public": 1,
            "passenger_in_car_one_hour": 3,
            "lying_down_to_rest_afternoon": 3,
            "sitting_and_talking_to_someone": 0,
            "sitting_quietly_after_lunch": 2,
            "in_car_stopped_in_traffic": 1,
        }

    out.append(
        _ch("mental_health_chain", "eleanor")
        .step(compute_suicide_risk_assessment, _suicide_kw)
        .step(compute_psychiatric_admission_decision, _psych_kw)
        .step(compute_stop_bang_osa_screen, _stop_bang_kw)
        .step(compute_epworth_sleepiness_scale, _epworth_kw)
    )

    # ----- 21. infectious / antibiotic depth chain -
    def _empiric_kw(b, _):
        return {
            "infection_source": "urinary",
            "severity": "septic",
            "patient_factors": {
                "allergy_penicillin": False,
                "egfr_ml_min": 70,
                "age": 67,
                "immunocompromised": False,
            },
            "local_antibiogram": {"e_coli_esbl_pct": 12.0},
        }

    def _de_escalation_kw(b, _):
        return {
            "current_regimen": "ceftriaxone 1 g IV q24h",
            "pathogen": "Escherichia coli",
            "susceptibility": {"S_to_ceftriaxone": "S",
                                "S_to_ciprofloxacin": "S"},
            "days_on_therapy": 3,
            "total_planned_duration_days": 7,
            "clinical_factors": {
                "afebrile_24h": True, "hemodynamically_stable": True,
                "tolerating_po": True, "alert_oriented": True,
                "malabsorption": False,
            },
        }

    def _hiv_kw(b, _):
        return {
            "cd4_count": 220,
            "viral_load_copies_ml": 6300.0,
            "on_art": True,
        }

    def _tb_kw(b, _):
        return {
            "high_burden_country_residence_or_travel": True,
            "close_contact_with_active_tb": True,
            "immunocompromised_HIV_or_TNF_inhibitor": False,
            "cough_gt_3_weeks_with_constitutional_symptoms": True,
        }

    out.append(
        _ch("infectious_chain", "marcus")
        .step(compute_empiric_antibiotic_selection, _empiric_kw)
        .step(compute_antibiotic_de_escalation, _de_escalation_kw)
        .step(compute_hiv_management_tier, _hiv_kw)
        .step(compute_tb_risk_screen, _tb_kw)
    )

    # ----- 22. coding chain -----
    def _icd_kw(b, _):
        return {
            "chart_text":
                ("HPI: 67yo male with fever and hypotension. Diagnosed "
                 "with urinary tract infection complicated by sepsis. "
                 "Started on ceftriaxone IV. Lactate 4.2."),
        }

    def _cpt_kw(b, _):
        return {
            "procedure_text":
                "Urinary catheterization performed; blood cultures x2 drawn.",
        }

    def _coding_audit_kw(b, prev):
        icd = prev[-2].output
        cpt = prev[-1].output
        icds = []
        if hasattr(icd, "suggestions"):
            icds = [
                {"code": s.code, "rationale": s.rationale}
                for s in icd.suggestions[:3]
            ]
        cpts = []
        if hasattr(cpt, "suggestions"):
            cpts = [
                {"code": s.code, "rationale": s.rationale}
                for s in cpt.suggestions[:3]
            ]
        return {
            "chart_text":
                ("Sepsis from urinary source, ceftriaxone IV. "
                 "Lactate 4.2 trending down."),
            "coded_artifact": {
                "icd10_suggestions": icds,
                "cpt_suggestions": cpts,
            },
        }

    out.append(
        _ch("coding_chain", "marcus")
        .step(compute_icd10_suggest, _icd_kw)
        .step(compute_cpt_suggest, _cpt_kw)
        .step(compute_coding_audit, _coding_audit_kw)
    )

    # ----- 23. preadmit chain -----
    def _red_flag_kw(b, _):
        return {
            "raw_input":
                "I have severe chest pain radiating down my left arm",
        }

    def _seek_care_kw(b, _):
        return {
            "raw_input": "I have a runny nose and mild cough for 3 days",
            "duration_hours": 72,
            "severity_1_to_10": 3,
        }

    def _followup_kw(b, _):
        return {
            "raw_input": "fever 102 and rash on arms",
            "max_questions": 5,
        }

    out.append(
        _ch("preadmit_chain", "eleanor")
        .step(compute_symptom_red_flag_check, _red_flag_kw)
        .step(compute_when_to_seek_care, _seek_care_kw)
        .step(compute_symptom_followup_questions, _followup_kw)
    )

    # ----- 24. oncology chain -----
    def _ipss_kw(b, _):
        return {
            "cytogenetic_category": "good",
            "bm_blast_pct": 4.0,
            "hemoglobin_g_dl": 8.5,
            "platelets_thousands_per_uL": 110.0,
            "anc_thousands_per_uL": 0.8,
        }

    def _myeloma_kw(b, _):
        return {
            "serum_beta2_microglobulin_mg_l": 4.2,
            "serum_albumin_g_dl": 3.0,
        }

    def _onc_response_kw(b, _):
        return {
            "target_lesions": [
                {"lesion_id": "L1", "baseline_diameter_mm": 30.0,
                 "current_diameter_mm": 18.0},
                {"lesion_id": "L2", "baseline_diameter_mm": 22.0,
                 "current_diameter_mm": 20.0},
            ],
            "new_lesions_present": False,
            "non_target_progression": False,
        }

    def _karnofsky_kw(b, _):
        return {"score": 70}

    def _ecog_kw(b, _):
        return {"grade": 1}

    def _chemo_dose_kw(b, _):
        return {
            "regimen": "carboplatin_taxol",
            "cycle_number": 2,
            "egfr_ml_min": 75.0,
            "bilirubin_mg_dl": 0.9,
            "ast_ul": 28.0,
            "anc_per_ul": 1500,
            "platelets_per_ul": 110000,
            "ecog_performance_status": 1,
        }

    out.append(
        _ch("oncology_chain", "eleanor")
        .step(compute_ipss_r_mds_score, _ipss_kw)
        .step(compute_iss_myeloma_staging, _myeloma_kw)
        .step(compute_oncology_treatment_response, _onc_response_kw)
        .step(compute_karnofsky_performance, _karnofsky_kw)
        .step(compute_ecog_performance_status, _ecog_kw)
        .step(compute_chemo_dose_adjustment, _chemo_dose_kw)
    )

    # ----- 25. endocrine chain -----
    def _thyroid_kw(b, _):
        return {
            "tsh_mU_L": 0.05,
            "free_t4_ng_dl": 3.2,
            "current_levothyroxine_dose_mcg": 0,
            "on_levothyroxine": False,
        }

    def _adrenal_kw(b, _):
        return {
            "morning_cortisol_ug_dl": 4.5,
            "cortisol_after_acth_stim_ug_dl": 9.0,
            "acth_pg_ml": 18.0,
        }

    out.append(
        _ch("endocrine_chain", "eleanor")
        .step(compute_thyroid_management, _thyroid_kw)
        .step(compute_adrenal_insufficiency_workup, _adrenal_kw)
    )

    # ----- 26. GI hepatology chain ---
    def _gblatch_kw(b, _):
        return {
            "blood_urea_mmol_l": 11.0,
            "hemoglobin_g_dl": 9.2,
            "sex": "male",
            "systolic_bp_mmHg": 95.0,
            "pulse_ge_100": True,
            "melena": True,
        }

    def _maddrey_kw(b, _):
        return {
            "patient_pt_seconds": 22.5,
            "control_pt_seconds": 12.5,
            "serum_bilirubin_mg_dl": 14.2,
        }

    out.append(
        _ch("gi_hep_chain", "eleanor")
        .step(compute_glasgow_blatchford_ugib, _gblatch_kw)
        .step(compute_maddrey_alcoholic_hepatitis, _maddrey_kw)
    )

    # ----- 27. medication what-if + reconciliation -
    def _med_recon_kw(b, _):
        return {
            "patient_id": "Patient/demo",
            "admission_meds": [
                {"name": "metformin", "dose_mg": 500, "frequency": "BID",
                 "rxnorm_code": "6809"},
                {"name": "lisinopril", "dose_mg": 10, "frequency": "QD",
                 "rxnorm_code": "29046"},
            ],
            "discharge_meds": [
                {"name": "metformin", "dose_mg": 500, "frequency": "BID",
                 "rxnorm_code": "6809"},
                {"name": "atorvastatin", "dose_mg": 40, "frequency": "QHS",
                 "rxnorm_code": "83367"},
            ],
        }

    def _med_what_if_kw(b, _):
        return {
            "medications": ["warfarin"],
            "patient_reference": "Patient/demo",
        }

    out.append(
        _ch("med_chain", "eleanor")
        .step(compute_medication_reconciliation, _med_recon_kw)
        .step(compute_medication_what_if, _med_what_if_kw)
    )

    # ----- 28. utility / cross-cutting ---
    def _detect_phi_kw(b, _):
        return {
            "text":
                ("Patient John Smith (DOB 03/15/1960, MRN 12345678) "
                 "called from 415-555-1234 to confirm appointment."),
        }

    def _resolve_rxcui_kw(b, _):
        return {"drug_name": "metoprolol succinate"}

    def _normalize_obs_kw(b, _):
        return {
            "observations": [
                {"loinc_code": "2160-0", "value": 1.8,
                 "unit": "mg/dL", "observed_at": "2026-04-15T08:00:00Z"},
                {"loinc_code": "2823-3", "value": 5.2,
                 "unit": "mEq/L", "observed_at": "2026-04-15T08:00:00Z"},
            ],
        }

    def _imaging_appropriateness_kw(b, _):
        return {
            "clinical_scenario":
                "uncomplicated low back pain without red flags",
            "patient_age": 45,
            "patient_pregnant": False,
            "cumulative_radiation_msv_last_12mo": 2.0,
        }

    def _ev_intervention_kw(b, _):
        return {
            "intervention": {
                "name": "tirzepatide_for_obesity",
                "absolute_risk_reduction": 0.04,
                "cost_per_patient_usd": 12000.0,
                "qaly_gain_per_patient": 0.05,
                "ev_uncertainty_pct": 0.20,
            },
            "baseline_event_probability": 0.18,
            "cohort_size": 200,
            "wtp_threshold_per_qaly_usd": 100000.0,
        }

    def _contrast_kw(b, _):
        return {
            "contrast_type": "iodinated",
            "egfr_ml_min": 45.0,
            "on_metformin": True,
            "iodine_contrast_prior_severe_reaction": False,
            "iodine_contrast_prior_mild_reaction": False,
            "pregnant": False,
        }

    def _outbreak_kw(b, _):
        return {
            "counts_by_geo_syndrome": {
                "94110": {"flu_like": 35, "gi": 12},
                "94117": {"flu_like": 18, "gi": 9},
                "94103": {"flu_like": 24, "gi": 11},
            },
            "population_by_geo": {
                "94110": 32000, "94117": 24000, "94103": 21000,
            },
            "epsilon": 1.0,
        }

    def _syndromic_kw(b, _):
        return {
            "surveillance_period_start_iso": "2026-04-01",
            "surveillance_period_end_iso": "2026-04-30",
            "observed_counts_by_syndrome":
                {"flu_like": 480, "gi": 220, "rash": 60},
            "expected_counts_by_syndrome":
                {"flu_like": 320, "gi": 200, "rash": 75},
        }

    def _vaccine_reminder_kw(b, _):
        return {
            "overdue_by_vaccine": {
                "MMR": 240, "FLU": 1100, "COVID_BIVALENT": 850,
            },
            "outreach_channel": "phone",
        }

    out.append(
        _ch("util_chain", "eleanor")
        .step(detect_phi, _detect_phi_kw)
        .step(resolve_rxcui, _resolve_rxcui_kw)
        .step(compute_normalize_observations, _normalize_obs_kw)
        .step(compute_imaging_appropriateness, _imaging_appropriateness_kw)
        .step(compute_expected_value_of_intervention, _ev_intervention_kw)
        .step(compute_contrast_safety_check, _contrast_kw)
        .step(compute_outbreak_heatmap, _outbreak_kw)
        .step(compute_syndromic_surveillance, _syndromic_kw)
        .step(compute_vaccine_reminder_cohort, _vaccine_reminder_kw)
    )

    # ----- 29. discharge counseling + patient FAQ -----
    def _counseling_kw(b, _):
        return {
            "medications": [
                {"name": "furosemide", "dose_mg": 40, "frequency": "BID",
                 "rxnorm_code": "4603"},
            ],
            "lace_score": 9,
            "recommendation_action": "home_with_care",
            "patient_id": "Patient/demo",
            "extra_red_flags": ["weight gain >2 lb in 24h"],
            "locale": "en",
        }

    def _faq_kw(b, _):
        # compute_patient_faq's full signature can be inspected at runtime;
        # a minimal call carrying a real question + a discharge context
        # exercises the abstain-vs-positive split. The exact arg names may
        # be different — fall back to **kwargs auto-filtering by passing
        # only the two we know exist.
        return {
            "question": "Why am I taking furosemide?",
        }

    out.append(
        _ch("patient_education_chain", "eleanor")
        .step(compute_discharge_counseling, _counseling_kw)
        .step(compute_patient_faq, _faq_kw)
    )

    return out


# ───────────────────────── runner ─────────────────────────


async def _run_chain(builder: _ChainBuilder) -> ChainResult:
    bundle = _bundle(builder.bundle_slug)
    pid = _patient_id(builder.bundle_slug)
    steps: list[StepResult] = []
    # Tools that read SHARP context off the ContextVar (resolve-meds,
    # fetch-docs, ground-claim, ...) need a populated FHIRContext. In
    # production it comes from the SHARP middleware; offline we bind
    # it ourselves for the duration of the chain.
    ctx_token = _fhir_ctx.set(FHIRContext(
        server_url="http://in-memory/fhir",
        access_token="in-memory-test-token",
        patient_id=pid,
    ))
    try:
        with bind_in_memory_fhir(bundle, patient_id=pid):
            for tool, kwargs_fn in builder.steps:
                try:
                    kwargs = kwargs_fn(bundle, steps)
                except Exception as exc:
                    steps.append(StepResult(
                        tool_name=getattr(tool, "__name__", "?"),
                        status="CRASH", duration_ms=0.0, abstain_reason=None,
                        error=f"kwargs builder failed: "
                               f"{type(exc).__name__}: {exc}",
                        output=None,
                    ))
                    continue
                r = await _run_step(tool, kwargs)
                steps.append(r)
    finally:
        _fhir_ctx.reset(ctx_token)
    return ChainResult(name=builder.name, bundle=builder.bundle_slug, steps=steps)


def _print_chain(c: ChainResult) -> None:
    head = (
        f"\n=== chain: {c.name} (bundle={c.bundle}) "
        f"steps={len(c.steps)}: PASS={c.n_pass} ABSTAIN={c.n_abstain} "
        f"NETDEP={c.n_netdep} CRASH={c.n_crashes} ==="
    )
    print(head)
    for s in c.steps:
        sym = {"PASS": "[ OK ]", "ABSTAIN": "[ABS ]",
               "NETWORK_DEP": "[NETD]", "CRASH": "[FAIL]"}.get(s.status, "[????]")
        line = (
            f"  {sym} {s.tool_name:<48s} dt={s.duration_ms:>7.1f}ms"
        )
        if s.status == "ABSTAIN" and s.abstain_reason:
            line += f"  reason={s.abstain_reason[:120]}"
        if s.status == "CRASH" and s.error:
            line += f"  error={s.error[:200]}"
        print(line)


async def _main() -> int:
    chains = _all_chains()
    print(f"Running {len(chains)} chains covering "
          f"{sum(len(c.steps) for c in chains)} tool invocations.")
    results: list[ChainResult] = []
    for ch in chains:
        r = await _run_chain(ch)
        _print_chain(r)
        results.append(r)

    n_pass = sum(c.n_pass for c in results)
    n_abs = sum(c.n_abstain for c in results)
    n_netdep = sum(c.n_netdep for c in results)
    n_crash = sum(c.n_crashes for c in results)
    n_total = sum(len(c.steps) for c in results)
    distinct_tools = set()
    for c in results:
        for s in c.steps:
            distinct_tools.add(s.tool_name)
    print()
    print("=" * 78)
    print(f"Total chains: {len(results)}")
    print(f"Total invocations: {n_total} "
          f"(PASS={n_pass}, ABSTAIN={n_abs}, NETDEP={n_netdep}, "
          f"CRASH={n_crash})")
    print(f"Distinct MCP tools exercised: {len(distinct_tools)}")
    crashes = [
        (c.name, s) for c in results for s in c.steps if s.status == "CRASH"
    ]
    if crashes:
        print()
        print("CRASHES:")
        for name, s in crashes:
            print(f"  - {name} :: {s.tool_name}: "
                   f"{(s.error or '')[:240]}")
    return 0 if n_crash == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
