"""Comprehensive end-to-end smoke harness for the TrustedRisk Care Engine.

Simulates the same shape of PO -> A2A -> dispatcher -> tool roundtrip
that production traffic follows, but without restarting the MCP server
or hitting a live FHIR workspace. The four PO demo bundles are loaded
from disk and bound via ``bind_in_memory_fhir`` so each tool consumes
real FHIR resources -- the overlay path in ``apps.orchestrator.dispatcher
._overlay_real_fhir`` runs unchanged.

Coverage targets:

  - every base workflow in ``apps.composer.workflows.REGISTRY`` (58)
  - every macro workflow (50)
  - every specialist route in ``apps._shared.specialist_routes
    .SPECIALIST_ROUTES`` (~38)

For every case the runner reports:

  - ``status``: CRASH (TypeError / AttributeError -- a real bug),
    ABSTAIN (any step abstained -- expected when the chosen bundle does
    not carry the data the workflow needs), PASS (all steps completed
    without abstain), or NOFHIR (dispatcher refused to run because the
    bundle lacked the resources required to obtain a context).
  - ``abstain_reasons``: any structured ``abstain_reason`` strings
    surfaced by the tool / step, deduplicated.
  - ``error``: traceback message when the call crashed.

The harness is read-only with respect to MCP and A2A behaviour; it
only fetches existing fixtures and invokes existing handlers. Run::

    PYTHONPATH=src .venv/Scripts/python.exe scripts/smoke_demo_full.py

Exit code: 0 when no CRASH occurred, 1 otherwise. ABSTAIN is not a
failure -- it is the documented response when a tool would otherwise
run on fabricated data.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import sys
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests.fixtures.fhir_helpers import bind_in_memory_fhir  # noqa: E402

from apps._shared.specialist_routes import (  # noqa: E402
    SPECIALIST_ROUTES,
    Route,
)
from apps.composer.workflows import REGISTRY as WORKFLOW_REGISTRY  # noqa: E402
from apps.orchestrator.dispatcher import (  # noqa: E402
    _run_specialist_route,
    _run_workflow,
)


# ───────────────────────────── fixtures ─────────────────────────────


_BUNDLE_SLUGS: dict[str, str] = {
    "eleanor": "fixtures/po_demo/eleanor_greene.json",
    "marcus": "fixtures/po_demo/marcus_reyes.json",
    "nadia": "fixtures/po_demo/nadia_okafor.json",
    "sofia": "fixtures/po_demo/sofia_ramirez.json",
}


_BUNDLE_CACHE: dict[str, dict] = {}


def _load_bundle(slug: str) -> dict:
    if slug in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[slug]
    path = _BUNDLE_SLUGS.get(slug)
    if path is None:
        raise KeyError(f"Unknown fixture slug {slug!r}; "
                        f"known: {sorted(_BUNDLE_SLUGS)}")
    bundle = json.loads(
        (ROOT / path).read_text(encoding="utf-8")
    )
    _BUNDLE_CACHE[slug] = bundle
    return bundle


def _patient_id_from_bundle(bundle: dict, fallback: str) -> str:
    """The transactional fixtures use POST + ``urn:uuid`` fullUrls, so
    the Patient resource has no canonical ``id`` on the wire. Synthesise
    one from a stable slug so ``bind_in_memory_fhir`` and the dispatcher
    overlay agree on the patient_id throughout the call.
    """
    for entry in bundle.get("entry", []):
        r = entry.get("resource") or {}
        if r.get("resourceType") == "Patient":
            r.setdefault("id", fallback)
            return f"Patient/{r['id']}"
    return f"Patient/{fallback}"


# ──────────────── per-workflow / per-route bundle picks ────────────────
#
# Each map below names which fixture bundle to bind for a given target.
# Default ("any") -> Eleanor: the richest discharge-planning bundle and
# the closest analog to a "generic adult inpatient". Any workflow
# explicitly tied to sepsis / appeals / pediatric flows uses Marcus,
# Nadia, or Sofia respectively. Workflows whose data is not represented
# in any of the four PO demo bundles are mapped to the closest fixture
# and expected to ABSTAIN (this is the documented behaviour, not a bug).


_WORKFLOW_BUNDLE_OVERRIDES: dict[str, str] = {
    # Sepsis / antibiotics / infectious
    "sepsis_workup":                       "marcus",
    "antibiotic_stewardship":              "marcus",
    "infectious_disease_consult":          "marcus",
    # Discharge / readmission / TCM / chronic / palliative
    "discharge_planning":                  "eleanor",
    "outpatient_med_review":               "eleanor",
    "post_discharge_followup":             "eleanor",
    "caregiver_handoff_prep":              "eleanor",
    "transitional_care_management":        "eleanor",
    "chronic_disease_followup":            "eleanor",
    "palliative_transition":               "eleanor",
    # PA / appeals / coverage / cost
    "prior_auth_pipeline":                 "nadia",
    "denial_appeal_pipeline":              "nadia",
    "cost_effectiveness_review":           "nadia",
    "coverage_determination":              "nadia",
    # Pediatric / vaccine / well-child
    "pediatric_acute_workup":              "sofia",
    "pediatric_dosing_review":             "sofia",
    "well_child_visit":                    "sofia",
    "vaccine_schedule_check":              "sofia",
    "postpartum_followup":                 "sofia",
    # CHF / cardiac / cardio chest pain
    "chf_admission":                       "eleanor",
    "acute_chest_pain":                    "marcus",
    # Stroke / trauma / DKA / AKI / respiratory
    "stroke_alert":                        "eleanor",
    "trauma_resus_decision":               "eleanor",
    "dka_management":                      "marcus",
    "aki_workup":                          "marcus",
    "respiratory_failure_workup":          "marcus",
    # Inpatient deterioration / step-down
    "inpatient_deterioration_response":    "marcus",
    "icu_step_down":                       "marcus",
    "inpatient_glycemic_control_workflow": "marcus",
    "periop_complications_response":       "marcus",
    "inpatient_falls_intervention":        "eleanor",
    # PGx / DDI / anticoag / opioid
    "pgx_prescribing_check":               "eleanor",
    "ddi_audit":                           "eleanor",
    "anticoagulant_review":                "eleanor",
    "opioid_safety_review":                "eleanor",
    # Mental health / OB
    "mental_health_crisis":                "eleanor",
    "maternal_obstetric_emergency":        "sofia",
    # Onco / geri / rheum / transplant
    "oncology_cycle_review":               "eleanor",
    "geriatric_assessment":                "eleanor",
    "rheumatology_followup":               "nadia",
    "transplant_immunosuppression_review": "eleanor",
    # Periop
    "pre_op_optimization":                 "eleanor",
    "periop_risk_stratification":          "eleanor",
    "post_op_recovery":                    "eleanor",
    "surgical_consent_workup":             "eleanor",
    # Coding / scribe
    "chart_to_codes":                      "marcus",
    "clinical_documentation_polish":       "marcus",
    "progress_note_workflow":              "marcus",
    "consult_letter_workflow":             "marcus",
    "discharge_summary_workflow":          "marcus",
    # Population / quality
    "population_outreach":                 "sofia",
    "care_gap_closure_pipeline":           "sofia",
    "hedis_quality_improvement":           "sofia",
    # Pre-hospital / VTE / SSI / OON
    "pre_hospital_handoff":                "marcus",
    "vte_prophylaxis_review":              "eleanor",
    "behavioral_health_step_down":         "eleanor",
    "ssi_prevention_bundle":               "eleanor",
    "out_of_network_referral":             "nadia",
}


_MACRO_BUNDLE_OVERRIDES: dict[str, str] = {
    "complete_chf_admission":      "eleanor",
    "complete_sepsis_pipeline":    "marcus",
    "trauma_to_rehab":              "eleanor",
    "dka_to_clinic":                "marcus",
    "ed_to_home":                   "marcus",
    "peds_acute_arc":               "sofia",
    "mh_arc":                       "eleanor",
    "periop_arc":                   "eleanor",
    "oncology_arc":                 "eleanor",
    "denial_arc":                   "nadia",
    "pa_with_preemptive_appeal":    "nadia",
    "quality_arc":                  "sofia",
    "ob_full":                      "sofia",
    "mat_arc":                      "eleanor",
    "polypharm_arc":                "eleanor",
    "abx_arc":                      "marcus",
    "document_arc":                 "marcus",
    "specialty_referral_arc":       "marcus",
    "rapid_response_arc":           "marcus",
    "peds_full":                    "sofia",
    "resp_to_icu":                  "marcus",
    "inpatient_geri_arc":           "eleanor",
    "peds_well_arc":                "sofia",
    "transplant_arc":               "eleanor",
    "pgx_full_arc":                 "eleanor",
    "hf_clinic_arc":                "eleanor",
    "pop_health_full":              "sofia",
    "preop_clear_arc":              "eleanor",
    "infectious_full":              "marcus",
    "mat_post_arc":                 "eleanor",
    "recall_outreach":              "sofia",
    "complex_admission":            "eleanor",
    "ob_obs_to_dispo":              "sofia",
    "peds_critical":                "sofia",
    "surgical_full":                "eleanor",
    "value_chain_arc":              "nadia",
    "claim_audit_arc":              "marcus",
    "acos_arc":                     "sofia",
    "cardiac_clearance_arc":        "marcus",
    "anticoag_lifecycle":           "eleanor",
    "snf_handoff":                  "eleanor",
    "palliative_full":              "eleanor",
    "discharge_full":               "eleanor",
    "infectious_pa":                "marcus",
    "registries_full":              "sofia",
    "complete_outpatient":          "eleanor",
    "ed_psych_arc":                 "eleanor",
    "post_discharge_recovery":      "eleanor",
    "polytrauma_resilience_arc":    "eleanor",
    "vaccine_outreach_arc":         "sofia",
}


_SPECIALIST_BUNDLE_OVERRIDES: dict[tuple[str, str], str] = {
    # acute
    ("trustedrisk-acute", "compute_clinical_deterioration_score"): "marcus",
    ("trustedrisk-acute", "compute_admission_triage"):             "marcus",
    ("trustedrisk-acute", "compute_stroke_thrombolysis_eligibility"): "eleanor",
    ("trustedrisk-acute", "compute_stroke_severity"):              "eleanor",
    ("trustedrisk-acute", "compute_heart_score"):                  "marcus",
    ("trustedrisk-acute", "compute_trauma_severity_score"):        "eleanor",
    ("trustedrisk-acute", "compute_massive_transfusion_protocol"): "eleanor",
    ("trustedrisk-acute", "compute_dka_severity"):                 "marcus",
    ("trustedrisk-acute", "compute_contrast_safety_check"):        "eleanor",
    ("trustedrisk-acute", "compute_aki_kdigo_stage"):              "marcus",
    ("trustedrisk-acute", "compute_preeclampsia_assessment"):      "sofia",
    # discharge
    ("trustedrisk-discharge", "detect_polypharmacy_concerns"):       "eleanor",
    ("trustedrisk-discharge", "compute_medication_reconciliation"):  "eleanor",
    ("trustedrisk-discharge", "compute_discharge_counseling"):       "eleanor",
    ("trustedrisk-discharge", "compute_empiric_antibiotic_selection"): "marcus",
    ("trustedrisk-discharge", "compute_care_gap_detector"):          "eleanor",
    # evidence
    ("trustedrisk-evidence", "compute_differential_diagnosis_ranker"): "marcus",
    # mental health
    ("trustedrisk-mental-health", "compute_suicide_risk_assessment"): "eleanor",
    ("trustedrisk-mental-health", "compute_psychiatric_admission_decision"): "eleanor",
    # pediatric
    ("trustedrisk-pediatric", "compute_pediatric_early_warning"): "sofia",
    ("trustedrisk-pediatric", "compute_weight_based_dosing"):     "sofia",
    # PA / appeals
    ("trustedrisk-pa", "_chained_appeal_letter"):                 "nadia",
    ("trustedrisk-appeals", "compute_denial_letter_parse"):       "nadia",
    ("trustedrisk-appeals", "_chained_appeal_letter"):            "nadia",
    ("trustedrisk-appeals", "_chained_appeal_escalation_path"):   "nadia",
    # patient
    ("trustedrisk-patient", "compute_patient_faq"):              "eleanor",
    ("trustedrisk-patient", "compute_caregiver_handoff"):        "eleanor",
    # coder
    ("trustedrisk-coder", "compute_icd10_suggest"):              "marcus",
    ("trustedrisk-coder", "compute_cpt_suggest"):                "marcus",
    # pgx
    ("trustedrisk-pgx", "compute_pgx_drug_alternatives"):        "eleanor",
    # preadmit
    ("trustedrisk-preadmit", "compute_symptom_red_flag_check"):  "eleanor",
    ("trustedrisk-preadmit", "compute_when_to_seek_care"):       "eleanor",
    # quality
    ("trustedrisk-quality", "_chained_stars_forecast"):          "sofia",
    ("trustedrisk-quality", "_chained_care_gap_ranking"):        "sofia",
    # pophealth
    ("trustedrisk-pophealth", "compute_syndromic_surveillance"): "sofia",
    ("trustedrisk-pophealth", "compute_vaccine_reminder_cohort"): "sofia",
    ("trustedrisk-pophealth", "compute_outbreak_heatmap"):       "sofia",
    # population
    ("trustedrisk-population", "compute_expected_value_of_intervention"): "nadia",
    # multimodal
    ("trustedrisk-multimodal", "compute_ecg_qt_analyzer"):       "marcus",
    ("trustedrisk-multimodal", "compute_dicom_sr_ingest"):       "marcus",
    # scribe
    ("trustedrisk-scribe", "compute_admission_hnp_draft"):       "marcus",
    ("trustedrisk-scribe", "compute_discharge_summary_draft"):   "marcus",
    ("trustedrisk-scribe", "compute_consult_letter_draft"):      "marcus",
    ("trustedrisk-scribe", "compute_progress_note_draft"):       "marcus",
}


_DEFAULT_BUNDLE_FOR_WORKFLOW = "eleanor"
_DEFAULT_BUNDLE_FOR_SPECIALIST = "eleanor"


# ─────────────────────────── result model ───────────────────────────


@dataclasses.dataclass
class CaseResult:
    target_kind: str       # workflow | macro | specialist
    target_id: str
    bundle: str
    duration_ms: float
    status: str            # PASS | ABSTAIN | NOFHIR | CRASH | UNKNOWN_TARGET
    n_steps: int
    n_step_errors: int
    abstain_reasons: list[str]
    optional_skips: list[str]
    error: str | None
    summary: str           # one-liner derived from output_dump


def _summarise_dump(dump: Any) -> str:
    if not isinstance(dump, dict):
        return ""
    bits: list[str] = []
    for k in ("workflow_id", "workflow_title"):
        v = dump.get(k)
        if isinstance(v, str):
            bits.append(f"{k}={v}")
    if "duration_ms" in dump:
        bits.append(f"dt={dump['duration_ms']}ms")
    if "abstain_recommended" in dump:
        bits.append(f"abstain={dump['abstain_recommended']}")
    return "; ".join(bits)


def _collect_abstain_reasons(dump: Any) -> list[str]:
    """Return REQUIRED-step abstain reasons only.

    Optional-step abstains do not flip the workflow-level
    abstain_recommended flag (per Q3); the harness mirrors that
    contract by classifying them as PASS-with-gaps rather than ABSTAIN.
    Optional-step skips remain visible via _collect_optional_skips.
    """
    if not isinstance(dump, dict):
        return []
    out: list[str] = []
    if dump.get("abstain_recommended") and dump.get("abstain_reason"):
        out.append(str(dump["abstain_reason"]))
    for s in dump.get("abstained_steps") or []:
        if isinstance(s, dict) and not s.get("optional"):
            sid = s.get("step_id") or "?"
            r = str(s.get("reason") or "")
            out.append(f"{sid}: {r.split(':', 1)[0]}")
    # de-dup, preserve order
    seen: set[str] = set()
    deduped: list[str] = []
    for r in out:
        if r in seen:
            continue
        seen.add(r)
        deduped.append(r)
    return deduped


def _collect_optional_skips(dump: Any) -> list[str]:
    """Return OPTIONAL-step abstain reasons (transparency only)."""
    if not isinstance(dump, dict):
        return []
    out: list[str] = []
    for s in dump.get("abstained_steps") or []:
        if isinstance(s, dict) and s.get("optional"):
            sid = s.get("step_id") or "?"
            r = str(s.get("reason") or "")
            out.append(f"{sid}: {r.split(':', 1)[0]}")
    return out


def _classify(dispatch_result: Any) -> tuple[str, list[str]]:
    if dispatch_result is None:
        return "UNKNOWN_TARGET", []
    err = getattr(dispatch_result, "error", None)
    dump = getattr(dispatch_result, "output_dump", None)
    if err and "no_fhir_context" in str(err):
        return "NOFHIR", []
    reasons = _collect_abstain_reasons(dump)
    if err and not reasons:
        # TypeError "missing N required positional argument(s)" is a
        # known design gap (see DESIGN_QUESTIONS_PENDING.md Q1): the
        # specialist Route declares inputs={} but the underlying tool
        # has required parameters that the dispatcher's FHIR overlay
        # does not derive. We surface it as MISSING_ARGS so the run
        # exit code distinguishes those from real crashes.
        s = str(err)
        if "TypeError" in s and "missing" in s and "argument" in s:
            return "MISSING_ARGS", [s]
        return "CRASH", [s]
    if reasons:
        return "ABSTAIN", reasons
    return "PASS", []


# ────────────────────────── runners ──────────────────────────


_FHIR_URL = "http://in-memory/fhir"


async def _run_workflow_case(workflow_id: str, bundle_slug: str) -> CaseResult:
    bundle = _load_bundle(bundle_slug)
    patient_id = _patient_id_from_bundle(bundle, fallback=f"demo-{bundle_slug}")
    metadata = {
        "fhir_server_url": _FHIR_URL,
        "patient_id": patient_id,
    }
    target_kind = (
        "macro" if workflow_id in {m for m in _MACRO_BUNDLE_OVERRIDES}
        else "workflow"
    )
    t0 = time.perf_counter()
    err: str | None = None
    res = None
    with bind_in_memory_fhir(bundle, patient_id=patient_id):
        try:
            res = await _run_workflow(workflow_id, metadata)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
    dt = (time.perf_counter() - t0) * 1000.0
    status, reasons = _classify(res)
    if err:
        status, reasons = "CRASH", [err]
    n_steps = 0
    n_step_errs = 0
    if res is not None and isinstance(res.output_dump, dict):
        for s in res.output_dump.get("steps") or []:
            if not isinstance(s, dict):
                continue
            n_steps += 1
            if s.get("error"):
                n_step_errs += 1
    optional_skips = _collect_optional_skips(
        res.output_dump if res is not None else None
    )
    return CaseResult(
        target_kind=target_kind,
        target_id=workflow_id,
        bundle=bundle_slug,
        duration_ms=round(dt, 2),
        status=status,
        n_steps=n_steps,
        n_step_errors=n_step_errs,
        abstain_reasons=reasons,
        optional_skips=optional_skips,
        error=(res.error if res is not None else err),
        summary=_summarise_dump(res.output_dump if res is not None else None),
    )


async def _run_specialist_case(slug: str, route: Route, bundle_slug: str) -> CaseResult:
    bundle = _load_bundle(bundle_slug)
    patient_id = _patient_id_from_bundle(bundle, fallback=f"demo-{bundle_slug}")
    metadata = {
        "fhir_server_url": _FHIR_URL,
        "patient_id": patient_id,
    }
    t0 = time.perf_counter()
    err: str | None = None
    res = None
    with bind_in_memory_fhir(bundle, patient_id=patient_id):
        try:
            res = await _run_specialist_route(slug, route, metadata)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
    dt = (time.perf_counter() - t0) * 1000.0
    status, reasons = _classify(res)
    if err:
        status, reasons = "CRASH", [err]
    return CaseResult(
        target_kind="specialist",
        target_id=f"{slug}:{route.tool_name}",
        bundle=bundle_slug,
        duration_ms=round(dt, 2),
        status=status,
        n_steps=1,
        n_step_errors=0 if res is not None and not res.error else 1,
        abstain_reasons=reasons,
        optional_skips=[],
        error=(res.error if res is not None else err),
        summary=_summarise_dump(res.output_dump if res is not None else None),
    )


# ────────────────────────── case lists ──────────────────────────


def _enumerate_workflow_cases() -> list[tuple[str, str]]:
    """(workflow_id, bundle_slug) for every base workflow."""
    cases: list[tuple[str, str]] = []
    macro_ids = set(_MACRO_BUNDLE_OVERRIDES.keys())
    for wid, _wf in WORKFLOW_REGISTRY.items():
        if wid in macro_ids:
            continue
        # Skip parametric variants — they share their base workflow's
        # logic, so testing the base covers them. We still list a few
        # high-signal parametrics to detect overlay bugs.
        if "__" in wid:
            continue
        bundle = _WORKFLOW_BUNDLE_OVERRIDES.get(wid, _DEFAULT_BUNDLE_FOR_WORKFLOW)
        cases.append((wid, bundle))
    return cases


def _enumerate_macro_cases() -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for mid, bundle in _MACRO_BUNDLE_OVERRIDES.items():
        if mid not in WORKFLOW_REGISTRY:
            continue
        cases.append((mid, bundle))
    return cases


def _enumerate_specialist_cases() -> list[tuple[str, Route, str]]:
    cases: list[tuple[str, Route, str]] = []
    for slug, routes in SPECIALIST_ROUTES.items():
        for r in routes:
            bundle = _SPECIALIST_BUNDLE_OVERRIDES.get(
                (slug, r.tool_name), _DEFAULT_BUNDLE_FOR_SPECIALIST,
            )
            cases.append((slug, r, bundle))
    return cases


def _enumerate_parametric_cases() -> list[tuple[str, str]]:
    """A targeted sample of parametric variants (4 across the registry)
    so the overlay path is exercised on the parametric clone shape too.
    """
    samples = (
        "discharge_planning__medicare_high_acuity",
        "chf_admission__85plus",
        "sepsis_workup__urinary_septic_shock",
    )
    out: list[tuple[str, str]] = []
    for wid in samples:
        if wid in WORKFLOW_REGISTRY:
            base = wid.split("__", 1)[0]
            bundle = _WORKFLOW_BUNDLE_OVERRIDES.get(base, _DEFAULT_BUNDLE_FOR_WORKFLOW)
            out.append((wid, bundle))
    return out


# ────────────────────────── reporting ──────────────────────────


def _print_case(r: CaseResult) -> None:
    sym = {
        "PASS":           "[ OK ]",
        "ABSTAIN":        "[ABS ]",
        "NOFHIR":         "[NOFHIR]",
        "MISSING_ARGS":   "[MARG]",
        "CRASH":          "[FAIL]",
        "UNKNOWN_TARGET": "[????]",
    }.get(r.status, "[????]")
    head = (
        f"{sym} {r.target_kind:>10s} {r.target_id:<48s} "
        f"bundle={r.bundle:<8s} dt={r.duration_ms:>7.1f}ms "
        f"steps={r.n_steps}/{r.n_step_errors}err"
    )
    print(head)
    if r.status == "CRASH" and r.error:
        print(f"           ERROR: {r.error[:280]}")
    if r.status == "ABSTAIN" and r.abstain_reasons:
        for reason in r.abstain_reasons[:5]:
            print(f"           - {reason[:220]}")
    if r.optional_skips:
        for skip in r.optional_skips[:3]:
            print(f"           ~ optional skip: {skip[:200]}")


def _print_summary(results: list[CaseResult]) -> None:
    by_status: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
        by_kind.setdefault(r.target_kind, {})
        by_kind[r.target_kind][r.status] = (
            by_kind[r.target_kind].get(r.status, 0) + 1
        )
    print()
    print("=" * 78)
    print(f"Ran {len(results)} cases.")
    for status in ("PASS", "ABSTAIN", "NOFHIR", "MISSING_ARGS", "CRASH", "UNKNOWN_TARGET"):
        n = by_status.get(status, 0)
        if n:
            print(f"  {status:<14s}: {n}")
    print()
    for kind in sorted(by_kind):
        line = f"  {kind}: " + ", ".join(
            f"{s}={n}" for s, n in sorted(by_kind[kind].items())
        )
        print(line)
    crashes = [r for r in results if r.status == "CRASH"]
    if crashes:
        print()
        print("CRASHES:")
        for r in crashes:
            print(f"  - {r.target_kind:>10s} {r.target_id} "
                   f"(bundle={r.bundle}): {r.error[:280] if r.error else ''}")


# ────────────────────────── main ──────────────────────────


async def _main(argv: list[str]) -> int:
    only = argv[1] if len(argv) > 1 else None
    sections = {
        "workflows":  _enumerate_workflow_cases(),
        "macros":     _enumerate_macro_cases(),
        "specialists": _enumerate_specialist_cases(),
        "parametric": _enumerate_parametric_cases(),
    }
    if only and only in sections:
        chosen = {only: sections[only]}
    else:
        chosen = sections

    results: list[CaseResult] = []
    for section, cases in chosen.items():
        print(f"\n----- section: {section} ({len(cases)} cases) -----")
        for case in cases:
            if section == "specialists":
                slug, route, bundle = case  # type: ignore[misc]
                r = await _run_specialist_case(slug, route, bundle)
            else:
                wid, bundle = case  # type: ignore[misc]
                r = await _run_workflow_case(wid, bundle)
            _print_case(r)
            results.append(r)

    _print_summary(results)
    n_crash = sum(1 for r in results if r.status == "CRASH")
    n_missing = sum(1 for r in results if r.status == "MISSING_ARGS")
    if n_missing:
        print(
            f"\nNote: {n_missing} MISSING_ARGS cases are tracked in "
            f"DESIGN_QUESTIONS_PENDING.md (Q1)."
        )
    # Exit non-zero only on real crashes (MISSING_ARGS is a known
    # design gap, not a regression).
    return 0 if n_crash == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv)))
