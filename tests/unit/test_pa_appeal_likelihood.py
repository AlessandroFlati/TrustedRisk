"""Unit tests for compute_pa_appeal_likelihood (Phase 2.1 PA-3)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.pa_appeal_likelihood import compute_pa_appeal_likelihood
from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
from mcp_server.tools.pa_payer_rules_match import compute_pa_payer_rules_match
from shared.schemas import PARequestedService


def _run(coro):
    return asyncio.run(coro)


def _bundle_full() -> dict:
    return {"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {
            "resourceType": "Condition",
            "id": "cond-back-pain",
            "code": {
                "coding": [{
                    "code": "M54.5",
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "display": "Low back pain",
                }],
                "text": "Low back pain",
            },
            "recordedDate": "2026-04-01",
        }},
        {"resource": {
            "resourceType": "Observation",
            "id": "obs-cmp",
            "code": {"coding": [{"code": "1742-6", "display": "ALT"}]},
            "valueQuantity": {"value": 35, "unit": "U/L"},
            "effectiveDateTime": "2026-04-15T00:00:00Z",
        }},
        {"resource": {
            "resourceType": "MedicationRequest",
            "id": "med-ibuprofen",
            "medicationCodeableConcept": {"text": "ibuprofen"},
            "authoredOn": "2026-03-01",
        }},
        {"resource": {
            "resourceType": "Procedure",
            "id": "proc-xray",
            "code": {"text": "Plain-film X-ray lumbar spine"},
            "performedDateTime": "2026-02-15",
        }},
    ]}


async def _full_pipeline(payer: str, n_prior_denials: int = 0):
    pack = await compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced",
            cpt_codes=["72148"],
            description="MRI lumbar spine",
        ),
        payer=payer,
        fhir_bundle=_bundle_full(),
        chart_excerpts=[
            {"text": "specialist consult", "fhir_resource_id": "chart-1"},
        ],
    )
    rules = await compute_pa_payer_rules_match(pack)
    est = await compute_pa_appeal_likelihood(pack, rules, n_prior_denials)
    return pack, rules, est


# ─────────────────────── Behaviour ───────────────────────

def test_estimate_in_unit_range():
    _, _, est = _run(_full_pipeline("unitedhealth"))
    assert 0.0 <= est.probability_of_approval <= 1.0
    assert est.ci95[0] <= est.probability_of_approval <= est.ci95[1]


def test_estimate_carries_drivers():
    _, _, est = _run(_full_pipeline("unitedhealth"))
    assert est.drivers
    assert any("base approval rate" in d.lower() for d in est.drivers)
    assert any("rules alignment" in d.lower() for d in est.drivers)


def test_prior_denials_lower_probability():
    _, _, est0 = _run(_full_pipeline("unitedhealth", n_prior_denials=0))
    _, _, est3 = _run(_full_pipeline("unitedhealth", n_prior_denials=3))
    assert est3.probability_of_approval < est0.probability_of_approval


def test_medicare_higher_than_commercial_for_imaging():
    """Medicare base approval rate is higher than commercial payers."""
    _, _, est_uhc = _run(_full_pipeline("unitedhealth"))
    _, _, est_medicare = _run(_full_pipeline("medicare"))
    # Medicare approval rates are systematically higher (~91% vs ~69%)
    assert est_medicare.probability_of_approval > est_uhc.probability_of_approval


def test_probability_clamped_to_unit_floor():
    """Many prior denials shouldn't drive probability negative."""
    _, _, est = _run(_full_pipeline("unitedhealth", n_prior_denials=100))
    assert est.probability_of_approval >= 0.02


def test_unknown_payer_falls_back_to_generic():
    """A payer/service combo without a specific prior should still
    produce a usable estimate from the GENERIC base rate."""
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="dme",
            description="walker",
        ),
        payer="generic",
        fhir_bundle=_bundle_full(),
    ))
    rules = _run(compute_pa_payer_rules_match(pack))
    est = _run(compute_pa_appeal_likelihood(pack, rules))
    assert 0.0 <= est.probability_of_approval <= 1.0
    assert any("generic" in d.lower() or "fallback" in d.lower()
                  for d in est.drivers)


def test_confidence_tier_set():
    _, _, est = _run(_full_pipeline("unitedhealth"))
    assert est.confidence in ("preferred", "degraded", "abstain_recommended")


def test_ci_brackets_probability():
    _, _, est = _run(_full_pipeline("unitedhealth"))
    lo, hi = est.ci95
    assert lo <= est.probability_of_approval <= hi
    assert hi - lo > 0


def test_full_aligned_higher_than_misaligned():
    """A pack that meets all payer rules produces a higher probability
    than one that meets none -- sanity check on the alignment-boost
    factor."""
    bundle_strong = _bundle_full()
    bundle_weak = {"resourceType": "Bundle", "entry": [
        {"resource": {
            "resourceType": "Condition",
            "id": "c", "code": {"coding": [{"code": "M54.9"}]},
        }},
        {"resource": {
            "resourceType": "Observation",
            "id": "o",
            "code": {"coding": [{"code": "1742-6"}]},
            "valueQuantity": {"value": 1},
            "effectiveDateTime": "2020-01-01T00:00:00Z",
        }},
        {"resource": {
            "resourceType": "MedicationRequest",
            "id": "m",
            "medicationCodeableConcept": {"text": "x"},
        }},
    ]}
    pack_s = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced", description="MRI",
        ),
        payer="unitedhealth", fhir_bundle=bundle_strong,
        chart_excerpts=[
            {"text": "specialist", "fhir_resource_id": "chart-1"},
        ],
    ))
    pack_w = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=PARequestedService(
            service_type="imaging_advanced", description="MRI",
        ),
        payer="unitedhealth", fhir_bundle=bundle_weak,
    ))
    rules_s = _run(compute_pa_payer_rules_match(pack_s))
    rules_w = _run(compute_pa_payer_rules_match(pack_w))
    est_s = _run(compute_pa_appeal_likelihood(pack_s, rules_s))
    est_w = _run(compute_pa_appeal_likelihood(pack_w, rules_w))
    assert est_s.probability_of_approval > est_w.probability_of_approval
