"""Unit tests for compute_pa_evidence_pack (Phase 2.1 PA-1)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from mcp_server.tools.pa_evidence_pack import compute_pa_evidence_pack
from shared.schemas import PARequestedService


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Fixtures ───────────────────────

def _patient_bundle(
    *,
    diagnoses: list[dict] | None = None,
    observations: list[dict] | None = None,
    meds: list[dict] | None = None,
    procedures: list[dict] | None = None,
) -> dict:
    entries: list[dict] = []
    for i, d in enumerate(diagnoses or []):
        entries.append({"resource": {
            "resourceType": "Condition",
            "id": d.get("id", f"cond-{i}"),
            "code": {
                "coding": [{
                    "code": d["code"],
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "display": d.get("display", d["code"]),
                }],
                "text": d.get("display", d["code"]),
            },
            "recordedDate": d.get("date", "2026-04-01"),
        }})
    for i, o in enumerate(observations or []):
        entries.append({"resource": {
            "resourceType": "Observation",
            "id": o.get("id", f"obs-{i}"),
            "code": {
                "coding": [{
                    "code": o["loinc"],
                    "system": "http://loinc.org",
                    "display": o.get("display", o["loinc"]),
                }],
            },
            "valueQuantity": {"value": o["value"], "unit": o.get("unit", "")},
            "effectiveDateTime": o.get("date", "2026-04-15T00:00:00Z"),
        }})
    for i, m in enumerate(meds or []):
        entries.append({"resource": {
            "resourceType": "MedicationRequest",
            "id": m.get("id", f"med-{i}"),
            "medicationCodeableConcept": {"text": m["name"]},
            "authoredOn": m.get("date", "2026-03-01"),
        }})
    for i, p in enumerate(procedures or []):
        entries.append({"resource": {
            "resourceType": "Procedure",
            "id": p.get("id", f"proc-{i}"),
            "code": {"text": p["name"]},
            "performedDateTime": p.get("date", "2026-02-15"),
        }})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


_REQ_IMAGING = PARequestedService(
    service_type="imaging_advanced",
    cpt_codes=["72148"],
    description="MRI lumbar spine without contrast",
)


# ─────────────────────── Behaviour ───────────────────────

def test_pack_aggregates_each_kind():
    bundle = _patient_bundle(
        diagnoses=[{"code": "M54.5", "display": "Low back pain"}],
        observations=[
            {"loinc": "1742-6", "display": "ALT", "value": 35, "unit": "U/L"},
        ],
        meds=[{"name": "ibuprofen 600 mg"}],
        procedures=[{"name": "Plain-film X-ray lumbar spine"}],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    assert len(pack.diagnoses) == 1
    assert len(pack.relevant_observations) == 1
    assert len(pack.prior_treatments_tried) == 2  # med + procedure
    assert pack.n_evidence_items == 4


def test_pack_strength_score_in_unit_range():
    bundle = _patient_bundle(
        diagnoses=[{"code": "M54.5"}],
        observations=[{"loinc": "1742-6", "value": 35}],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    assert 0.0 <= pack.evidence_strength_score <= 1.0


def test_pack_specific_icd10_higher_relevance():
    bundle_specific = _patient_bundle(
        diagnoses=[{"code": "I50.21", "display": "CHF, acute on chronic"}],
    )
    bundle_unspec = _patient_bundle(
        diagnoses=[{"code": "I50.9", "display": "Heart failure unspecified"}],
    )
    pack_s = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle_specific,
    ))
    pack_u = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle_unspec,
    ))
    assert pack_s.diagnoses[0].relevance_score > pack_u.diagnoses[0].relevance_score


def test_pack_abstains_with_no_diagnosis():
    # A bundle with entries but no Condition resources triggers the
    # no-diagnosis abstain guard (not the empty-bundle guard).
    bundle = _patient_bundle(
        observations=[{"loinc": "2160-0", "value": 1.1, "unit": "mg/dL"}],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    assert pack.abstain_recommended is True
    assert "Condition" in (pack.abstain_reason or "")


def test_pack_abstains_with_empty_bundle():
    # An empty FHIR bundle (entry=[]) triggers the early missing-bundle guard.
    bundle = _patient_bundle()  # entry=[]
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    assert pack.abstain_recommended is True
    assert "missing_fhir_bundle" in (pack.abstain_reason or "")


def test_pack_abstains_when_only_one_evidence_item():
    bundle = _patient_bundle(
        diagnoses=[{"code": "M54.5"}],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    assert pack.abstain_recommended is True
    assert "minimum threshold" in (pack.abstain_reason or "").lower()


def test_pack_observation_recency_drives_relevance():
    """Observations within 30 days score higher than older ones."""
    today_iso = datetime.now(timezone.utc).isoformat()
    bundle = _patient_bundle(
        diagnoses=[{"code": "M54.5"}],
        observations=[
            {"loinc": "1742-6", "value": 35, "date": today_iso, "id": "fresh"},
            {"loinc": "1742-6", "value": 35, "date": "2024-01-01", "id": "stale"},
        ],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    fresh = next(o for o in pack.relevant_observations
                     if o.fhir_resource_id == "fresh")
    stale = next(o for o in pack.relevant_observations
                     if o.fhir_resource_id == "stale")
    assert fresh.relevance_score > stale.relevance_score


def test_pack_handles_dict_request_service():
    """Service can come in as a dict (Pydantic round-trip)."""
    bundle = _patient_bundle(
        diagnoses=[{"code": "M54.5"}],
        observations=[{"loinc": "1742-6", "value": 35}],
        meds=[{"name": "ibuprofen"}],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service={
            "service_type": "imaging_advanced",
            "cpt_codes": ["72148"],
            "description": "MRI lumbar",
        },
        payer="aetna",
        fhir_bundle=bundle,
    ))
    assert pack.requested_service.cpt_codes == ["72148"]


def test_pack_chart_excerpts_added_with_cite_back():
    bundle = _patient_bundle(diagnoses=[{"code": "M54.5"}])
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
        chart_excerpts=[
            {"text": "6 weeks of conservative therapy without improvement",
             "fhir_resource_id": "chart-1", "relevance_score": 0.9},
        ],
    ))
    assert len(pack.chart_excerpts) == 1
    assert pack.chart_excerpts[0].fhir_resource_id == "chart-1"


def test_pack_includes_references():
    bundle = _patient_bundle(
        diagnoses=[{"code": "M54.5"}],
        observations=[{"loinc": "1742-6", "value": 35}],
        meds=[{"name": "ibuprofen"}],
    )
    pack = _run(compute_pa_evidence_pack(
        patient_reference="Patient/abc",
        requested_service=_REQ_IMAGING,
        payer="unitedhealth",
        fhir_bundle=bundle,
    ))
    assert any("AMA" in r for r in pack.references)
    assert any("AHRQ" in r for r in pack.references)
