"""Phase 16.H1 - Bulk FHIR $export consumer tests."""

from __future__ import annotations

import json

from a2a_agent.bulk_fhir import (
    BulkFhirSummary, consume_bulk_fhir_lines,
)


def _to_ndjson(resources: list[dict]) -> list[str]:
    return [json.dumps(r) for r in resources]


def test_aggregates_patient_age_band_distribution():
    resources = [
        {"resourceType": "Patient", "id": "p1", "birthDate": "1958-01-01"},
        {"resourceType": "Patient", "id": "p2", "birthDate": "1990-01-01"},
        {"resourceType": "Patient", "id": "p3", "birthDate": "1940-01-01"},
    ]
    res = consume_bulk_fhir_lines(_to_ndjson(resources))
    assert res.n_patients == 3
    assert res.age_band_distribution["65-74"] >= 1
    assert res.age_band_distribution["85+"] >= 1


def test_top_condition_codes_ranked():
    resources = [
        {"resourceType": "Condition",
         "code": {"coding": [{"code": "I50.9"}]}},
        {"resourceType": "Condition",
         "code": {"coding": [{"code": "I50.9"}]}},
        {"resourceType": "Condition",
         "code": {"coding": [{"code": "N17.9"}]}},
    ]
    res = consume_bulk_fhir_lines(_to_ndjson(resources))
    assert res.top_condition_codes[0] == ("I50.9", 2)


def test_handles_missing_birth_date_gracefully():
    res = consume_bulk_fhir_lines(_to_ndjson([
        {"resourceType": "Patient", "id": "x"},
    ]))
    assert res.age_band_distribution.get("unknown", 0) == 1


def test_skips_lines_with_invalid_json():
    res = consume_bulk_fhir_lines([
        '{"resourceType": "Patient", "id": "ok"}',
        'not-json-at-all',
        '',
    ])
    assert res.n_patients == 1
    assert res.n_resources_total == 1


def test_counts_resource_types_separately():
    resources = [
        {"resourceType": "Patient", "id": "a"},
        {"resourceType": "Observation",
         "code": {"coding": [{"code": "718-7"}]}},
        {"resourceType": "MedicationRequest",
         "medicationCodeableConcept": {
             "coding": [{"code": "11289"}]}},
    ]
    res = consume_bulk_fhir_lines(_to_ndjson(resources))
    assert res.n_per_resource_type == {
        "Patient": 1, "Observation": 1, "MedicationRequest": 1,
    }


def test_handles_text_only_medication_code():
    resources = [
        {"resourceType": "MedicationRequest",
         "medicationCodeableConcept": {"text": "warfarin 5 mg"}},
    ]
    res = consume_bulk_fhir_lines(_to_ndjson(resources))
    assert res.top_medication_codes[0] == ("warfarin 5 mg", 1)


def test_round_trip_through_pydantic():
    resources = [{"resourceType": "Patient", "id": "p"}]
    res = consume_bulk_fhir_lines(_to_ndjson(resources))
    payload = res.model_dump(mode="json")
    rebuilt = BulkFhirSummary.model_validate(payload)
    assert rebuilt.n_patients == res.n_patients


def test_top_n_caps_returned_codes():
    resources = [
        {"resourceType": "Condition",
         "code": {"coding": [{"code": f"C{i}"}]}}
        for i in range(20)
    ]
    res = consume_bulk_fhir_lines(_to_ndjson(resources), top_n=5)
    assert len(res.top_condition_codes) == 5
