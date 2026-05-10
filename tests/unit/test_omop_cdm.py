"""Phase 16.H2 - OMOP CDM exporter tests."""

from __future__ import annotations

import pytest

from a2a_agent.omop_cdm import OMOPExportReport, export_to_omop


def _bundle():
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "p1",
                "birthDate": "1955-03-15", "gender": "female",
            }},
            {"resource": {
                "resourceType": "Encounter", "id": "e1",
                "class": {"code": "IMP"},
                "period": {
                    "start": "2026-04-22T00:00:00Z",
                    "end": "2026-04-29T00:00:00Z",
                },
            }},
            {"resource": {
                "resourceType": "Condition", "id": "c1",
                "code": {"coding": [{"code": "I50.21",
                                       "display": "CHF"}]},
                "recordedDate": "2026-04-22",
            }},
            {"resource": {
                "resourceType": "Observation", "id": "o1",
                "code": {"coding": [{"code": "30934-4",
                                       "display": "BNP"}]},
                "valueQuantity": {"value": 800, "unit": "pg/mL"},
                "effectiveDateTime": "2026-04-26T10:00:00Z",
            }},
            {"resource": {
                "resourceType": "MedicationRequest", "id": "m1",
                "medicationCodeableConcept": {
                    "coding": [{"code": "11289",
                                  "display": "Warfarin"}],
                    "text": "warfarin 5 mg",
                },
                "authoredOn": "2026-04-23",
            }},
        ],
    }


def test_export_emits_one_row_per_resource():
    report = export_to_omop(fhir_bundle=_bundle())
    assert report.n_rows_total == 5
    assert report.n_per_table == {
        "PERSON": 1, "VISIT_OCCURRENCE": 1,
        "CONDITION_OCCURRENCE": 1, "MEASUREMENT": 1,
        "DRUG_EXPOSURE": 1,
    }


def test_export_with_decision_card_adds_note_row():
    card = {
        "decision_card_id": "card-1",
        "created_at_iso": "2026-04-30T08:00:00Z",
        "recommendation": {
            "action": "discharge_home",
            "confidence": "high",
            "rationale": "Stable on oral diuretics.",
        },
        "risk_estimate": {"probability_mean": 0.12},
    }
    report = export_to_omop(fhir_bundle=_bundle(), decision_card=card)
    assert "NOTE" in report.n_per_table
    note_row = next(r for r in report.rows if r.table == "NOTE")
    assert "discharge_home" in note_row.row["note_text"]
    assert "0.12" in note_row.row["note_text"]


def test_person_gender_concept_id_mapped_for_female():
    report = export_to_omop(fhir_bundle=_bundle())
    person = next(r for r in report.rows if r.table == "PERSON").row
    assert person["gender_concept_id"] == 8532
    assert person["gender_source_value"] == "female"


def test_year_of_birth_extracted_from_birth_date():
    report = export_to_omop(fhir_bundle=_bundle())
    person = next(r for r in report.rows if r.table == "PERSON").row
    assert person["year_of_birth"] == 1955
    assert person["month_of_birth"] == 3


def test_visit_occurrence_propagates_to_dependent_rows():
    report = export_to_omop(fhir_bundle=_bundle())
    for table in ("CONDITION_OCCURRENCE", "MEASUREMENT",
                  "DRUG_EXPOSURE"):
        row = next(r for r in report.rows if r.table == table).row
        assert row["visit_occurrence_id"] == "e1"


def test_export_rejects_non_bundle():
    with pytest.raises(ValueError):
        export_to_omop(fhir_bundle={"resourceType": "Patient"})


def test_export_rejects_empty_dict():
    with pytest.raises(ValueError):
        export_to_omop(fhir_bundle={})


def test_round_trip_through_pydantic():
    report = export_to_omop(fhir_bundle=_bundle())
    payload = report.model_dump(mode="json")
    rebuilt = OMOPExportReport.model_validate(payload)
    assert rebuilt.n_rows_total == report.n_rows_total
