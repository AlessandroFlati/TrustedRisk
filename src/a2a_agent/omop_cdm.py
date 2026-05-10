"""Phase 16.H2 - OMOP CDM v5.4 exporter.

Many academic medical centers + OHDSI-affiliated networks use the OMOP
Common Data Model rather than FHIR. This module turns a TrustedRisk
DecisionCard + the FHIR Bundle that drove it into the equivalent OMOP
CDM v5.4 row dicts:

  - PERSON               (1 row per patient)
  - VISIT_OCCURRENCE     (1 row per encounter)
  - CONDITION_OCCURRENCE (1 row per Condition)
  - MEASUREMENT          (1 row per quantitative Observation)
  - DRUG_EXPOSURE        (1 row per MedicationRequest)
  - NOTE                 (1 row per DecisionCard - the recommendation +
                          rationale lands as a clinical note)

Pure-Python, no OMOP vocabulary lookup - we emit string codes
(`source_value` columns) and leave the standardized concept_id as 0
(unmapped). Production would chain through Athena's vocabulary tables.

References:
- OMOP CDM v5.4 (OHDSI 2022).
- OHDSI book - https://ohdsi.github.io/CommonDataModel/.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class OMOPRow(BaseModel):
    table: str
    row: dict[str, Any]


class OMOPExportReport(BaseModel):
    n_rows_total: int = Field(ge=0)
    n_per_table: dict[str, int]
    rows: list[OMOPRow]
    rationale: str


_GENDER_TO_OMOP = {
    "male": 8507,
    "female": 8532,
    "m": 8507,
    "f": 8532,
}


def _date_only(iso: str | None) -> str | None:
    if not iso:
        return None
    return iso[:10]


def _person_row(patient: dict[str, Any]) -> dict[str, Any]:
    bd = patient.get("birthDate") or ""
    year = int(bd[:4]) if len(bd) >= 4 and bd[:4].isdigit() else None
    month = int(bd[5:7]) if len(bd) >= 7 and bd[5:7].isdigit() else None
    day = int(bd[8:10]) if len(bd) >= 10 and bd[8:10].isdigit() else None
    gender = (patient.get("gender") or "").lower()
    return {
        "person_id": patient.get("id", ""),
        "gender_concept_id": _GENDER_TO_OMOP.get(gender, 0),
        "gender_source_value": gender or "unknown",
        "year_of_birth": year,
        "month_of_birth": month,
        "day_of_birth": day,
        "birth_datetime": bd or None,
        "race_concept_id": 0,
        "ethnicity_concept_id": 0,
        "person_source_value": patient.get("id", ""),
    }


def _visit_occurrence_row(
    enc: dict[str, Any], person_id: str,
) -> dict[str, Any]:
    period = enc.get("period") or {}
    return {
        "visit_occurrence_id": enc.get("id", ""),
        "person_id": person_id,
        "visit_concept_id": 0,
        "visit_start_date": _date_only(period.get("start")),
        "visit_end_date": _date_only(period.get("end")),
        "visit_start_datetime": period.get("start"),
        "visit_end_datetime": period.get("end"),
        "visit_source_value": (
            (enc.get("class") or {}).get("code") or ""
        ),
    }


def _condition_occurrence_row(
    cond: dict[str, Any], person_id: str,
    visit_id: str | None,
) -> dict[str, Any]:
    code_obj = cond.get("code") or {}
    coding = (code_obj.get("coding") or [{}])[0]
    return {
        "condition_occurrence_id": cond.get("id", ""),
        "person_id": person_id,
        "condition_concept_id": 0,
        "condition_start_date": _date_only(
            cond.get("recordedDate") or cond.get("onsetDateTime")
        ),
        "condition_source_value": (
            coding.get("code") or code_obj.get("text") or ""
        ),
        "visit_occurrence_id": visit_id,
    }


def _measurement_row(
    obs: dict[str, Any], person_id: str,
    visit_id: str | None,
) -> dict[str, Any]:
    code_obj = obs.get("code") or {}
    coding = (code_obj.get("coding") or [{}])[0]
    val = obs.get("valueQuantity") or {}
    return {
        "measurement_id": obs.get("id", ""),
        "person_id": person_id,
        "measurement_concept_id": 0,
        "measurement_date": _date_only(obs.get("effectiveDateTime")),
        "measurement_source_value": coding.get("code") or "",
        "value_as_number": val.get("value"),
        "unit_source_value": val.get("unit"),
        "visit_occurrence_id": visit_id,
    }


def _drug_exposure_row(
    med: dict[str, Any], person_id: str,
    visit_id: str | None,
) -> dict[str, Any]:
    mc = med.get("medicationCodeableConcept") or {}
    coding = (mc.get("coding") or [{}])[0]
    return {
        "drug_exposure_id": med.get("id", ""),
        "person_id": person_id,
        "drug_concept_id": 0,
        "drug_exposure_start_date": _date_only(med.get("authoredOn")),
        "drug_source_value": (
            coding.get("code") or mc.get("text") or ""
        ),
        "visit_occurrence_id": visit_id,
    }


def _note_row(
    decision_card: dict[str, Any], person_id: str,
    visit_id: str | None,
) -> dict[str, Any]:
    rec = decision_card.get("recommendation") or {}
    risk = decision_card.get("risk_estimate") or {}
    text = (
        f"TrustedRisk DecisionCard - action: "
        f"{rec.get('action', 'n/a')}, "
        f"confidence: {rec.get('confidence', 'n/a')}, "
        f"30-day readmission risk: "
        f"{risk.get('probability_mean', 'n/a')}. "
        f"Rationale: {rec.get('rationale', '')}"
    )
    return {
        "note_id": decision_card.get("decision_card_id", ""),
        "person_id": person_id,
        "note_date": _date_only(decision_card.get("created_at_iso")),
        "note_class_concept_id": 0,
        "note_text": text,
        "visit_occurrence_id": visit_id,
        "note_source_value": "TrustedRisk DecisionCard",
    }


def export_to_omop(
    *,
    fhir_bundle: dict[str, Any],
    decision_card: dict[str, Any] | None = None,
) -> OMOPExportReport:
    """Translate a FHIR Bundle (+ optional DecisionCard) into OMOP CDM
    v5.4 rows.

    Args:
        fhir_bundle: a FHIR R4 Bundle dict with `entry[].resource`.
        decision_card: optional TrustedRisk DecisionCard - emitted as
            an OMOP NOTE row.
    """
    if not fhir_bundle or fhir_bundle.get("resourceType") != "Bundle":
        raise ValueError("fhir_bundle must be a FHIR Bundle resource")

    rows: list[OMOPRow] = []
    person_id = ""
    visit_id: str | None = None

    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") or {}
        rt = r.get("resourceType")
        if rt == "Patient":
            person_id = r.get("id", "")
            rows.append(OMOPRow(table="PERSON", row=_person_row(r)))

    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") or {}
        rt = r.get("resourceType")
        if rt == "Encounter":
            visit_id = r.get("id")
            rows.append(OMOPRow(
                table="VISIT_OCCURRENCE",
                row=_visit_occurrence_row(r, person_id),
            ))

    for entry in fhir_bundle.get("entry") or []:
        r = entry.get("resource") or {}
        rt = r.get("resourceType")
        if rt == "Condition":
            rows.append(OMOPRow(
                table="CONDITION_OCCURRENCE",
                row=_condition_occurrence_row(r, person_id, visit_id),
            ))
        elif rt == "Observation":
            rows.append(OMOPRow(
                table="MEASUREMENT",
                row=_measurement_row(r, person_id, visit_id),
            ))
        elif rt == "MedicationRequest":
            rows.append(OMOPRow(
                table="DRUG_EXPOSURE",
                row=_drug_exposure_row(r, person_id, visit_id),
            ))

    if decision_card:
        rows.append(OMOPRow(
            table="NOTE",
            row=_note_row(decision_card, person_id, visit_id),
        ))

    n_per_table: dict[str, int] = {}
    for row in rows:
        n_per_table[row.table] = n_per_table.get(row.table, 0) + 1

    return OMOPExportReport(
        n_rows_total=len(rows),
        n_per_table=n_per_table,
        rows=rows,
        rationale=(
            f"OMOP CDM v5.4 export: {len(rows)} rows across "
            f"{len(n_per_table)} tables. concept_id columns are "
            f"0 (unmapped) - source_value strings retained."
        ),
    )
