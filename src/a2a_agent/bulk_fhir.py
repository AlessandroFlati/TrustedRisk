"""Phase 16.H1 - Bulk FHIR $export consumer.

Streams an NDJSON output produced by a SMART/Backend-Services Bulk
Data $export and aggregates it into a population-level analytics
summary. Pure-Python, NDJSON streaming - never holds the whole file
in memory.

Supports the four resource types most commonly exported for
risk-stratification workloads: Patient, Condition, Observation,
MedicationRequest. Other resource types are counted but not
aggregated.

References:
- HL7 FHIR Bulk Data Access (Flat FHIR) v2.0.0, Section 6 Output.
- ONC SMART/Backend Services Authorization v1.0.0.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, Field


_LACE_AGE_BANDS = [(0, 39), (40, 64), (65, 74), (75, 84), (85, 200)]
_LACE_AGE_LABELS = ["0-39", "40-64", "65-74", "75-84", "85+"]


class BulkFhirSummary(BaseModel):
    n_resources_total: int = Field(ge=0)
    n_per_resource_type: dict[str, int]
    n_patients: int = Field(ge=0)
    n_conditions: int = Field(ge=0)
    n_observations: int = Field(ge=0)
    n_medication_requests: int = Field(ge=0)
    age_band_distribution: dict[str, int]
    sex_distribution: dict[str, int]
    top_condition_codes: list[tuple[str, int]]
    top_observation_codes: list[tuple[str, int]]
    top_medication_codes: list[tuple[str, int]]
    rationale: str


def _stream_ndjson_lines(
    lines: Iterable[str],
) -> Iterator[dict[str, Any]]:
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            yield json.loads(ln)
        except json.JSONDecodeError:
            continue


def _age_band_for(age: int | None) -> str:
    if age is None:
        return "unknown"
    for (lo, hi), label in zip(_LACE_AGE_BANDS, _LACE_AGE_LABELS):
        if lo <= age <= hi:
            return label
    return "unknown"


def _patient_age(patient: dict[str, Any]) -> int | None:
    """Best-effort parse of FHIR Patient.birthDate -> age in years."""
    bd = patient.get("birthDate")
    if not bd:
        return None
    try:
        year = int(bd[:4])
    except (ValueError, IndexError):
        return None
    # Use 2026 as a deterministic reference year for the calibrated
    # cohort; production would use date.today().year.
    return max(0, 2026 - year)


def _first_code(coding: list[dict[str, Any]] | None) -> str | None:
    if not coding:
        return None
    first = coding[0]
    return first.get("code") or first.get("display")


def consume_bulk_fhir_lines(
    lines: Iterable[str], *, top_n: int = 10,
) -> BulkFhirSummary:
    """Aggregate an NDJSON stream of FHIR resources into a single
    population summary.

    Args:
        lines: iterator/iterable of NDJSON strings (one resource each).
        top_n: how many top codes to report per resource type.
    """
    n_total = 0
    n_per_type: Counter[str] = Counter()
    age_dist: Counter[str] = Counter()
    sex_dist: Counter[str] = Counter()
    cond_codes: Counter[str] = Counter()
    obs_codes: Counter[str] = Counter()
    med_codes: Counter[str] = Counter()
    patient_ids: set[str] = set()

    for resource in _stream_ndjson_lines(lines):
        rt = resource.get("resourceType")
        if not rt:
            continue
        n_total += 1
        n_per_type[rt] += 1

        if rt == "Patient":
            pid = resource.get("id")
            if pid:
                patient_ids.add(pid)
            age = _patient_age(resource)
            age_dist[_age_band_for(age)] += 1
            sex = resource.get("gender")
            sex_dist[(sex or "unknown").lower()] += 1
        elif rt == "Condition":
            code = _first_code(
                (resource.get("code") or {}).get("coding")
            )
            if code:
                cond_codes[code] += 1
        elif rt == "Observation":
            code = _first_code(
                (resource.get("code") or {}).get("coding")
            )
            if code:
                obs_codes[code] += 1
        elif rt == "MedicationRequest":
            mc = resource.get("medicationCodeableConcept") or {}
            code = _first_code(mc.get("coding"))
            if code is None:
                code = mc.get("text")
            if code:
                med_codes[code] += 1

    return BulkFhirSummary(
        n_resources_total=n_total,
        n_per_resource_type=dict(n_per_type),
        n_patients=len(patient_ids) or n_per_type.get("Patient", 0),
        n_conditions=n_per_type.get("Condition", 0),
        n_observations=n_per_type.get("Observation", 0),
        n_medication_requests=n_per_type.get("MedicationRequest", 0),
        age_band_distribution=dict(age_dist),
        sex_distribution=dict(sex_dist),
        top_condition_codes=cond_codes.most_common(top_n),
        top_observation_codes=obs_codes.most_common(top_n),
        top_medication_codes=med_codes.most_common(top_n),
        rationale=(
            f"Bulk FHIR aggregate: {n_total:,} resources across "
            f"{len(n_per_type)} resource types; {len(patient_ids)} "
            f"distinct patients."
        ),
    )


def consume_bulk_fhir_path(path: Path | str, *, top_n: int = 10
                           ) -> BulkFhirSummary:
    """Convenience wrapper - read a single NDJSON file from disk."""
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        return consume_bulk_fhir_lines(f, top_n=top_n)
