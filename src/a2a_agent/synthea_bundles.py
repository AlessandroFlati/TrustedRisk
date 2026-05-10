"""Phase 10.5 -- Synthea-style FHIR R4 Bundle generator.

Produces deterministic FHIR R4 transaction Bundles that mimic the
Synthea synthetic-cohort distributions but ship as inline Python
fixtures (no external Synthea-jar dependency). Each generated patient
has a stable identifier so the loader can PUT-by-identifier against
HAPI for idempotent upsert.

Output Bundle shape:
    Bundle (transaction) of:
        Patient
        Encounter
        Condition (1-3)
        Observation (3-6 vitals + labs)
        MedicationRequest (0-4)

Distributions are calibrated to match the W1 LACE marginals already
used by `synthea_10k_validation.py` so that the runtime calibration
metrics on the 1k cohort fall in the same band as the 10k one.

Pure-deterministic given (seed, patient_index). No network or jar.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


# ─────────────────────────────────────────────────────────────────────
# Calibrated priors (W1 marginals)
# ─────────────────────────────────────────────────────────────────────

_RACES = ["white", "black", "hispanic", "asian", "other"]
_RACE_WEIGHTS = [55, 17, 18, 6, 4]
_SEXES = ["male", "female"]
_INSURANCES = ["medicare", "medicaid", "commercial", "self_pay"]
_INSURANCE_WEIGHTS = [35, 18, 40, 7]

_CONDITION_PALETTE = [
    ("I50.9", "Heart failure, unspecified"),
    ("E11.9", "Type 2 diabetes mellitus without complications"),
    ("J44.9", "Chronic obstructive pulmonary disease"),
    ("N18.3", "Chronic kidney disease, stage 3"),
    ("I10",   "Essential (primary) hypertension"),
    ("I25.10","Atherosclerotic heart disease without angina"),
    ("F32.9", "Major depressive disorder, single episode, unspecified"),
    ("M19.90","Unspecified osteoarthritis, unspecified site"),
]

_MEDICATION_PALETTE = [
    ("404642", "Metformin 500 MG Oral Tablet"),
    ("311036", "Lisinopril 10 MG Oral Tablet"),
    ("197361", "Atorvastatin 20 MG Oral Tablet"),
    ("197517", "Furosemide 40 MG Oral Tablet"),
    ("310965", "Levothyroxine 0.05 MG Oral Tablet"),
    ("197591", "Carvedilol 25 MG Oral Tablet"),
    ("198440", "Aspirin 81 MG Oral Tablet"),
    ("212033", "Warfarin 5 MG Oral Tablet"),
]

# (LOINC code, display, unit, ucum)
_OBSERVATION_PALETTE = [
    ("8462-4",  "Diastolic blood pressure", "mm[Hg]"),
    ("8480-6",  "Systolic blood pressure",  "mm[Hg]"),
    ("8867-4",  "Heart rate",                "/min"),
    ("9279-1",  "Respiratory rate",          "/min"),
    ("8310-5",  "Body temperature",          "Cel"),
    ("4548-4",  "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    ("2160-0",  "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
    ("718-7",   "Hemoglobin [Mass/volume] in Blood", "g/dL"),
]


# ─────────────────────────────────────────────────────────────────────
# Stable identifier
# ─────────────────────────────────────────────────────────────────────

_TR_SYS = "https://trustedrisk.local/synthea-id"


def _stable_id(prefix: str, seed: int, idx: int, kind: str = "") -> str:
    """A stable but readable resource id derived from (seed, idx)."""
    raw = f"{seed}-{idx}-{kind}".encode("utf-8")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    return f"{prefix}-{digest}"


# ─────────────────────────────────────────────────────────────────────
# Sub-builders
# ─────────────────────────────────────────────────────────────────────


def _patient(seed: int, idx: int, rng: random.Random) -> dict[str, Any]:
    pid = _stable_id("syn-pt", seed, idx, "patient")
    age = rng.randint(40, 92)
    sex = rng.choice(_SEXES)
    race = rng.choices(_RACES, weights=_RACE_WEIGHTS)[0]
    return {
        "resourceType": "Patient",
        "id": pid,
        "identifier": [{
            "system": _TR_SYS,
            "value": pid,
        }],
        "name": [{
            "use": "official",
            "family": f"Synth-{idx:06d}",
            "given": [f"Test{idx:06d}"],
        }],
        "gender": sex,
        "birthDate": f"{2026 - age:04d}-01-01",
        "extension": [
            {
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                "valueCodeableConcept": {"text": race},
            },
        ],
    }


def _encounter(seed: int, idx: int, patient_ref: str) -> dict[str, Any]:
    eid = _stable_id("syn-enc", seed, idx, "encounter")
    return {
        "resourceType": "Encounter",
        "id": eid,
        "identifier": [{"system": _TR_SYS, "value": eid}],
        "status": "finished",
        "class": {
            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
            "code": "IMP", "display": "inpatient encounter",
        },
        "subject": {"reference": f"Patient/{patient_ref}"},
        "period": {
            "start": "2026-04-15T08:00:00Z",
            "end":   "2026-04-19T15:30:00Z",
        },
    }


def _conditions(seed: int, idx: int, patient_ref: str,
                  rng: random.Random) -> list[dict[str, Any]]:
    n = rng.choices([1, 2, 3], weights=[40, 40, 20])[0]
    chosen = rng.sample(_CONDITION_PALETTE, k=n)
    out: list[dict[str, Any]] = []
    for j, (code, display) in enumerate(chosen):
        cid = _stable_id("syn-cond", seed, idx, f"cond-{j}")
        out.append({
            "resourceType": "Condition",
            "id": cid,
            "identifier": [{"system": _TR_SYS, "value": cid}],
            "subject": {"reference": f"Patient/{patient_ref}"},
            "code": {
                "coding": [{
                    "system": "http://hl7.org/fhir/sid/icd-10-cm",
                    "code": code, "display": display,
                }],
                "text": display,
            },
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                }],
            },
            "verificationStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                }],
            },
        })
    return out


def _observations(seed: int, idx: int, patient_ref: str,
                     rng: random.Random) -> list[dict[str, Any]]:
    n = rng.choices([3, 4, 5, 6], weights=[20, 30, 30, 20])[0]
    chosen = rng.sample(_OBSERVATION_PALETTE, k=n)
    out: list[dict[str, Any]] = []
    for j, (code, display, unit) in enumerate(chosen):
        oid = _stable_id("syn-obs", seed, idx, f"obs-{j}")
        # crude value distribution per LOINC
        if code == "8480-6":   # systolic BP
            value = round(rng.gauss(132.0, 18.0), 1)
        elif code == "8462-4": # diastolic BP
            value = round(rng.gauss(82.0, 11.0), 1)
        elif code == "8867-4": # HR
            value = round(rng.gauss(78.0, 14.0), 1)
        elif code == "4548-4": # HbA1c %
            value = round(rng.gauss(7.5, 1.7), 1)
        elif code == "2160-0": # Creatinine
            value = round(rng.gauss(1.2, 0.6), 2)
        elif code == "718-7":  # Hemoglobin
            value = round(rng.gauss(13.0, 1.6), 1)
        elif code == "8310-5": # Temperature C
            value = round(rng.gauss(37.0, 0.6), 1)
        else:
            value = round(rng.gauss(60.0, 10.0), 1)

        out.append({
            "resourceType": "Observation",
            "id": oid,
            "identifier": [{"system": _TR_SYS, "value": oid}],
            "status": "final",
            "subject": {"reference": f"Patient/{patient_ref}"},
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs"
                        if code in ("8480-6", "8462-4", "8867-4", "9279-1", "8310-5")
                        else "laboratory",
                }],
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": code, "display": display,
                }],
            },
            "valueQuantity": {
                "value": value, "unit": unit,
                "system": "http://unitsofmeasure.org", "code": unit,
            },
            "effectiveDateTime": "2026-04-17T09:00:00Z",
        })
    return out


def _medications(seed: int, idx: int, patient_ref: str,
                       rng: random.Random) -> list[dict[str, Any]]:
    n = rng.choices([0, 1, 2, 3, 4], weights=[10, 22, 30, 25, 13])[0]
    if n == 0:
        return []
    chosen = rng.sample(_MEDICATION_PALETTE, k=n)
    out: list[dict[str, Any]] = []
    for j, (rxnorm, display) in enumerate(chosen):
        mid = _stable_id("syn-medreq", seed, idx, f"med-{j}")
        out.append({
            "resourceType": "MedicationRequest",
            "id": mid,
            "identifier": [{"system": _TR_SYS, "value": mid}],
            "status": "active",
            "intent": "order",
            "subject": {"reference": f"Patient/{patient_ref}"},
            "medicationCodeableConcept": {
                "coding": [{
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": rxnorm, "display": display,
                }],
                "text": display,
            },
            "authoredOn": "2026-04-17",
        })
    return out


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def generate_bundle(*, patient_index: int, seed: int = 4242) -> dict[str, Any]:
    """Generate one transaction Bundle. Deterministic given (seed, idx).

    Args:
        patient_index: 0-based ordinal of the patient in the cohort.
        seed: campaign seed; reuse to reproduce a cohort.

    Returns:
        FHIR R4 Bundle as a dict.
    """
    # Per-patient RNG so that the contents of patient i are independent
    # of those of patient i+1 (no carry-over).
    rng = random.Random(hash((seed, patient_index)) & 0xFFFF_FFFF)
    patient = _patient(seed, patient_index, rng)
    pid = patient["id"]
    encounter = _encounter(seed, patient_index, pid)
    conditions = _conditions(seed, patient_index, pid, rng)
    observations = _observations(seed, patient_index, pid, rng)
    medications = _medications(seed, patient_index, pid, rng)

    entries: list[dict[str, Any]] = []
    # PUT semantics with ?identifier= query -> idempotent upsert on HAPI
    for resource in [patient, encounter, *conditions, *observations,
                          *medications]:
        rt = resource["resourceType"]
        ident = resource["identifier"][0]
        url = (
            f"{rt}?identifier="
            f"{ident['system']}|{ident['value']}"
        )
        entries.append({
            "fullUrl": f"urn:uuid:{resource['id']}",
            "resource": resource,
            "request": {"method": "PUT", "url": url},
        })

    return {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": entries,
    }


def generate_cohort(*, n: int, seed: int = 4242) -> list[dict[str, Any]]:
    """Generate `n` Bundles. Deterministic given (seed, n)."""
    if n < 1:
        raise ValueError("n must be ≥ 1")
    return [generate_bundle(patient_index=i, seed=seed) for i in range(n)]
