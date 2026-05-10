"""Unit tests for compute_medication_reconciliation."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mcp_server.tools.medication_reconciliation import (
    _classify_drug,
    _diff_by_name,
    _dose_changes,
    _normalize_med,
    _parse_iso_datetime,
    compute_medication_reconciliation,
)
from shared.schemas import Medication


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── _classify_drug ───────────────────────

@pytest.mark.parametrize("name,expected", [
    ("warfarin 5 mg", "anticoagulant_vka"),
    ("apixaban", "anticoagulant_doac"),
    ("Eliquis 5 mg", "anticoagulant_doac"),
    ("Heparin sodium", "anticoagulant_heparin"),
    ("insulin glargine", "insulin"),
    ("Lantus", "insulin"),
    ("empagliflozin 10 mg", "sglt2_inhibitor"),
    ("metformin 500 mg", "biguanide"),
    ("lisinopril 10 mg", "ace_inhibitor"),
    ("losartan 50 mg", "arb"),
    ("metoprolol succinate", "beta_blocker"),
    ("spironolactone 25 mg", "mra"),
    ("furosemide 40 mg", "loop_diuretic"),
    ("tacrolimus", "immunosuppressant"),
    ("oxycodone 5 mg", "opioid"),
])
def test_classify_known_drugs(name, expected):
    assert _classify_drug(name) == expected


def test_classify_unknown_drug():
    assert _classify_drug("unknown_compound_xyz") is None


def test_classify_empty_string():
    assert _classify_drug("") is None


# ─────────────────────── _normalize_med ───────────────────────

def test_normalize_med_with_dict():
    m = _normalize_med({"name": "warfarin 5 mg", "dose": "5 mg", "route": "PO"})
    assert m.name == "warfarin 5 mg"
    assert m.dose == "5 mg"
    assert m.drug_class == "anticoagulant_vka"


def test_normalize_med_with_alt_keys():
    m = _normalize_med({"medication": "metformin 500 mg"})
    assert m.name == "metformin 500 mg"
    assert m.drug_class == "biguanide"


def test_normalize_med_unknown_class():
    m = _normalize_med({"name": "vitamin C"})
    assert m.drug_class is None


def test_normalize_med_string_input():
    m = _normalize_med("oxycodone 5mg")
    assert m.name == "oxycodone 5mg"
    assert m.drug_class == "opioid"


# ─────────────────────── _diff_by_name + _dose_changes ───────────────────────

def test_diff_by_name_added():
    adm = [Medication(name="lisinopril 10 mg")]
    dis = [
        Medication(name="lisinopril 10 mg"),
        Medication(name="warfarin 5 mg", drug_class="anticoagulant_vka"),
    ]
    added = _diff_by_name(dis, adm)
    assert len(added) == 1
    assert added[0].name == "warfarin 5 mg"


def test_diff_by_name_removed():
    adm = [Medication(name="metformin 500 mg")]
    dis: list[Medication] = []
    removed = _diff_by_name(adm, dis)
    assert len(removed) == 1


def test_dose_changes():
    adm = [Medication(name="lisinopril 10 mg", dose="10 mg")]
    dis = [Medication(name="lisinopril 10 mg", dose="20 mg")]
    changes = _dose_changes(adm, dis)
    assert len(changes) == 1
    assert changes[0].dose == "20 mg"


def test_no_dose_changes_when_same():
    adm = [Medication(name="lisinopril 10 mg", dose="10 mg")]
    dis = [Medication(name="lisinopril 10 mg", dose="10 mg")]
    assert _dose_changes(adm, dis) == []


# ─────────────────────── _parse_iso_datetime ───────────────────────

def test_parse_iso_with_z():
    d = _parse_iso_datetime("2025-12-15T10:00:00Z")
    assert d is not None
    assert d.year == 2025


def test_parse_iso_offset():
    d = _parse_iso_datetime("2025-12-15T10:00:00+02:00")
    assert d is not None


def test_parse_iso_invalid():
    assert _parse_iso_datetime("garbage") is None
    assert _parse_iso_datetime(None) is None


# ─────────────────────── End-to-end via stub bundle ───────────────────────

def _stub_bundle(*, admission_meds: list[dict], discharge_meds: list[dict],
                 observations: list[dict] | None = None) -> dict:
    """Construct a minimal FHIR-shaped Bundle with the requested resources."""
    entries = [{"resource": {"resourceType": "Patient", "id": "pt-test",
                              "birthDate": "1955-01-01"}}]
    for m in admission_meds:
        entries.append({"resource": {
            "resourceType": "MedicationRequest",
            "status": "stopped",
            "medicationCodeableConcept": {"text": m["name"]},
            "dosageInstruction": [{"text": m.get("dose", "")}] if m.get("dose") else [],
        }})
    for m in discharge_meds:
        entries.append({"resource": {
            "resourceType": "MedicationRequest",
            "status": "active",
            "medicationCodeableConcept": {"text": m["name"]},
            "dosageInstruction": [{"text": m.get("dose", "")}] if m.get("dose") else [],
        }})
    for o in observations or []:
        entries.append({"resource": {
            "resourceType": "Observation",
            "code": {"text": o["text"]},
            "effectiveDateTime": o.get("when",
                (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()),
        }})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def _patch_fhir(monkeypatch, bundle: dict):
    from mcp_server.tools import medication_reconciliation as mr

    async def stub_fetch(pid):
        return bundle

    async def stub_resolve(explicit):
        return explicit or "pt-test"

    monkeypatch.setattr(mr, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(mr, "resolve_patient_id", stub_resolve)


def test_med_recon_clean_discharge_no_concerns(monkeypatch):
    """Discharge with insulin + recent fingerstick -> no missing-monitoring concern."""
    bundle = _stub_bundle(
        admission_meds=[{"name": "metformin 500 mg"}],
        discharge_meds=[
            {"name": "metformin 500 mg"},
            {"name": "insulin glargine 20 units"},
        ],
        observations=[
            {"text": "Glucose blood fingerstick 110 mg/dL"},
        ],
    )
    _patch_fhir(monkeypatch, bundle)

    report = _run(compute_medication_reconciliation(patient_id="pt-test"))
    assert report.n_admission_meds == 1
    assert report.n_discharge_meds == 2
    assert any(m.name == "insulin glargine 20 units" for m in report.added)
    # Insulin's monitoring substring "glucose" matched -> no concern
    insulin_concerns = [c for c in report.concerns if "insulin" in c.medication_name.lower()]
    assert insulin_concerns == []


def test_med_recon_anticoagulant_no_inr_high_severity(monkeypatch):
    """Discharge with warfarin but no INR observation -> high severity concern."""
    bundle = _stub_bundle(
        admission_meds=[],
        discharge_meds=[{"name": "warfarin 5 mg"}],
        observations=[
            {"text": "Hemoglobin 14 g/dL"},  # not INR
        ],
    )
    _patch_fhir(monkeypatch, bundle)

    report = _run(compute_medication_reconciliation(patient_id="pt-test"))
    high = [c for c in report.concerns if c.severity == "high"]
    assert any("warfarin" in c.medication_name.lower() for c in high)
    assert report.discharge_contract_satisfied is False


def test_med_recon_doac_with_recent_creatinine_ok(monkeypatch):
    """DOAC + recent creatinine -> no concern."""
    bundle = _stub_bundle(
        admission_meds=[],
        discharge_meds=[{"name": "Apixaban 5 mg"}],
        observations=[
            {"text": "Creatinine 0.9 mg/dL"},
        ],
    )
    _patch_fhir(monkeypatch, bundle)

    report = _run(compute_medication_reconciliation(patient_id="pt-test"))
    doac_concerns = [c for c in report.concerns if "apixaban" in c.medication_name.lower()]
    assert doac_concerns == []


def test_med_recon_opioid_always_flagged(monkeypatch):
    """Opioid prescription always gets a concern (PDMP query unverifiable)."""
    bundle = _stub_bundle(
        admission_meds=[],
        discharge_meds=[{"name": "oxycodone 5 mg"}],
        observations=[],
    )
    _patch_fhir(monkeypatch, bundle)

    report = _run(compute_medication_reconciliation(patient_id="pt-test"))
    opioid_concerns = [c for c in report.concerns if "oxycodone" in c.medication_name.lower()]
    assert len(opioid_concerns) == 1
    assert opioid_concerns[0].severity == "medium"


def test_med_recon_polypharmacy_concern_when_5_high_risk(monkeypatch):
    """≥5 high-risk classes at discharge -> polypharmacy concern."""
    bundle = _stub_bundle(
        admission_meds=[],
        discharge_meds=[
            {"name": "warfarin 5 mg"},
            {"name": "insulin glargine"},
            {"name": "lisinopril 10 mg"},
            {"name": "spironolactone 25 mg"},
            {"name": "Apixaban 5 mg"},  # interaction in real life -- flagged via class count
        ],
        observations=[
            {"text": "INR 2.1"},
            {"text": "Glucose 110"},
            {"text": "Potassium 4.2"},
            {"text": "Creatinine 0.8"},
        ],
    )
    _patch_fhir(monkeypatch, bundle)

    report = _run(compute_medication_reconciliation(patient_id="pt-test"))
    poly = [c for c in report.concerns if c.concern_type == "polypharmacy_high_risk"]
    assert len(poly) == 1


def test_med_recon_explicit_lists_bypasses_fhir():
    """If admission_meds + discharge_meds are passed explicitly, no FHIR fetch."""
    report = _run(compute_medication_reconciliation(
        admission_meds=[{"name": "metformin 500 mg"}],
        discharge_meds=[{"name": "metformin 500 mg"}, {"name": "warfarin 5 mg"}],
        # No observations -> warfarin will trigger missing INR concern
    ))
    assert report.n_admission_meds == 1
    assert report.n_discharge_meds == 2
    assert any("warfarin" in c.medication_name.lower() for c in report.concerns)


def test_med_recon_dose_change_detected():
    report = _run(compute_medication_reconciliation(
        admission_meds=[{"name": "lisinopril 10 mg", "dose": "10 mg"}],
        discharge_meds=[{"name": "lisinopril 10 mg", "dose": "20 mg"}],
    ))
    assert len(report.dose_changed) == 1
    assert report.dose_changed[0].dose == "20 mg"
