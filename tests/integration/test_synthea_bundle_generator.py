"""Phase 10.5 -- Synthea-style FHIR R4 Bundle generator integration tests.

These tests exercise `a2a_agent.synthea_bundles.generate_cohort` at
1k scale and verify R4 conformance + idempotent identifier wiring
without hitting any network. The live HAPI loader is opt-in via
TRUSTEDRISK_LIVE_FHIR=1 and is NOT exercised here.
"""

from __future__ import annotations

import time

import pytest

from a2a_agent.synthea_bundles import generate_bundle, generate_cohort


# ─────────────────────────────────────────────────────────────────────
# Bundle structure
# ─────────────────────────────────────────────────────────────────────

def test_single_bundle_is_transaction_type():
    b = generate_bundle(patient_index=0, seed=4242)
    assert b["resourceType"] == "Bundle"
    assert b["type"] == "transaction"


def test_single_bundle_has_required_resources():
    b = generate_bundle(patient_index=0, seed=4242)
    rts = {entry["resource"]["resourceType"] for entry in b["entry"]}
    # Patient + Encounter + at least one Condition + at least one Obs
    assert "Patient" in rts
    assert "Encounter" in rts
    assert "Condition" in rts
    assert "Observation" in rts


def test_single_bundle_uses_put_for_idempotent_upsert():
    b = generate_bundle(patient_index=0, seed=4242)
    methods = {entry["request"]["method"] for entry in b["entry"]}
    assert methods == {"PUT"}, f"All entries must use PUT, got {methods}"


def test_request_url_carries_identifier_query():
    """Idempotent upsert via PUT-by-identifier requires the
    `?identifier=system|value` query in the request URL."""
    b = generate_bundle(patient_index=0, seed=4242)
    for entry in b["entry"]:
        url = entry["request"]["url"]
        assert "?identifier=" in url, \
            f"Entry URL missing ?identifier= query: {url}"


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────

def test_same_seed_index_yields_identical_bundle():
    a = generate_bundle(patient_index=42, seed=4242)
    b = generate_bundle(patient_index=42, seed=4242)
    assert a == b


def test_different_seed_yields_different_patient_id():
    a = generate_bundle(patient_index=0, seed=4242)
    b = generate_bundle(patient_index=0, seed=4243)
    a_pid = a["entry"][0]["resource"]["id"]
    b_pid = b["entry"][0]["resource"]["id"]
    assert a_pid != b_pid


def test_different_index_yields_different_patient_id():
    a = generate_bundle(patient_index=0, seed=4242)
    b = generate_bundle(patient_index=1, seed=4242)
    a_pid = a["entry"][0]["resource"]["id"]
    b_pid = b["entry"][0]["resource"]["id"]
    assert a_pid != b_pid


# ─────────────────────────────────────────────────────────────────────
# R4 conformance (lightweight)
# ─────────────────────────────────────────────────────────────────────

def test_patient_has_identifier_with_trustedrisk_system():
    b = generate_bundle(patient_index=0, seed=4242)
    pat = next(e["resource"] for e in b["entry"]
                  if e["resource"]["resourceType"] == "Patient")
    idents = pat["identifier"]
    assert any(
        i["system"] == "https://trustedrisk.local/synthea-id"
        for i in idents
    )


def test_encounter_subject_references_patient():
    b = generate_bundle(patient_index=0, seed=4242)
    pat = next(e["resource"] for e in b["entry"]
                  if e["resource"]["resourceType"] == "Patient")
    enc = next(e["resource"] for e in b["entry"]
                  if e["resource"]["resourceType"] == "Encounter")
    assert enc["subject"]["reference"] == f"Patient/{pat['id']}"


def test_observation_value_quantity_has_unit():
    b = generate_bundle(patient_index=0, seed=4242)
    obs = [e["resource"] for e in b["entry"]
              if e["resource"]["resourceType"] == "Observation"]
    for o in obs:
        vq = o.get("valueQuantity")
        assert vq is not None
        assert "value" in vq and "unit" in vq


def test_condition_uses_icd10_coding():
    b = generate_bundle(patient_index=3, seed=4242)
    conds = [e["resource"] for e in b["entry"]
                if e["resource"]["resourceType"] == "Condition"]
    for c in conds:
        codings = c["code"]["coding"]
        assert any(
            coding["system"] == "http://hl7.org/fhir/sid/icd-10-cm"
            for coding in codings
        )


def test_medication_request_uses_rxnorm_coding():
    """Patient 5 happens to have at least one MedicationRequest."""
    for i in range(20):
        b = generate_bundle(patient_index=i, seed=4242)
        meds = [e["resource"] for e in b["entry"]
                   if e["resource"]["resourceType"] == "MedicationRequest"]
        if meds:
            for m in meds:
                codings = m["medicationCodeableConcept"]["coding"]
                assert any(
                    coding["system"]
                        == "http://www.nlm.nih.gov/research/umls/rxnorm"
                    for coding in codings
                )
            return
    pytest.fail("No MedicationRequest in first 20 patients (unexpected)")


# ─────────────────────────────────────────────────────────────────────
# 1k-scale cohort
# ─────────────────────────────────────────────────────────────────────

def test_generate_1k_cohort_under_8_seconds():
    """Generating the full 1k cohort must stay under 8 s on the
    reference machine -- keeps the CI smoke fast and is a regression
    floor against accidental O(n²) patterns in future edits."""
    t0 = time.perf_counter()
    cohort = generate_cohort(n=1000, seed=4242)
    elapsed = time.perf_counter() - t0
    assert len(cohort) == 1000
    assert elapsed < 8.0, f"1k generation took {elapsed:.2f}s"


def test_1k_cohort_resource_counts_within_expected_bands():
    cohort = generate_cohort(n=1000, seed=4242)
    counts: dict[str, int] = {}
    for b in cohort:
        for entry in b["entry"]:
            rt = entry["resource"]["resourceType"]
            counts[rt] = counts.get(rt, 0) + 1
    # 1 Patient + 1 Encounter per bundle = 1000 each
    assert counts["Patient"] == 1000
    assert counts["Encounter"] == 1000
    # 1-3 Conditions per bundle -> expect mean ~ 1.8 -> 1500-2300
    assert 1400 <= counts["Condition"] <= 2400
    # 3-6 Observations per bundle -> expect mean ~ 4.5 -> 4000-5400
    assert 3800 <= counts["Observation"] <= 5600
    # 0-4 Meds per bundle -> expect mean ~ 2.0 -> 1500-2500
    assert 1300 <= counts.get("MedicationRequest", 0) <= 2700


def test_1k_cohort_patient_ids_are_unique():
    cohort = generate_cohort(n=1000, seed=4242)
    pids = [
        next(e["resource"]["id"] for e in b["entry"]
                if e["resource"]["resourceType"] == "Patient")
        for b in cohort
    ]
    assert len(set(pids)) == len(pids)


def test_generate_invalid_n_raises():
    with pytest.raises(ValueError):
        generate_cohort(n=0)
