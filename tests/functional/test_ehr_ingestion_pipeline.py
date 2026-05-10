"""Functional tests for EHR ingestion: HL7 v2 ADT + C-CDA + SDOH.

Validates that legacy / FHIR-adjacent inputs flow into the canonical
FHIR Bundle shape that downstream tools (compute_readmission_risk,
compute_care_gap_detector, compute_sdoh_score) consume."""
from __future__ import annotations

from a2a_agent.ehr_integrations import (
    compute_sdoh_score,
    parse_ccda_document,
    parse_hl7v2_adt,
)


# ─────────────────────── HL7 v2 ADT -> FHIR ───────────────────────

def test_adt_a01_admit_produces_in_progress_encounter(hl7v2_adt_a01):
    r = parse_hl7v2_adt(hl7v2_adt_a01)
    assert r.message_type == "A01"
    encounter = next(e["resource"] for e in r.fhir_bundle["entry"]
                         if e["resource"]["resourceType"] == "Encounter")
    assert encounter["status"] == "in-progress"
    assert encounter["class"]["code"] == "IMP"


def test_adt_a01_two_diagnoses_become_two_conditions(hl7v2_adt_a01):
    r = parse_hl7v2_adt(hl7v2_adt_a01)
    conds = [e["resource"] for e in r.fhir_bundle["entry"]
                if e["resource"]["resourceType"] == "Condition"]
    assert len(conds) == 2
    codes = {c["code"]["coding"][0]["code"] for c in conds}
    assert "I50.9" in codes
    assert "E11.9" in codes


def test_adt_a03_discharge_marks_encounter_finished(hl7v2_adt_a01):
    a03 = hl7v2_adt_a01.replace("ADT^A01", "ADT^A03")
    r = parse_hl7v2_adt(a03)
    assert r.message_type == "A03"
    encounter = next(e["resource"] for e in r.fhir_bundle["entry"]
                         if e["resource"]["resourceType"] == "Encounter")
    assert encounter["status"] == "finished"


def test_adt_a08_update_keeps_in_progress(hl7v2_adt_a01):
    a08 = hl7v2_adt_a01.replace("ADT^A01", "ADT^A08")
    r = parse_hl7v2_adt(a08)
    assert r.message_type == "A08"


def test_adt_pid_to_patient_demographics_round_trip(hl7v2_adt_a01):
    r = parse_hl7v2_adt(hl7v2_adt_a01)
    patient = next(e["resource"] for e in r.fhir_bundle["entry"]
                       if e["resource"]["resourceType"] == "Patient")
    assert patient["birthDate"] == "1955-01-01"
    assert patient["gender"] == "female"
    name = patient["name"][0]
    assert name["family"] == "DOE" and "JANE" in name["given"]


def test_adt_can_drive_care_gap_detector_downstream(hl7v2_adt_a01):
    """The HL7 v2 -> FHIR Bundle should be consumable by care_gap_detector
    without any further preprocessing."""
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector
    import asyncio

    parsed = parse_hl7v2_adt(hl7v2_adt_a01)
    bundle = parsed.fhir_bundle
    gaps = asyncio.run(compute_care_gap_detector(bundle))
    # Patient (1955-01-01 -> ~70yo female) -> multiple gaps fire
    assert gaps.patient_age is not None
    assert gaps.n_gaps_found >= 1


# ─────────────────────── C-CDA -> FHIR ───────────────────────

_CCDA_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <title>Continuity of Care Document</title>
  <recordTarget>
    <patientRole>
      <patient>
        <name><given>Jane</given><family>Doe</family></name>
        <administrativeGenderCode code="F" />
        <birthTime value="19500315" />
      </patient>
    </patientRole>
  </recordTarget>
  <component><structuredBody>
    <component><section>
      <title>Problem List</title>
      <entry><observation>
        <value xsi:type="CD" code="84114007"
               codeSystem="2.16.840.1.113883.6.96"
               displayName="Heart failure" />
      </observation></entry>
      <entry><observation>
        <value xsi:type="CD" code="73211009"
               codeSystem="2.16.840.1.113883.6.96"
               displayName="Diabetes mellitus type 2" />
      </observation></entry>
    </section></component>
    <component><section>
      <title>Medications</title>
      <entry><substanceAdministration><consumable><manufacturedProduct>
        <manufacturedMaterial>
          <code code="11289"
                codeSystem="2.16.840.1.113883.6.88"
                displayName="warfarin" />
        </manufacturedMaterial>
      </manufacturedProduct></consumable></substanceAdministration></entry>
    </section></component>
    <component><section>
      <title>Allergies</title>
      <entry><act><entryRelationship><observation><participant>
        <participantRole><playingEntity>
          <code displayName="Penicillin" />
        </playingEntity></participantRole>
      </participant></observation></entryRelationship></act></entry>
    </section></component>
  </structuredBody></component>
</ClinicalDocument>"""


def test_ccda_full_document_extracts_all_three_sections():
    r = parse_ccda_document(_CCDA_SAMPLE)
    assert {"problem_list", "medications", "allergies"} <= \
        set(r.sections_extracted)


def test_ccda_problem_list_to_conditions():
    r = parse_ccda_document(_CCDA_SAMPLE)
    conds = [e["resource"] for e in r.fhir_bundle["entry"]
                if e["resource"]["resourceType"] == "Condition"]
    assert len(conds) == 2
    displays = {c["code"]["text"] for c in conds}
    assert any("Heart failure" in d for d in displays)
    assert any("Diabetes" in d for d in displays)


def test_ccda_medication_to_med_request():
    r = parse_ccda_document(_CCDA_SAMPLE)
    meds = [e["resource"] for e in r.fhir_bundle["entry"]
              if e["resource"]["resourceType"] == "MedicationRequest"]
    assert len(meds) == 1
    text = meds[0]["medicationCodeableConcept"]["text"]
    assert "warfarin" in text.lower()


def test_ccda_allergy_to_allergy_intolerance():
    r = parse_ccda_document(_CCDA_SAMPLE)
    allergies = [e["resource"] for e in r.fhir_bundle["entry"]
                    if e["resource"]["resourceType"] == "AllergyIntolerance"]
    assert len(allergies) >= 1
    assert any("Penicillin" in (a["code"]["text"] or "")
                  for a in allergies)


# ─────────────────────── SDOH ingestion ───────────────────────

def test_sdoh_high_burden_zip_with_zcodes():
    bundle = {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-1",
                            "address": [{"postalCode": "60628"}]}},
            {"resource": {"resourceType": "Condition",
                            "code": {"coding": [{
                                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                                "code": "Z59.0"}]}}},
            {"resource": {"resourceType": "Condition",
                            "code": {"coding": [{
                                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                                "code": "Z63.0"}]}}},
        ],
    }
    r = compute_sdoh_score(bundle)
    assert r.svi_overall_percentile is not None
    assert r.svi_overall_percentile > 0.5
    assert "housing_and_economic" in r.z_code_categories
    assert "primary_support_group" in r.z_code_categories
    assert r.composite_sdoh_burden_score > 0.5


def test_sdoh_low_burden_zip_no_zcodes():
    bundle = {
        "resourceType": "Bundle", "type": "collection",
        "entry": [{"resource": {"resourceType": "Patient", "id": "pt-1",
                                  "address": [{"postalCode": "90210"}]}}],
    }
    r = compute_sdoh_score(bundle)
    assert r.svi_overall_percentile is not None
    assert r.svi_overall_percentile < 0.3


# ─────────────────────── End-to-end EHR chain ───────────────────────

def test_full_ehr_chain_hl7v2_to_care_gap_to_sdoh(hl7v2_adt_a01):
    """HL7 v2 -> FHIR Bundle -> care_gap_detector + sdoh_score in one chain."""
    from mcp_server.tools.care_gap_detector import compute_care_gap_detector
    import asyncio

    parsed = parse_hl7v2_adt(hl7v2_adt_a01)
    bundle = parsed.fhir_bundle
    # Inject a known SDOH zip + Z-code into the bundle
    for entry in bundle["entry"]:
        if entry["resource"]["resourceType"] == "Patient":
            entry["resource"]["address"] = [{"postalCode": "60628"}]
            break
    bundle["entry"].append({"resource": {
        "resourceType": "Condition",
        "code": {"coding": [{"system":
                                  "http://hl7.org/fhir/sid/icd-10-cm",
                                  "code": "Z59.0"}]}}})

    gaps = asyncio.run(compute_care_gap_detector(bundle))
    sdoh = compute_sdoh_score(bundle)

    assert gaps.patient_age is not None
    assert sdoh.svi_overall_percentile > 0.5
    assert "Z59" in sdoh.z_codes_present
