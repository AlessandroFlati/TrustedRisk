"""Unit tests for EHR-1/2/3 -- legacy integrations."""
from __future__ import annotations

import pytest

from a2a_agent.ehr_integrations import (
    compute_sdoh_score,
    parse_ccda_document,
    parse_hl7v2_adt,
)


# ───────────────────────────────────────────────────────────────────
# EHR-1 -- HL7 v2 ADT parser
# ───────────────────────────────────────────────────────────────────

_SAMPLE_ADT_A01 = (
    "MSH|^~\\&|EPIC|HOSPITAL|RECEIVER|FACILITY|20240429120000||"
    "ADT^A01|MSG12345|P|2.5\r"
    "PID|1||MRN12345^^^HOSPITAL^MR||DOE^JANE||19550101|F\r"
    "PV1|1|I|2N^201^A^HOSPITAL|||||||||||||||VISIT12345"
    "|||||||||||||||||||||||||||20240429120000\r"
    "DG1|1|I10|I50.9^Heart failure unspecified|Heart failure\r"
    "AL1|1|DA|^Penicillin|MO|Hives\r"
)


def test_hl7v2_invalid_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        parse_hl7v2_adt("")


def test_hl7v2_no_msh_raises():
    with pytest.raises(ValueError, match="MSH"):
        parse_hl7v2_adt("PID|1|||DOE^JANE\r")


def test_hl7v2_a01_parses_message_type():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    assert r.message_type == "A01"


def test_hl7v2_a01_extracts_sending_app_and_facility():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    assert r.sending_application == "EPIC"
    assert r.sending_facility == "HOSPITAL"


def test_hl7v2_a01_message_datetime_iso():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    assert r.message_datetime_iso is not None
    assert r.message_datetime_iso.startswith("2024-04-29")


def test_hl7v2_extracts_patient_demographics():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    patient = next(e["resource"] for e in r.fhir_bundle["entry"]
                       if e["resource"]["resourceType"] == "Patient")
    assert patient["birthDate"] == "1955-01-01"
    assert patient["gender"] == "female"
    name = patient["name"][0]
    assert name["family"] == "DOE"
    assert "JANE" in name["given"]


def test_hl7v2_extracts_diagnosis_as_condition():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    conds = [e["resource"] for e in r.fhir_bundle["entry"]
                if e["resource"]["resourceType"] == "Condition"]
    assert len(conds) == 1
    code = conds[0]["code"]["coding"][0]["code"]
    assert code == "I50.9"


def test_hl7v2_extracts_allergy():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    allergies = [e["resource"] for e in r.fhir_bundle["entry"]
                    if e["resource"]["resourceType"] == "AllergyIntolerance"]
    assert len(allergies) == 1
    assert "Penicillin" in allergies[0]["code"]["text"]


def test_hl7v2_a03_marks_encounter_finished():
    msg = _SAMPLE_ADT_A01.replace("ADT^A01", "ADT^A03")
    r = parse_hl7v2_adt(msg)
    encounter = next(e["resource"] for e in r.fhir_bundle["entry"]
                         if e["resource"]["resourceType"] == "Encounter")
    assert encounter["status"] == "finished"


def test_hl7v2_inpatient_class_code_mapped():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    encounter = next(e["resource"] for e in r.fhir_bundle["entry"]
                         if e["resource"]["resourceType"] == "Encounter")
    assert encounter["class"]["code"] == "IMP"


def test_hl7v2_unknown_event_marked_unknown():
    msg = _SAMPLE_ADT_A01.replace("ADT^A01", "ADT^A99")
    r = parse_hl7v2_adt(msg)
    assert r.message_type == "unknown"


def test_hl7v2_n_resources_consistent():
    r = parse_hl7v2_adt(_SAMPLE_ADT_A01)
    assert r.n_resources == len(r.fhir_bundle["entry"])


# ───────────────────────────────────────────────────────────────────
# EHR-2 -- C-CDA parser
# ───────────────────────────────────────────────────────────────────

_SAMPLE_CCDA = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalDocument xmlns="urn:hl7-org:v3"
                  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <title>Continuity of Care Document</title>
  <recordTarget>
    <patientRole>
      <patient>
        <name>
          <given>Jane</given>
          <family>Doe</family>
        </name>
        <administrativeGenderCode code="F" />
        <birthTime value="19550101" />
      </patient>
    </patientRole>
  </recordTarget>
  <component>
    <structuredBody>
      <component>
        <section>
          <title>Problem List</title>
          <entry>
            <observation>
              <value xsi:type="CD"
                     code="I50.9"
                     codeSystem="2.16.840.1.113883.6.90"
                     displayName="Heart failure" />
            </observation>
          </entry>
          <entry>
            <observation>
              <value xsi:type="CD"
                     code="E11.9"
                     codeSystem="2.16.840.1.113883.6.90"
                     displayName="Type 2 diabetes" />
            </observation>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <title>Medications</title>
          <entry>
            <substanceAdministration>
              <consumable>
                <manufacturedProduct>
                  <manufacturedMaterial>
                    <code code="11289"
                          codeSystem="2.16.840.1.113883.6.88"
                          displayName="warfarin" />
                  </manufacturedMaterial>
                </manufacturedProduct>
              </consumable>
            </substanceAdministration>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <title>Allergies</title>
          <entry>
            <act>
              <entryRelationship>
                <observation>
                  <participant>
                    <participantRole>
                      <playingEntity>
                        <code displayName="Penicillin" />
                      </playingEntity>
                    </participantRole>
                  </participant>
                </observation>
              </entryRelationship>
            </act>
          </entry>
        </section>
      </component>
    </structuredBody>
  </component>
</ClinicalDocument>"""


def test_ccda_invalid_input_raises():
    with pytest.raises(ValueError, match="non-empty"):
        parse_ccda_document("")


def test_ccda_invalid_xml_raises():
    with pytest.raises(ValueError, match="Invalid XML"):
        parse_ccda_document("<broken>")


def test_ccda_extracts_patient_demographics():
    r = parse_ccda_document(_SAMPLE_CCDA)
    patient = next(e["resource"] for e in r.fhir_bundle["entry"]
                       if e["resource"]["resourceType"] == "Patient")
    assert patient["birthDate"] == "1955-01-01"
    assert patient["gender"] == "female"
    assert patient["name"][0]["family"] == "Doe"


def test_ccda_extracts_problem_list():
    r = parse_ccda_document(_SAMPLE_CCDA)
    conds = [e["resource"] for e in r.fhir_bundle["entry"]
                if e["resource"]["resourceType"] == "Condition"]
    assert len(conds) == 2
    codes = {c["code"]["coding"][0]["code"] for c in conds
                if c["code"]["coding"]}
    assert "I50.9" in codes
    assert "E11.9" in codes


def test_ccda_extracts_medications():
    r = parse_ccda_document(_SAMPLE_CCDA)
    meds = [e["resource"] for e in r.fhir_bundle["entry"]
              if e["resource"]["resourceType"] == "MedicationRequest"]
    assert len(meds) == 1
    assert "warfarin" in meds[0]["medicationCodeableConcept"]["text"].lower()


def test_ccda_extracts_allergies():
    r = parse_ccda_document(_SAMPLE_CCDA)
    allergies = [e["resource"] for e in r.fhir_bundle["entry"]
                    if e["resource"]["resourceType"] == "AllergyIntolerance"]
    assert len(allergies) >= 1
    assert any("Penicillin" in (a["code"]["text"] or "")
                  for a in allergies)


def test_ccda_sections_extracted_listed():
    r = parse_ccda_document(_SAMPLE_CCDA)
    expected = {"problem_list", "medications", "allergies"}
    assert expected <= set(r.sections_extracted)


def test_ccda_document_type_extracted():
    r = parse_ccda_document(_SAMPLE_CCDA)
    assert r.document_type and "Continuity" in r.document_type


# ───────────────────────────────────────────────────────────────────
# EHR-3 -- SDOH ingestion
# ───────────────────────────────────────────────────────────────────

def _bundle_with(*, zip_code: str | None = None,
                    z_codes: list[str] | None = None) -> dict:
    bundle: dict = {"resourceType": "Bundle", "type": "collection",
                       "entry": []}
    patient: dict = {"resourceType": "Patient", "id": "pt-1"}
    if zip_code:
        patient["address"] = [{"postalCode": zip_code}]
    bundle["entry"].append({"resource": patient})
    for code in z_codes or []:
        bundle["entry"].append({"resource": {
            "resourceType": "Condition",
            "code": {"coding": [{
                "system": "http://hl7.org/fhir/sid/icd-10-cm",
                "code": code,
            }]}}})
    return bundle


def test_sdoh_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a dict"):
        compute_sdoh_score("nope")  # type: ignore[arg-type]


def test_sdoh_no_zcodes_no_zip_low_burden():
    r = compute_sdoh_score(_bundle_with())
    assert r.composite_sdoh_burden_score == 0.0
    assert r.z_codes_present == []


def test_sdoh_zcodes_increment_burden():
    r = compute_sdoh_score(_bundle_with(z_codes=["Z59.0"]))
    assert "Z59" in r.z_codes_present
    assert "housing_and_economic" in r.z_code_categories
    assert r.composite_sdoh_burden_score > 0.0


def test_sdoh_svi_lookup_for_known_zip():
    r = compute_sdoh_score(_bundle_with(zip_code="60628"))
    assert r.svi_overall_percentile is not None
    assert r.svi_overall_percentile > 0.5  # high-vulnerability zip
    assert r.svi_themes


def test_sdoh_svi_for_low_vulnerability_zip():
    r = compute_sdoh_score(_bundle_with(zip_code="90210"))
    assert r.svi_overall_percentile is not None
    assert r.svi_overall_percentile < 0.3


def test_sdoh_unknown_zip_no_svi():
    r = compute_sdoh_score(_bundle_with(zip_code="00000"))
    assert r.svi_overall_percentile is None
    assert r.svi_themes == {}


def test_sdoh_combined_zcodes_and_svi():
    r = compute_sdoh_score(_bundle_with(
        zip_code="60628",
        z_codes=["Z59.0", "Z63.0", "Z55.9"],
    ))
    assert len(r.z_codes_present) == 3
    assert r.composite_sdoh_burden_score > 0.5


def test_sdoh_composite_in_unit_range():
    r = compute_sdoh_score(_bundle_with(
        zip_code="60628",
        z_codes=[f"Z{55+i}.0" for i in range(10)],
    ))
    assert 0.0 <= r.composite_sdoh_burden_score <= 1.0


def test_sdoh_fairness_modifier_matches_composite():
    r = compute_sdoh_score(_bundle_with(zip_code="10027"))
    assert r.fairness_audit_modifier == r.composite_sdoh_burden_score


def test_sdoh_zip_extraction_handles_zip_plus_4():
    r = compute_sdoh_score(_bundle_with(zip_code="60628-1234"))
    assert r.patient_zip == "60628"
