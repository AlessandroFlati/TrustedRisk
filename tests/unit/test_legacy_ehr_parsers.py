"""Phase 14.13 O1 -- HL7 v2 + C-CDA parser tests."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.legacy_ehr_parsers import (
    compute_ccda_document_parse, compute_hl7v2_message_parse,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# HL7 v2 -- header / patient / encounter
# ─────────────────────────────────────────────────────────────────────

_HL7_BASE = (
    "MSH|^~\\&|EPIC|HOSP|RECV|FACILITY|20260430085900||ADT^A01|MSGID0001|P|2.5\r"
    "PID|1||MRN0001234^^^HOSP^MR||DOE^JOHN^A||19601115|M\r"
    "PV1|1|I|3W^301^A||||DR123^SMITH^J|||MED|||||||||V123||||||||||"
    "||||||||||||||||20260430083000\r"
)


def test_hl7v2_parses_message_type():
    res = _run(compute_hl7v2_message_parse(_HL7_BASE))
    assert res.message_type == "ADT^A01"
    assert res.n_segments == 3


def test_hl7v2_parses_patient_demographics():
    res = _run(compute_hl7v2_message_parse(_HL7_BASE))
    assert res.patient["patient_id"] == "MRN0001234"
    assert res.patient["name"]["family"] == "DOE"
    assert res.patient["name"]["given"] == ["JOHN"]
    assert res.patient["gender"] == "male"
    assert res.patient["birthDate"] == "1960-11-15"


def test_hl7v2_parses_encounter_class():
    res = _run(compute_hl7v2_message_parse(_HL7_BASE))
    assert res.encounter["class"] == "inpatient"


# ─────────────────────────────────────────────────────────────────────
# HL7 v2 -- observations / allergies / medications
# ─────────────────────────────────────────────────────────────────────

def test_hl7v2_parses_obx_observation():
    msg = _HL7_BASE + (
        "OBX|1|NM|718-7^Hemoglobin^LN||9.8|g/dL|13.0-17.0|L|||F\r"
    )
    res = _run(compute_hl7v2_message_parse(msg))
    assert len(res.observations) == 1
    obs = res.observations[0]
    assert obs["code"] == "718-7"
    assert obs["display"] == "Hemoglobin"
    assert obs["value"] == 9.8
    assert obs["unit"] == "g/dL"


def test_hl7v2_parses_al1_allergy():
    msg = _HL7_BASE + "AL1|1|DA|7980^Penicillin^RXN|SV|hives\r"
    res = _run(compute_hl7v2_message_parse(msg))
    assert len(res.allergies) == 1
    allergy = res.allergies[0]
    assert allergy["substance_code"] == "7980"
    assert allergy["substance"] == "Penicillin"
    assert allergy["severity"] == "SV"


def test_hl7v2_parses_rxa_medication():
    msg = _HL7_BASE + (
        "RXA|0|1|20260430|20260430|11289^Warfarin^RXN|5|mg|||OP|||||||||\r"
    )
    res = _run(compute_hl7v2_message_parse(msg))
    assert len(res.medications) == 1
    med = res.medications[0]
    assert med["medication_code"] == "11289"
    assert med["medication_name"] == "Warfarin"
    assert med["dose_amount"] == 5.0
    assert med["dose_unit"] == "mg"


def test_hl7v2_unknown_segment_counted_but_ignored():
    msg = _HL7_BASE + "ZZZ|extra|garbage|payload\r"
    res = _run(compute_hl7v2_message_parse(msg))
    assert res.n_segments == 4
    assert res.observations == []


def test_hl7v2_rejects_empty_message():
    with pytest.raises(ValueError):
        _run(compute_hl7v2_message_parse(""))


def test_hl7v2_rejects_non_msh_first_segment():
    with pytest.raises(ValueError):
        _run(compute_hl7v2_message_parse("PID|1||X^^^H^MR||DOE^J\r"))


# ─────────────────────────────────────────────────────────────────────
# C-CDA -- patient + sections
# ─────────────────────────────────────────────────────────────────────

_CCD_XML = """<?xml version="1.0"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <templateId root="2.16.840.1.113883.10.20.22.1.2"/>
  <recordTarget>
    <patientRole>
      <id extension="MRN-9876"/>
      <patient>
        <name>
          <family>SMITH</family>
          <given>JANE</given>
        </name>
        <administrativeGenderCode code="F"/>
        <birthTime value="19720315"/>
      </patient>
    </patientRole>
  </recordTarget>
  <component>
    <structuredBody>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.6.1"/>
          <entry>
            <observation>
              <code code="7980" displayName="Penicillin"/>
            </observation>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.1.1"/>
          <entry>
            <substanceAdministration>
              <code code="11289" displayName="Warfarin 5 mg"/>
            </substanceAdministration>
          </entry>
        </section>
      </component>
      <component>
        <section>
          <templateId root="2.16.840.1.113883.10.20.22.2.5.1"/>
          <entry>
            <act>
              <code code="I50.9" displayName="Heart failure, unspecified"/>
            </act>
          </entry>
        </section>
      </component>
    </structuredBody>
  </component>
</ClinicalDocument>
"""


def test_ccda_parses_document_template_id():
    res = _run(compute_ccda_document_parse(_CCD_XML))
    assert res.document_template_id == "2.16.840.1.113883.10.20.22.1.2"


def test_ccda_parses_patient_demographics():
    res = _run(compute_ccda_document_parse(_CCD_XML))
    assert res.patient["patient_id"] == "MRN-9876"
    assert res.patient["name"]["family"] == "SMITH"
    assert res.patient["name"]["given"] == ["JANE"]
    assert res.patient["gender"] == "female"
    assert res.patient["birthDate"] == "1972-03-15"


def test_ccda_parses_three_sections():
    res = _run(compute_ccda_document_parse(_CCD_XML))
    assert res.n_sections == 3


def test_ccda_parses_allergy_entry():
    res = _run(compute_ccda_document_parse(_CCD_XML))
    assert len(res.allergies) == 1
    assert res.allergies[0]["substance_code"] == "7980"
    assert res.allergies[0]["substance"] == "Penicillin"


def test_ccda_parses_medication_entry():
    res = _run(compute_ccda_document_parse(_CCD_XML))
    assert len(res.medications) == 1
    assert res.medications[0]["medication_code"] == "11289"
    assert res.medications[0]["medication_name"] == "Warfarin 5 mg"


def test_ccda_parses_problem_entry():
    res = _run(compute_ccda_document_parse(_CCD_XML))
    assert len(res.problems) == 1
    assert res.problems[0]["code"] == "I50.9"
    assert res.problems[0]["display"] == "Heart failure, unspecified"


def test_ccda_rejects_empty_input():
    with pytest.raises(ValueError):
        _run(compute_ccda_document_parse(""))


def test_ccda_rejects_malformed_xml():
    with pytest.raises(ValueError):
        _run(compute_ccda_document_parse("<not><closed>"))


def test_ccda_unknown_section_skipped():
    xml = """<?xml version="1.0"?>
<ClinicalDocument xmlns="urn:hl7-org:v3">
  <recordTarget><patientRole><id extension="X"/>
    <patient><name><family>X</family></name></patient>
  </patientRole></recordTarget>
  <component><structuredBody><component>
    <section>
      <templateId root="9.9.9.9.NOT.A.REAL.TID"/>
      <entry><observation><code code="ZZZ" displayName="unknown"/></observation></entry>
    </section>
  </component></structuredBody></component>
</ClinicalDocument>
"""
    res = _run(compute_ccda_document_parse(xml))
    assert res.allergies == []
    assert res.medications == []
    assert res.problems == []
    assert res.n_sections == 1


# ─────────────────────────────────────────────────────────────────────
# Cross-format equivalence
# ─────────────────────────────────────────────────────────────────────

def test_hl7v2_and_ccda_produce_same_patient_shape_keys():
    """Both parsers yield a patient dict with the same set of canonical
    keys when fully populated -- this is the FHIR-equivalent uniformity
    that downstream bundles depend on."""
    hl7_res = _run(compute_hl7v2_message_parse(_HL7_BASE))
    ccd_res = _run(compute_ccda_document_parse(_CCD_XML))
    common = {"patient_id", "name", "gender", "birthDate"}
    assert common <= set(hl7_res.patient)
    assert common <= set(ccd_res.patient)
