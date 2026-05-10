"""healthcare.compute_hl7v2_message_parse / compute_ccda_document_parse
-- Phase 14.13 O1 legacy-EHR ingestion bundle.

Two pure-stdlib parsers that handle the real-world legacy formats
which still dominate EHR integration:

  - **HL7 v2.x** pipe-delimited messages (PID, PV1, ORU/OBR/OBX,
    AL1, RXA). The parser surfaces Patient + Encounter + Allergies
    + Observations + Medications as FHIR-equivalent dicts.
  - **C-CDA / CCD** XML documents -- parse the standard sections
    (Patient, Allergies, Medications, Problems) into the same
    FHIR-equivalent dict shape so the downstream tools can consume
    either source uniformly.

Both tools are pure-deterministic -- no schema validation against the
full HL7 v2 spec, but enough to unlock the standard chart-summary
ingestion path in TrustedRisk's clinical bundles.

References:
- HL7 v2.x Standard, Chapter 3 (Patient Administration).
- HL7 C-CDA R2.1 Implementation Guide.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any, Literal

from pydantic import BaseModel, Field


class HL7v2ParseReport(BaseModel):
    message_type: str
    n_segments: int = Field(ge=0)
    patient: dict[str, Any] = Field(default_factory=dict)
    encounter: dict[str, Any] = Field(default_factory=dict)
    allergies: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    medications: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


class CCDAParseReport(BaseModel):
    document_template_id: str | None = None
    n_sections: int = Field(ge=0)
    patient: dict[str, Any] = Field(default_factory=dict)
    allergies: list[dict[str, Any]] = Field(default_factory=list)
    medications: list[dict[str, Any]] = Field(default_factory=list)
    problems: list[dict[str, Any]] = Field(default_factory=list)
    rationale: str
    references: list[str] = Field(default_factory=list)
    abstain_recommended: bool = False
    abstain_reason: str | None = None


# ─────────────────────────────────────────────────────────────────────
# HL7 v2 parser
# ─────────────────────────────────────────────────────────────────────


def _split_segment(seg: str) -> list[str]:
    return seg.split("|")


def _parse_pid(fields: list[str]) -> dict[str, Any]:
    """PID -- Patient Identification."""
    out: dict[str, Any] = {}
    if len(fields) > 3 and fields[3]:
        out["patient_id"] = fields[3].split("^")[0]
    if len(fields) > 5 and fields[5]:
        name_parts = fields[5].split("^")
        family = name_parts[0] if len(name_parts) > 0 else ""
        given = name_parts[1] if len(name_parts) > 1 else ""
        out["name"] = {"family": family, "given": [given] if given else []}
    if len(fields) > 7 and fields[7]:
        out["birthDate"] = _hl7_date_to_iso(fields[7])
    if len(fields) > 8 and fields[8]:
        out["gender"] = {
            "M": "male", "F": "female",
        }.get(fields[8].upper(), "unknown")
    return out


def _parse_pv1(fields: list[str]) -> dict[str, Any]:
    """PV1 -- Patient Visit."""
    out: dict[str, Any] = {}
    if len(fields) > 2 and fields[2]:
        out["class"] = {
            "I": "inpatient", "O": "outpatient",
            "E": "emergency",
        }.get(fields[2].upper(), "unknown")
    if len(fields) > 3 and fields[3]:
        out["assigned_location"] = fields[3].split("^")[0]
    if len(fields) > 44 and fields[44]:
        out["admit_datetime"] = _hl7_date_to_iso(fields[44])
    return out


def _parse_obx(fields: list[str]) -> dict[str, Any]:
    """OBX -- Observation/Result."""
    out: dict[str, Any] = {}
    if len(fields) > 3 and fields[3]:
        code_parts = fields[3].split("^")
        out["code"] = code_parts[0]
        if len(code_parts) > 1:
            out["display"] = code_parts[1]
    if len(fields) > 5 and fields[5]:
        try:
            out["value"] = float(fields[5])
        except (ValueError, TypeError):
            out["value"] = fields[5]
    if len(fields) > 6 and fields[6]:
        out["unit"] = fields[6]
    return out


def _parse_al1(fields: list[str]) -> dict[str, Any]:
    """AL1 -- Allergy."""
    out: dict[str, Any] = {}
    if len(fields) > 3 and fields[3]:
        parts = fields[3].split("^")
        out["substance"] = parts[1] if len(parts) > 1 else parts[0]
        out["substance_code"] = parts[0]
    if len(fields) > 4 and fields[4]:
        out["severity"] = fields[4]
    return out


def _parse_rxa(fields: list[str]) -> dict[str, Any]:
    """RXA -- Pharmacy Administration (medication)."""
    out: dict[str, Any] = {}
    if len(fields) > 5 and fields[5]:
        parts = fields[5].split("^")
        out["medication_code"] = parts[0]
        if len(parts) > 1:
            out["medication_name"] = parts[1]
    if len(fields) > 6 and fields[6]:
        try:
            out["dose_amount"] = float(fields[6])
        except (ValueError, TypeError):
            pass
    if len(fields) > 7 and fields[7]:
        out["dose_unit"] = fields[7]
    return out


def _hl7_date_to_iso(raw: str) -> str:
    raw = (raw or "").strip()
    if len(raw) >= 8:
        date = f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
        if len(raw) >= 14:
            return f"{date}T{raw[8:10]}:{raw[10:12]}:{raw[12:14]}Z"
        return date
    return raw


async def compute_hl7v2_message_parse(
    message: str,
) -> HL7v2ParseReport:
    """Parse an HL7 v2.x pipe-delimited message into a FHIR-equivalent
    dict shape.

    Recognised segments: MSH, PID, PV1, AL1, OBX, RXA. Unknown
    segments are counted but ignored.
    """
    if not message or "|" not in message:
        raise ValueError("message must be a non-empty HL7 v2 string")
    # HL7 segment delimiter is CR (\r); be liberal and accept \n too
    raw_segments = re.split(r"[\r\n]+", message.strip())
    raw_segments = [s for s in raw_segments if s.strip()]
    if not raw_segments:
        raise ValueError("no segments found")

    msh = raw_segments[0]
    if not msh.startswith("MSH"):
        raise ValueError("first segment must be MSH")
    msh_fields = _split_segment(msh)
    msg_type = msh_fields[8] if len(msh_fields) > 8 else "UNKNOWN"

    patient: dict[str, Any] = {}
    encounter: dict[str, Any] = {}
    allergies: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    medications: list[dict[str, Any]] = []

    for seg in raw_segments[1:]:
        fields = _split_segment(seg)
        seg_id = fields[0]
        if seg_id == "PID":
            patient = _parse_pid(fields)
        elif seg_id == "PV1":
            encounter = _parse_pv1(fields)
        elif seg_id == "AL1":
            allergies.append(_parse_al1(fields))
        elif seg_id == "OBX":
            observations.append(_parse_obx(fields))
        elif seg_id == "RXA":
            medications.append(_parse_rxa(fields))

    rationale = (
        f"HL7 v2 message {msg_type} -- {len(raw_segments)} segments, "
        f"{len(observations)} observation(s), "
        f"{len(allergies)} allergy(ies), "
        f"{len(medications)} medication(s)."
    )
    return HL7v2ParseReport(
        message_type=msg_type, n_segments=len(raw_segments),
        patient=patient, encounter=encounter,
        allergies=allergies, observations=observations,
        medications=medications,
        rationale=rationale,
        references=["HL7 v2.x Standard, Chapter 3."],
    )


# ─────────────────────────────────────────────────────────────────────
# C-CDA parser
# ─────────────────────────────────────────────────────────────────────

_CDA_NS = "urn:hl7-org:v3"
_CDA_NS_TAG = f"{{{_CDA_NS}}}"


def _strip_ns(elem: ET.Element) -> str:
    return elem.tag.split("}", 1)[1] if "}" in elem.tag else elem.tag


def _findall_ns(elem: ET.Element, path: str) -> list[ET.Element]:
    """ElementTree.findall with an `urn:hl7-org:v3` namespace prefix."""
    parts = path.split("/")
    fully_qualified = "/".join(
        f"{_CDA_NS_TAG}{p}" if p and not p.startswith(".") else p
        for p in parts
    )
    return list(elem.findall(fully_qualified))


def _ccda_patient(root: ET.Element) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pat in _findall_ns(root, ".//recordTarget/patientRole/patient"):
        name_el = pat.find(f"{_CDA_NS_TAG}name")
        if name_el is not None:
            family_el = name_el.find(f"{_CDA_NS_TAG}family")
            given_el = name_el.find(f"{_CDA_NS_TAG}given")
            family = (family_el.text or "").strip() if family_el is not None \
                else ""
            given = (given_el.text or "").strip() if given_el is not None \
                else ""
            out["name"] = {
                "family": family, "given": [given] if given else [],
            }
        gender_el = pat.find(f"{_CDA_NS_TAG}administrativeGenderCode")
        if gender_el is not None:
            code = gender_el.attrib.get("code", "").upper()
            out["gender"] = {
                "M": "male", "F": "female",
            }.get(code, "unknown")
        bd = pat.find(f"{_CDA_NS_TAG}birthTime")
        if bd is not None:
            raw = bd.attrib.get("value", "")
            if raw:
                out["birthDate"] = _hl7_date_to_iso(raw)
        break
    pid = root.find(f".//{_CDA_NS_TAG}recordTarget/{_CDA_NS_TAG}patientRole"
                    f"/{_CDA_NS_TAG}id")
    if pid is not None:
        out["patient_id"] = pid.attrib.get("extension") or pid.attrib.get("root")
    return out


def _ccda_section_entries(
    root: ET.Element, section_template_id: str,
) -> list[ET.Element]:
    """Find every <entry> under sections matching the templateId."""
    out: list[ET.Element] = []
    for section in root.iter(f"{_CDA_NS_TAG}section"):
        for tid in section.findall(f"{_CDA_NS_TAG}templateId"):
            if tid.attrib.get("root") == section_template_id:
                for entry in section.findall(f"{_CDA_NS_TAG}entry"):
                    out.append(entry)
                break
    return out


def _ccda_displays(elem: ET.Element) -> tuple[str, str]:
    """Pull (code, display) from the first <code> child."""
    code_el = elem.find(f".//{_CDA_NS_TAG}code")
    if code_el is None:
        return "", ""
    return (
        code_el.attrib.get("code", ""),
        code_el.attrib.get("displayName", ""),
    )


_ALLERGY_TID = "2.16.840.1.113883.10.20.22.2.6.1"
_MED_TID = "2.16.840.1.113883.10.20.22.2.1.1"
_PROBLEM_TID = "2.16.840.1.113883.10.20.22.2.5.1"


async def compute_ccda_document_parse(
    xml_text: str,
) -> CCDAParseReport:
    """Parse a C-CDA / CCD XML document into a FHIR-equivalent dict.

    Recognised sections (by templateId root):
      - 2.16.840.1.113883.10.20.22.2.6.1 -> Allergies
      - 2.16.840.1.113883.10.20.22.2.1.1 -> Medications
      - 2.16.840.1.113883.10.20.22.2.5.1 -> Problems
    """
    if not xml_text or "<" not in xml_text:
        raise ValueError("xml_text must be a non-empty XML string")
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"malformed XML: {exc}") from exc

    doc_template = ""
    for tid in root.findall(f"{_CDA_NS_TAG}templateId"):
        if tid.attrib.get("root"):
            doc_template = tid.attrib["root"]
            break

    patient = _ccda_patient(root)

    allergies: list[dict[str, Any]] = []
    for entry in _ccda_section_entries(root, _ALLERGY_TID):
        code, display = _ccda_displays(entry)
        allergies.append({"substance_code": code, "substance": display})

    medications: list[dict[str, Any]] = []
    for entry in _ccda_section_entries(root, _MED_TID):
        code, display = _ccda_displays(entry)
        medications.append({
            "medication_code": code, "medication_name": display,
        })

    problems: list[dict[str, Any]] = []
    for entry in _ccda_section_entries(root, _PROBLEM_TID):
        code, display = _ccda_displays(entry)
        problems.append({"code": code, "display": display})

    n_sections = len(list(root.iter(f"{_CDA_NS_TAG}section")))

    rationale = (
        f"C-CDA document -- {n_sections} section(s); "
        f"{len(allergies)} allergy(ies), "
        f"{len(medications)} medication(s), "
        f"{len(problems)} problem(s)."
    )
    return CCDAParseReport(
        document_template_id=doc_template or None,
        n_sections=n_sections,
        patient=patient, allergies=allergies,
        medications=medications, problems=problems,
        rationale=rationale,
        references=["HL7 C-CDA R2.1 Implementation Guide."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_hl7v2_message_parse)
    mcp.tool()(compute_ccda_document_parse)
