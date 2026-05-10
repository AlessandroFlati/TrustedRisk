"""EHR-1/2/3 -- Legacy-system integrations.

Three utilities that bridge non-FHIR EHR data into the agent's FHIR-
oriented pipeline:

  EHR-1 parse_hl7v2_adt        -- HL7 v2 ADT (A01/A02/A03/A04/A08) -> FHIR
  EHR-2 parse_ccda_document    -- Consolidated CDA XML -> FHIR Bundle
  EHR-3 compute_sdoh_score     -- FHIR Z-codes + AHRQ SVI -> SDOH burden

Each is dependency-light (stdlib only -- xml.etree for C-CDA), with no
runtime requirement on a live EHR connection.
"""

from __future__ import annotations

import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

from shared.schemas import (
    CCDAParsedDocument,
    HL7v2ParsedMessage,
    SDOHReport,
)


# ─────────────────────── EHR-1: HL7 v2 ADT parser ───────────────────────

_FIELD_SEP = "|"
_COMP_SEP = "^"


def _split_segments(message: str) -> list[list[str]]:
    """Split an HL7 v2 message into a list of segments, each a list of fields."""
    # HL7 v2 line break is \r in the wire format; some texts use \n
    raw = message.replace("\n", "\r").strip()
    if not raw:
        return []
    segments: list[list[str]] = []
    for line in raw.split("\r"):
        line = line.strip()
        if not line:
            continue
        segments.append(line.split(_FIELD_SEP))
    return segments


def _comp(field: str, idx: int = 0) -> str:
    parts = field.split(_COMP_SEP)
    if idx < len(parts):
        return parts[idx]
    return ""


def _parse_hl7_datetime(s: str) -> str | None:
    """Parse YYYYMMDDHHMMSS[+ZZZZ] HL7 v2 timestamp -> ISO-8601."""
    if not s:
        return None
    s = s.strip()
    # Strip TZ for simplicity in v1
    base = re.sub(r"[+\-]\d{4}$", "", s)
    formats = ["%Y%m%d%H%M%S", "%Y%m%d%H%M",
                  "%Y%m%d", "%Y%m"]
    for fmt in formats:
        try:
            dt = datetime.strptime(base, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    return None


_ADT_EVENT_TO_ENCOUNTER_STATUS = {
    "A01": "in-progress",
    "A02": "in-progress",
    "A03": "finished",
    "A04": "in-progress",
    "A08": "in-progress",
}


def parse_hl7v2_adt(message: str) -> HL7v2ParsedMessage:
    """Parse an HL7 v2 ADT message into a FHIR R4 Bundle.

    Args:
        message: raw HL7 v2 message text (segments separated by \\r or \\n).

    Returns:
        HL7v2ParsedMessage with the FHIR Bundle + per-resource counts.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string.")

    segments = _split_segments(message)
    if not segments:
        raise ValueError("No HL7 segments found in message.")

    msh = segments[0]
    if not msh or msh[0] not in ("MSH", "MSH|"):
        # Some senders put the field separator inside MSH|^~\&
        if msh and msh[0].startswith("MSH"):
            pass
        else:
            raise ValueError(
                f"First segment must be MSH; got {msh[0] if msh else '?'}.")

    sending_app = _comp(msh[2]) if len(msh) > 2 else None
    sending_fac = _comp(msh[3]) if len(msh) > 3 else None
    msg_dt = _parse_hl7_datetime(msh[6]) if len(msh) > 6 else None
    msg_type_field = msh[8] if len(msh) > 8 else ""
    msg_type_parts = msg_type_field.split(_COMP_SEP)
    event_code = msg_type_parts[1].strip() if len(msg_type_parts) > 1 \
        else "unknown"
    if event_code not in _ADT_EVENT_TO_ENCOUNTER_STATUS:
        event_code = "unknown"
    msg_ctrl_id = msh[9] if len(msh) > 9 else None

    parse_warnings: list[str] = []

    fhir_resources: list[dict[str, Any]] = []
    patient_id_internal = uuid.uuid4().hex
    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id_internal,
    }
    encounter: dict[str, Any] = {
        "resourceType": "Encounter",
        "id": uuid.uuid4().hex,
        "status": _ADT_EVENT_TO_ENCOUNTER_STATUS.get(event_code,
                                                          "unknown"),
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                     "code": "AMB"},
        "subject": {"reference": f"Patient/{patient_id_internal}"},
    }

    for seg in segments[1:]:
        if not seg:
            continue
        seg_type = seg[0]
        if seg_type == "PID":
            # PID-3 = patient identifier list, PID-5 = patient name,
            # PID-7 = birth date, PID-8 = sex
            if len(seg) > 3:
                ident = _comp(seg[3], 0)
                if ident:
                    patient.setdefault("identifier", []).append({
                        "system": "urn:hl7-v2:pid-3",
                        "value": ident,
                    })
            if len(seg) > 5:
                family = _comp(seg[5], 0)
                given = _comp(seg[5], 1)
                name_obj: dict[str, Any] = {}
                if family:
                    name_obj["family"] = family
                if given:
                    name_obj["given"] = [given]
                if name_obj:
                    patient["name"] = [name_obj]
            if len(seg) > 7 and seg[7]:
                bd = _parse_hl7_datetime(seg[7])
                if bd:
                    patient["birthDate"] = bd[:10]
            if len(seg) > 8 and seg[8]:
                sex_map = {"M": "male", "F": "female",
                              "O": "other", "U": "unknown"}
                patient["gender"] = sex_map.get(seg[8].strip().upper(),
                                                      "unknown")
        elif seg_type == "PV1":
            # PV1-2 = patient class, PV1-3 = location, PV1-44 = admit dt,
            # PV1-45 = discharge dt
            if len(seg) > 2:
                pclass = seg[2].strip().upper()
                class_map = {"I": "IMP", "O": "AMB",
                                "E": "EMER", "P": "AMB"}
                code = class_map.get(pclass, "AMB")
                encounter["class"] = {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": code,
                }
            period: dict[str, str] = {}
            if len(seg) > 44 and seg[44]:
                start = _parse_hl7_datetime(seg[44])
                if start:
                    period["start"] = start
            if len(seg) > 45 and seg[45]:
                end = _parse_hl7_datetime(seg[45])
                if end:
                    period["end"] = end
            if period:
                encounter["period"] = period
        elif seg_type == "DG1":
            # DG1-3 = diagnosis code (ICD-10), DG1-4 = description
            if len(seg) > 3 and seg[3]:
                code = _comp(seg[3], 0)
                desc = _comp(seg[3], 1) or (
                    seg[4].strip() if len(seg) > 4 else "")
                fhir_resources.append({
                    "resourceType": "Condition",
                    "id": uuid.uuid4().hex,
                    "subject": {"reference": f"Patient/{patient_id_internal}"},
                    "code": {
                        "coding": [{
                            "system": "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": code,
                            "display": desc,
                        }],
                        "text": desc or code,
                    },
                })
        elif seg_type == "AL1":
            # AL1-3 = allergen, AL1-5 = severity
            if len(seg) > 3 and seg[3]:
                allergen_text = _comp(seg[3], 1) or _comp(seg[3], 0)
                fhir_resources.append({
                    "resourceType": "AllergyIntolerance",
                    "id": uuid.uuid4().hex,
                    "patient": {"reference": f"Patient/{patient_id_internal}"},
                    "code": {"text": allergen_text},
                    "clinicalStatus": {"coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/"
                                     "allergyintolerance-clinical",
                        "code": "active"}]},
                })

    fhir_resources.insert(0, encounter)
    fhir_resources.insert(0, patient)

    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [{"resource": r} for r in fhir_resources],
    }

    return HL7v2ParsedMessage(
        message_type=event_code,                     # type: ignore[arg-type]
        sending_application=sending_app,
        sending_facility=sending_fac,
        message_control_id=msg_ctrl_id,
        message_datetime_iso=msg_dt,
        fhir_bundle=bundle,
        n_resources=len(fhir_resources),
        parse_warnings=parse_warnings,
    )


# ─────────────────────── EHR-2: C-CDA parser ───────────────────────

_HL7_NAMESPACES = {"hl7": "urn:hl7-org:v3"}


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _findall_no_ns(elem: ET.Element, name: str) -> list[ET.Element]:
    """Find direct + descendant elements by local-name (namespace-agnostic)."""
    return [e for e in elem.iter() if _strip_ns(e.tag) == name]


def parse_ccda_document(xml_text: str) -> CCDAParsedDocument:
    """Parse a Consolidated CDA (C-CDA R2.1) document into a FHIR Bundle.

    Args:
        xml_text: raw XML string of the C-CDA document.

    Returns:
        CCDAParsedDocument with the FHIR Bundle + sections_extracted list.
    """
    if not isinstance(xml_text, str) or not xml_text.strip():
        raise ValueError("xml_text must be a non-empty string.")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    parse_warnings: list[str] = []
    fhir_resources: list[dict[str, Any]] = []
    sections_extracted: list[str] = []

    # Document title (templateId or title element)
    title_elem = next((e for e in _findall_no_ns(root, "title")), None)
    document_type = title_elem.text.strip() if title_elem is not None \
        and title_elem.text else None

    # Patient
    patient_id = uuid.uuid4().hex
    patient_resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
    }

    record_targets = _findall_no_ns(root, "recordTarget")
    if record_targets:
        rt = record_targets[0]
        # patient/name (given + family)
        for name_elem in _findall_no_ns(rt, "name"):
            given_elems = _findall_no_ns(name_elem, "given")
            family_elems = _findall_no_ns(name_elem, "family")
            name_obj: dict[str, Any] = {}
            if given_elems:
                name_obj["given"] = [g.text for g in given_elems
                                          if g.text]
            if family_elems and family_elems[0].text:
                name_obj["family"] = family_elems[0].text
            if name_obj:
                patient_resource.setdefault("name", []).append(name_obj)
            break
        # patient/birthTime
        bd_elems = _findall_no_ns(rt, "birthTime")
        if bd_elems:
            v = bd_elems[0].get("value", "")
            if len(v) >= 8:
                patient_resource["birthDate"] = (
                    f"{v[:4]}-{v[4:6]}-{v[6:8]}")
        # patient/administrativeGenderCode
        gender_elems = _findall_no_ns(rt, "administrativeGenderCode")
        if gender_elems:
            code = gender_elems[0].get("code", "").upper()
            patient_resource["gender"] = {
                "M": "male", "F": "female", "U": "unknown"
            }.get(code, "unknown")

    fhir_resources.append(patient_resource)

    # Sections -- match by section title (loose) since templateId/code
    # systems vary across vendors.
    sections = _findall_no_ns(root, "section")
    for section in sections:
        title_elem = next(iter(_findall_no_ns(section, "title")), None)
        if title_elem is None or not title_elem.text:
            continue
        title = title_elem.text.strip().lower()

        if "problem" in title or "diagnos" in title:
            sections_extracted.append("problem_list")
            for entry in _findall_no_ns(section, "entry"):
                code_elem = next(iter(_findall_no_ns(entry, "code")), None)
                value_elem = next(iter(_findall_no_ns(entry, "value")), None)
                code_obj = code_elem if code_elem is not None else value_elem
                if code_obj is None:
                    continue
                code_val = code_obj.get("code", "")
                disp = code_obj.get("displayName", "")
                if not (code_val or disp):
                    continue
                fhir_resources.append({
                    "resourceType": "Condition",
                    "id": uuid.uuid4().hex,
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct"
                            if code_obj.get("codeSystem") ==
                            "2.16.840.1.113883.6.96" else
                            "http://hl7.org/fhir/sid/icd-10-cm",
                            "code": code_val,
                            "display": disp,
                        }] if code_val else [],
                        "text": disp or code_val,
                    },
                })

        elif "medication" in title:
            sections_extracted.append("medications")
            for entry in _findall_no_ns(section, "entry"):
                substance = next(
                    iter(_findall_no_ns(entry, "manufacturedMaterial")),
                    None)
                if substance is None:
                    substance = next(
                        iter(_findall_no_ns(entry, "consumable")), None)
                code_elem = (next(iter(_findall_no_ns(substance, "code")),
                                       None)
                               if substance is not None else None)
                if code_elem is None:
                    continue
                disp = code_elem.get("displayName", "")
                code_val = code_elem.get("code", "")
                if not (code_val or disp):
                    continue
                fhir_resources.append({
                    "resourceType": "MedicationRequest",
                    "id": uuid.uuid4().hex,
                    "status": "active",
                    "intent": "order",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "medicationCodeableConcept": {
                        "coding": [{
                            "system": "http://www.nlm.nih.gov/research/"
                                         "umls/rxnorm",
                            "code": code_val,
                            "display": disp,
                        }] if code_val else [],
                        "text": disp or code_val,
                    },
                })

        elif "allerg" in title:
            sections_extracted.append("allergies")
            for entry in _findall_no_ns(section, "entry"):
                substance = next(iter(_findall_no_ns(entry, "playingEntity")),
                                    None)
                if substance is None:
                    continue
                code_elem = next(iter(_findall_no_ns(substance, "code")),
                                    None)
                disp = code_elem.get("displayName", "") if code_elem is not None \
                    else ""
                if not disp:
                    continue
                fhir_resources.append({
                    "resourceType": "AllergyIntolerance",
                    "id": uuid.uuid4().hex,
                    "patient": {"reference": f"Patient/{patient_id}"},
                    "code": {"text": disp},
                    "clinicalStatus": {"coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/"
                                     "allergyintolerance-clinical",
                        "code": "active"}]},
                })

        elif "procedure" in title:
            sections_extracted.append("procedures")
            for entry in _findall_no_ns(section, "entry"):
                code_elem = next(iter(_findall_no_ns(entry, "code")),
                                    None)
                if code_elem is None:
                    continue
                disp = code_elem.get("displayName", "")
                code_val = code_elem.get("code", "")
                if not (code_val or disp):
                    continue
                fhir_resources.append({
                    "resourceType": "Procedure",
                    "id": uuid.uuid4().hex,
                    "status": "completed",
                    "subject": {"reference": f"Patient/{patient_id}"},
                    "code": {
                        "coding": [{
                            "system": "http://snomed.info/sct",
                            "code": code_val,
                            "display": disp,
                        }] if code_val else [],
                        "text": disp or code_val,
                    },
                })

    bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [{"resource": r} for r in fhir_resources],
    }

    return CCDAParsedDocument(
        document_type=document_type,
        patient_id=patient_id,
        fhir_bundle=bundle,
        n_resources=len(fhir_resources),
        sections_extracted=sorted(set(sections_extracted)),
        parse_warnings=parse_warnings,
    )


# ─────────────────────── EHR-3: SDOH ingestion ───────────────────────

_Z_CODE_CATEGORIES: dict[str, str] = {
    "Z55": "education_and_literacy",
    "Z56": "employment_and_work",
    "Z57": "occupational_exposure",
    "Z58": "physical_environment",
    "Z59": "housing_and_economic",
    "Z60": "social_environment",
    "Z62": "upbringing",
    "Z63": "primary_support_group",
    "Z64": "psychosocial_circumstances",
    "Z65": "other_psychosocial",
}


# AHRQ / CDC Social Vulnerability Index -- 4 themes (2020 release).
# Each theme is a 0..1 percentile; overall is the average of the four.
# The embedded table covers a small representative set of US zip codes.
# For production, swap in the full SVI dataset (free download).
_SVI_BY_ZIP: dict[str, dict[str, float]] = {
    "02118": {"socioeconomic": 0.85, "household_composition": 0.55,
                 "minority_status_language": 0.92, "housing_transportation": 0.68},
    "10027": {"socioeconomic": 0.78, "household_composition": 0.50,
                 "minority_status_language": 0.95, "housing_transportation": 0.72},
    "10282": {"socioeconomic": 0.10, "household_composition": 0.18,
                 "minority_status_language": 0.40, "housing_transportation": 0.20},
    "60601": {"socioeconomic": 0.30, "household_composition": 0.35,
                 "minority_status_language": 0.60, "housing_transportation": 0.45},
    "60628": {"socioeconomic": 0.92, "household_composition": 0.70,
                 "minority_status_language": 0.85, "housing_transportation": 0.78},
    "94110": {"socioeconomic": 0.62, "household_composition": 0.45,
                 "minority_status_language": 0.80, "housing_transportation": 0.55},
    "90210": {"socioeconomic": 0.05, "household_composition": 0.10,
                 "minority_status_language": 0.20, "housing_transportation": 0.08},
    "33125": {"socioeconomic": 0.88, "household_composition": 0.65,
                 "minority_status_language": 0.95, "housing_transportation": 0.74},
    "75201": {"socioeconomic": 0.22, "household_composition": 0.30,
                 "minority_status_language": 0.55, "housing_transportation": 0.40},
    "98101": {"socioeconomic": 0.45, "household_composition": 0.40,
                 "minority_status_language": 0.50, "housing_transportation": 0.35},
}


def _z_codes_in_bundle(bundle: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for entry in bundle.get("entry", []) or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict):
            continue
        if (r.get("resourceType") or "").lower() != "condition":
            continue
        cc = r.get("code") or {}
        for coding in cc.get("coding", []) or []:
            sys = (coding.get("system") or "").lower()
            if "icd-10" in sys or "icd10" in sys:
                code = (coding.get("code") or "").strip().upper()
                normalized = code.replace(".", "")[:3]
                if normalized.startswith("Z") and \
                        normalized in _Z_CODE_CATEGORIES:
                    codes.append(normalized)
    return sorted(set(codes))


def _patient_zip(bundle: dict[str, Any]) -> str | None:
    for entry in bundle.get("entry", []) or []:
        r = entry.get("resource") if isinstance(entry, dict) else None
        if not isinstance(r, dict):
            continue
        if (r.get("resourceType") or "").lower() != "patient":
            continue
        for addr in r.get("address", []) or []:
            postal = addr.get("postalCode")
            if isinstance(postal, str) and postal.strip():
                return postal.strip()[:5]
    return None


def compute_sdoh_score(
    fhir_bundle: dict[str, Any],
) -> SDOHReport:
    """Compute an SDOH burden score from FHIR Z-codes + AHRQ SVI by zip."""
    if not isinstance(fhir_bundle, dict):
        raise ValueError("fhir_bundle must be a dict.")

    z_codes = _z_codes_in_bundle(fhir_bundle)
    z_categories = sorted({_Z_CODE_CATEGORIES[c] for c in z_codes})
    zip_code = _patient_zip(fhir_bundle)
    svi = _SVI_BY_ZIP.get(zip_code) if zip_code else None
    svi_overall = (sum(svi.values()) / 4.0) if svi else None
    svi_themes = svi or {}

    # Composite: average of (Z-code burden, SVI overall percentile).
    z_burden = min(1.0, len(z_codes) / 5.0)   # cap: 5+ Z-codes -> 1.0
    parts = []
    parts.append(z_burden)
    if svi_overall is not None:
        parts.append(svi_overall)
    composite = sum(parts) / len(parts) if parts else 0.0
    fairness_modifier = composite

    rationale = (
        f"Patient zip {zip_code or 'unknown'}, {len(z_codes)} SDOH Z-code(s) "
        f"({', '.join(z_categories) or 'none'}). "
        f"SVI overall = "
        + (f"{svi_overall:.2f}" if svi_overall is not None else "n/a")
        + f". Composite SDOH burden {composite:.3f}."
    )

    return SDOHReport(
        patient_zip=zip_code,
        z_codes_present=z_codes,
        z_code_categories=z_categories,
        svi_overall_percentile=(round(svi_overall, 4)
                                       if svi_overall is not None else None),
        svi_themes=svi_themes,
        composite_sdoh_burden_score=round(composite, 4),
        fairness_audit_modifier=round(fairness_modifier, 4),
        rationale=rationale,
    )
