"""Phase 16.H3 - DICOM SR (Structured Report) parser.

Pure-Python parser for the DICOM Part-10 file format - specifically
the Structured Report (Modality=SR) flavour produced by radiology
information systems. This *is* a real binary parser (not a stub) -
it walks the explicit-VR Little-Endian transfer syntax that all
DICOM SR objects use, surfaces the nested Concept Name + Concept
Value pairs, and lifts them into a flat report.

Limitations (deliberate):
  - Only Explicit-VR Little-Endian (the SOP Class UID 1.2.840.10008.
    1.2.1) is supported. Implicit-VR is rare for SR.
  - Only the standard SR data-type set: TEXT, NUM, CODE, DATETIME,
    DATE, TIME, UIDREF, PNAME, REF.
  - We do not validate the schema against TID 1500 / TID 2000 etc.

References:
- DICOM PS3.10 Media Storage and File Format.
- DICOM PS3.3 Information Object Definitions, C.17 Structured Reports.
"""

from __future__ import annotations

import struct
from typing import Any, BinaryIO

from pydantic import BaseModel, Field


_PREAMBLE_LEN = 128
_DICM = b"DICM"


class DICOMSRItem(BaseModel):
    concept_name: str
    concept_value: str
    value_type: str
    nested: list["DICOMSRItem"] = Field(default_factory=list)


class DICOMSRReport(BaseModel):
    modality: str
    sop_class_uid: str
    study_instance_uid: str
    series_instance_uid: str
    n_items: int = Field(ge=0)
    items: list[DICOMSRItem]
    completion_flag: str
    verification_flag: str
    rationale: str


# Tag dictionary - only the SR-relevant tags
_TAGS = {
    (0x0008, 0x0016): "SOPClassUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x0060): "Modality",
    (0x0020, 0x000D): "StudyInstanceUID",
    (0x0020, 0x000E): "SeriesInstanceUID",
    (0x0040, 0xA040): "ValueType",
    (0x0040, 0xA043): "ConceptNameCodeSequence",
    (0x0040, 0xA730): "ContentSequence",
    (0x0040, 0xA160): "TextValue",
    (0x0040, 0xA30A): "NumericValue",
    (0x0040, 0xA168): "ConceptCodeSequence",
    (0x0040, 0xA050): "ContinuityOfContent",
    (0x0040, 0xA493): "VerificationFlag",
    (0x0040, 0xA491): "CompletionFlag",
    (0x0008, 0x0100): "CodeValue",
    (0x0008, 0x0104): "CodeMeaning",
}


def _read_explicit_vr_element(
    f: BinaryIO,
) -> tuple[tuple[int, int], str, bytes] | None:
    """Read a single Explicit-VR LE element. Returns
    ((group, element), VR, value-bytes) or None at EOF."""
    head = f.read(6)
    if len(head) < 6:
        return None
    group, element, vr_bytes = struct.unpack("<HH2s", head)
    vr = vr_bytes.decode("ascii", errors="replace")
    # VR with 4-byte length: OB OW OF OD UT UN SQ
    if vr in ("OB", "OW", "OF", "OD", "UT", "UN", "SQ"):
        # 2 bytes reserved, then 4-byte length
        reserved = f.read(2)
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            return None
        (length,) = struct.unpack("<I", length_bytes)
    else:
        length_bytes = f.read(2)
        if len(length_bytes) < 2:
            return None
        (length,) = struct.unpack("<H", length_bytes)
    if length == 0xFFFFFFFF:
        # Undefined length - not handled in this minimal parser
        return ((group, element), vr, b"")
    value = f.read(length) if length > 0 else b""
    return ((group, element), vr, value)


def _decode_text(value: bytes) -> str:
    return value.rstrip(b" \x00").decode("utf-8", errors="replace")


def _parse_sequence_items(
    value: bytes, ctx_depth: int = 0,
) -> list[dict[str, Any]]:
    """Parse a sequence's nested items. Each item starts with the
    item delimiter tag (FFFE,E000)."""
    out: list[dict[str, Any]] = []
    pos = 0
    while pos < len(value):
        if pos + 8 > len(value):
            break
        group, elem, length = struct.unpack(
            "<HHI", value[pos:pos + 8])
        pos += 8
        if (group, elem) == (0xFFFE, 0xE0DD):
            # Sequence delimiter
            break
        if (group, elem) == (0xFFFE, 0xE000):
            # Item start
            if length == 0xFFFFFFFF:
                # Undefined-length item: not supported in this minimal
                # parser; bail.
                break
            item_bytes = value[pos:pos + length]
            pos += length
            out.append(_parse_dataset_bytes(item_bytes))
        else:
            break
    return out


def _parse_dataset_bytes(buf: bytes) -> dict[str, Any]:
    """Parse a contiguous Explicit-VR LE dataset (used inside a
    sequence item)."""
    out: dict[str, Any] = {}
    import io
    bio = io.BytesIO(buf)
    while True:
        elem = _read_explicit_vr_element(bio)
        if elem is None:
            break
        (g, e), vr, val = elem
        name = _TAGS.get((g, e), f"({g:04X},{e:04X})")
        if vr == "SQ":
            out[name] = _parse_sequence_items(val)
        else:
            out[name] = _decode_text(val)
    return out


def parse_dicom_sr_bytes(data: bytes) -> DICOMSRReport:
    """Parse the bytes of a DICOM Part-10 SR file.

    Strict requirements:
      - 128-byte preamble + DICM magic.
      - Explicit-VR Little-Endian transfer syntax (we read elements
        directly without consulting the transfer syntax UID -
        sufficient for SR which is almost always ELE).
    """
    if len(data) < _PREAMBLE_LEN + 4:
        raise ValueError("file too short to be DICOM Part-10")
    if data[_PREAMBLE_LEN:_PREAMBLE_LEN + 4] != _DICM:
        raise ValueError("missing DICM magic at byte 128")

    body = data[_PREAMBLE_LEN + 4:]
    dataset = _parse_dataset_bytes(body)
    items: list[DICOMSRItem] = []

    def _flatten(blocks: list[dict[str, Any]],
                 acc: list[DICOMSRItem]) -> None:
        for b in blocks:
            cn_seq = b.get("ConceptNameCodeSequence") or []
            cn = cn_seq[0] if cn_seq else {}
            concept_name = (
                cn.get("CodeMeaning") or cn.get("CodeValue") or "?"
            )
            vt = b.get("ValueType", "")
            value = (
                b.get("TextValue")
                or b.get("NumericValue")
                or _maybe_concept_value(b.get("ConceptCodeSequence"))
                or ""
            )
            nested_items: list[DICOMSRItem] = []
            child = b.get("ContentSequence") or []
            if child:
                _flatten(child, nested_items)
            acc.append(DICOMSRItem(
                concept_name=concept_name,
                concept_value=str(value),
                value_type=vt,
                nested=nested_items,
            ))

    def _maybe_concept_value(seq: list[dict[str, Any]] | None
                              ) -> str | None:
        if not seq:
            return None
        first = seq[0]
        return (
            first.get("CodeMeaning") or first.get("CodeValue")
        )

    top_content = dataset.get("ContentSequence") or []
    _flatten(top_content, items)

    return DICOMSRReport(
        modality=dataset.get("Modality", ""),
        sop_class_uid=dataset.get("SOPClassUID", ""),
        study_instance_uid=dataset.get("StudyInstanceUID", ""),
        series_instance_uid=dataset.get("SeriesInstanceUID", ""),
        n_items=len(items),
        items=items,
        completion_flag=dataset.get("CompletionFlag", ""),
        verification_flag=dataset.get("VerificationFlag", ""),
        rationale=(
            f"DICOM SR parsed: {len(items)} top-level items, "
            f"modality={dataset.get('Modality', '?')}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Builder helper for tests + playground demo
# ─────────────────────────────────────────────────────────────────────


def _enc_tag(group: int, elem: int) -> bytes:
    return struct.pack("<HH", group, elem)


def _enc_short_vr(vr: str, value: bytes) -> bytes:
    return vr.encode("ascii") + struct.pack("<H", len(value)) + value


def _enc_long_vr(vr: str, value: bytes) -> bytes:
    return (
        vr.encode("ascii") + b"\x00\x00"
        + struct.pack("<I", len(value)) + value
    )


def _enc_str(value: str) -> bytes:
    s = value.encode("utf-8")
    if len(s) % 2:
        s += b" "
    return s


def build_minimal_dicom_sr_bytes(
    *,
    modality: str = "SR",
    sop_class_uid: str = "1.2.840.10008.5.1.4.1.1.88.11",
    study_instance_uid: str = "1.2.3.4.5.6",
    series_instance_uid: str = "1.2.3.4.5.7",
    items: list[tuple[str, str, str]] | None = None,
) -> bytes:
    """Build a tiny but well-formed DICOM Part-10 SR for tests +
    demos. Each ``items`` triple is (concept_name, value_type,
    text_value). All emitted as TEXT items at the top level."""
    items = items or [
        ("Imaging Procedure Description", "TEXT",
         "Diagnostic CT chest with contrast"),
    ]

    def encode_item(concept_name: str, value_type: str,
                    text_value: str) -> bytes:
        # Concept Name Code Sequence -> single item with CodeMeaning
        cn_inner = (
            _enc_tag(0x0008, 0x0104)
            + _enc_short_vr("LO", _enc_str(concept_name))
        )
        # Item start FFFE,E000
        item_header = struct.pack("<HHI", 0xFFFE, 0xE000,
                                    len(cn_inner))
        cn_seq_value = item_header + cn_inner
        cn_seq_elem = (
            _enc_tag(0x0040, 0xA043)
            + _enc_long_vr("SQ", cn_seq_value)
        )
        vt_elem = (
            _enc_tag(0x0040, 0xA040)
            + _enc_short_vr("CS", _enc_str(value_type))
        )
        text_elem = (
            _enc_tag(0x0040, 0xA160)
            + _enc_long_vr("UT", _enc_str(text_value))
        )
        item_body = cn_seq_elem + vt_elem + text_elem
        return (
            struct.pack("<HHI", 0xFFFE, 0xE000, len(item_body))
            + item_body
        )

    content_seq_value = b"".join(
        encode_item(name, vt, value) for name, vt, value in items
    )
    dataset = (
        _enc_tag(0x0008, 0x0016)
        + _enc_short_vr("UI", _enc_str(sop_class_uid))
        + _enc_tag(0x0008, 0x0060)
        + _enc_short_vr("CS", _enc_str(modality))
        + _enc_tag(0x0020, 0x000D)
        + _enc_short_vr("UI", _enc_str(study_instance_uid))
        + _enc_tag(0x0020, 0x000E)
        + _enc_short_vr("UI", _enc_str(series_instance_uid))
        + _enc_tag(0x0040, 0xA491)
        + _enc_short_vr("CS", _enc_str("PARTIAL"))
        + _enc_tag(0x0040, 0xA493)
        + _enc_short_vr("CS", _enc_str("UNVERIFIED"))
        + _enc_tag(0x0040, 0xA730)
        + _enc_long_vr("SQ", content_seq_value)
    )
    return b"\x00" * _PREAMBLE_LEN + _DICM + dataset
