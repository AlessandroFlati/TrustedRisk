"""Phase 14.10 N1 -- DecisionCard PDF export tests."""

from __future__ import annotations

import zlib

import pytest

from a2a_agent.decision_card_pdf import render_decision_card_pdf


# ─────────────────────────────────────────────────────────────────────
# Byte-level smoke
# ─────────────────────────────────────────────────────────────────────

def test_pdf_starts_with_pdf_1_4_header():
    pdf = render_decision_card_pdf()
    assert pdf.startswith(b"%PDF-1.4\n")


def test_pdf_ends_with_eof():
    pdf = render_decision_card_pdf()
    assert pdf.rstrip().endswith(b"%%EOF")


def test_pdf_contains_xref_and_trailer():
    pdf = render_decision_card_pdf()
    assert b"xref\n" in pdf
    assert b"trailer" in pdf
    assert b"startxref" in pdf


def test_pdf_contains_helvetica_font():
    pdf = render_decision_card_pdf()
    assert b"Helvetica" in pdf


def test_pdf_contains_compressed_content_stream():
    pdf = render_decision_card_pdf()
    assert b"FlateDecode" in pdf


# ─────────────────────────────────────────────────────────────────────
# Content visibility (decompressed stream)
# ─────────────────────────────────────────────────────────────────────


def _decompress_streams(pdf: bytes) -> bytes:
    """Pull out every FlateDecode stream and concatenate the decoded
    bytes -- sufficient to grep for inline text."""
    out = bytearray()
    offset = 0
    while True:
        start = pdf.find(b"stream\n", offset)
        if start == -1:
            break
        start += len(b"stream\n")
        end = pdf.find(b"\nendstream", start)
        if end == -1:
            break
        chunk = pdf[start:end]
        try:
            out += zlib.decompress(chunk)
        except zlib.error:
            pass
        offset = end + 1
    return bytes(out)


def test_pdf_contains_recommendation_text():
    pdf = render_decision_card_pdf(
        recommended_action="discharge_home",
    )
    text = _decompress_streams(pdf)
    assert b"discharge_home" in text


def test_pdf_renders_risk_estimate():
    pdf = render_decision_card_pdf(
        risk_point_estimate=0.158,
        risk_ci95=(0.123, 0.205),
    )
    text = _decompress_streams(pdf)
    assert b"0.158" in text
    assert b"0.123" in text
    assert b"0.205" in text


def test_pdf_renders_contributing_factors_table():
    pdf = render_decision_card_pdf(
        contributing_factors=[
            {"name": "LACE_length_of_stay", "lace_points": 4, "weight": 0.29},
            {"name": "LACE_acuity", "lace_points": 3, "weight": 0.21},
        ],
    )
    text = _decompress_streams(pdf)
    assert b"LACE_length_of_stay" in text
    assert b"LACE_acuity" in text


def test_pdf_renders_cited_evidence_list():
    pdf = render_decision_card_pdf(
        cited_evidence=["Condition/c-1", "Observation/o-7"],
    )
    text = _decompress_streams(pdf)
    assert b"Condition/c-1" in text
    assert b"Observation/o-7" in text


def test_pdf_audit_hash_in_footer():
    pdf = render_decision_card_pdf(
        recommended_action="discharge_home",
        timestamp_iso="2026-04-30T08:59:00+00:00",
    )
    text = _decompress_streams(pdf)
    assert b"audit-hash:" in text


# ─────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────

def test_pdf_is_deterministic_given_same_inputs():
    args = dict(
        recommended_action="discharge_home",
        risk_point_estimate=0.16,
        risk_ci95=(0.10, 0.20),
        timestamp_iso="2026-04-30T08:59:00+00:00",
    )
    pdf_a = render_decision_card_pdf(**args)
    pdf_b = render_decision_card_pdf(**args)
    assert pdf_a == pdf_b


def test_pdf_changes_when_recommendation_changes():
    args_a = dict(
        recommended_action="discharge_home",
        timestamp_iso="2026-04-30T08:59:00+00:00",
    )
    args_b = dict(
        recommended_action="continued_admission",
        timestamp_iso="2026-04-30T08:59:00+00:00",
    )
    assert render_decision_card_pdf(**args_a) != render_decision_card_pdf(
        **args_b,
    )


# ─────────────────────────────────────────────────────────────────────
# Patient label appears when supplied
# ─────────────────────────────────────────────────────────────────────

def test_pdf_renders_patient_label_when_supplied():
    pdf = render_decision_card_pdf(patient_label="MRN 0001234")
    text = _decompress_streams(pdf)
    assert b"MRN 0001234" in text
