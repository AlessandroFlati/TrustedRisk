"""Unit tests for ground_claim -- decomposition + verdict logic."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.ground_claim import (
    _aggregate_overall,
    _classify,
    _decompose_claim,
    ground_claim,
)
from shared.schemas import EvidenceSource, SubClaim


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── _decompose_claim ───────────────────────

def test_decompose_single_general():
    parts = _decompose_claim("Patient ready for review")
    assert len(parts) == 1
    text, kind = parts[0]
    assert "ready for review" in text.lower()


def test_decompose_vital_keyword():
    parts = _decompose_claim("Vital signs are stable.")
    kinds = {kind for _, kind in parts}
    assert "vital" in kinds


def test_decompose_medication_keyword():
    parts = _decompose_claim("Patient on anticoagulant therapy.")
    kinds = {kind for _, kind in parts}
    assert "medication" in kinds


def test_decompose_procedure_keyword():
    parts = _decompose_claim("Recent CABG surgery.")
    kinds = {kind for _, kind in parts}
    assert "procedure" in kinds


def test_decompose_multi_part_claim():
    parts = _decompose_claim("Vital signs stable; medication reconciled. Patient prepared.")
    # Should split into >= 2 sub-claims
    assert len(parts) >= 2


def test_decompose_filters_short_fragments():
    parts = _decompose_claim("Ok. Bad. The patient looks great today.")
    # "Ok" and "Bad" filtered as too short
    for text, _ in parts:
        assert len(text) >= 5


# ─────────────────────── _classify ───────────────────────

def test_classify_no_evidence_unsupported():
    verdict, conf, reason = _classify("ambiguous claim", "general", evidence=[])
    assert verdict == "unsupported"
    assert conf < 0.5
    assert reason is not None and len(reason) > 0


def test_classify_strong_evidence_supported():
    ev = [
        EvidenceSource(
            source_type="fhir_observation",
            source_id="obs-1",
            excerpt="BP 120/80 within normal range",
            relevance_score=0.95,
            recency_days=0,
        ),
        EvidenceSource(
            source_type="fhir_observation",
            source_id="obs-2",
            excerpt="HR 72 stable",
            relevance_score=0.85,
            recency_days=0,
        ),
    ]
    verdict, conf, _ = _classify("vitals stable", "vital", evidence=ev)
    # With high-relevance recent evidence, expect supported
    assert verdict in ("supported", "partially_supported")
    assert conf > 0.4


# ─────────────────────── _aggregate_overall ───────────────────────

def _make_sub(verdict: str) -> SubClaim:
    return SubClaim(
        text="x", atomic_kind="general", verdict=verdict,
        confidence=0.7, evidence_sources=[],
    )


def test_aggregate_all_supported():
    subs = [_make_sub("supported"), _make_sub("supported")]
    assert _aggregate_overall(subs) == "supported"


def test_aggregate_one_unsupported():
    subs = [_make_sub("supported"), _make_sub("unsupported")]
    # Single unsupported pulls down to partially_supported or unsupported
    assert _aggregate_overall(subs) in ("partially_supported", "unsupported")


def test_aggregate_all_unsupported():
    subs = [_make_sub("unsupported"), _make_sub("unsupported")]
    assert _aggregate_overall(subs) == "unsupported"


def test_aggregate_empty():
    assert _aggregate_overall([]) == "unsupported"


# ─────────────────────── End-to-end with stubbed FHIR ───────────────────────

def test_ground_claim_stub_corpus(monkeypatch):
    """ground_claim runs in stub mode (no corpus). Should still return a ClaimGrounding."""
    from mcp_server.tools import ground_claim as gc

    async def stub_fetch(pid):
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Observation", "id": "obs-bp",
                              "code": {"text": "Blood pressure"},
                              "valueString": "120/80",
                              "effectiveDateTime": "2025-12-15T10:00:00Z"}},
            ],
        }

    async def stub_resolve(explicit):
        return explicit or "pt-1"

    monkeypatch.setattr(gc, "fetch_patient_bundle", stub_fetch)
    monkeypatch.setattr(gc, "resolve_patient_id", stub_resolve)

    result = _run(ground_claim(claim_text="Vital signs are stable.", patient_id="pt-1"))
    assert result.claim_text == "Vital signs are stable."
    assert len(result.sub_claims) >= 1
    assert result.overall_verdict in ("supported", "partially_supported", "unsupported")
    assert result.context_fingerprint.startswith("sha256:")


def test_ground_claim_guidelines_only(monkeypatch):
    """guidelines-only path doesn't need patient_id."""
    result = _run(ground_claim(
        claim_text="LACE 10-19 indicates high readmission risk.",
        sources=["guidelines"],
    ))
    assert isinstance(result.sub_claims, list)
