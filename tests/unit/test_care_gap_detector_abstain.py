"""Tests for compute_care_gap_detector HL7-unknown sentinel abstain guard."""
from __future__ import annotations

import asyncio

from mcp_server.tools.care_gap_detector import compute_care_gap_detector


def _run(coro):
    return asyncio.run(coro)


_EMPTY_BUNDLE: dict = {"resourceType": "Bundle", "entry": []}


def test_unknown_age_string_abstains():
    """Passing patient_age='unknown' triggers abstain."""
    rep = _run(compute_care_gap_detector(
        fhir_bundle=_EMPTY_BUNDLE,
        patient_age="unknown",
    ))
    assert rep.abstain_recommended is True
    assert rep.abstain_reason is not None
    assert "patient_age" in rep.abstain_reason
    assert rep.n_gaps_found == 0
    assert rep.high_priority_gaps == []


def test_hl7_u_sentinel_age_abstains():
    """The HL7 two-letter code 'U' on patient_age triggers abstain."""
    rep = _run(compute_care_gap_detector(
        fhir_bundle=_EMPTY_BUNDLE,
        patient_age="U",
    ))
    assert rep.abstain_recommended is True
    assert rep.n_gaps_found == 0


def test_zero_age_abstains():
    """Age == 0 is treated as unknown sentinel for gap filtering purposes."""
    rep = _run(compute_care_gap_detector(
        fhir_bundle=_EMPTY_BUNDLE,
        patient_age=0,
    ))
    assert rep.abstain_recommended is True
    assert rep.n_gaps_found == 0


def test_unknown_sex_abstains():
    """Passing patient_sex='unknown' triggers abstain."""
    rep = _run(compute_care_gap_detector(
        fhir_bundle=_EMPTY_BUNDLE,
        patient_sex="unknown",
    ))
    assert rep.abstain_recommended is True
    assert "patient_sex" in (rep.abstain_reason or "")


def test_normal_age_does_not_abstain():
    """Valid patient_age does not trigger the abstain guard."""
    rep = _run(compute_care_gap_detector(
        fhir_bundle=_EMPTY_BUNDLE,
        patient_age=70,
        patient_sex="female",
    ))
    assert rep.abstain_recommended is False


def test_absent_age_does_not_abstain():
    """Passing no age override (None) does not trigger the sentinel guard --
    FHIR bundle inference path is used instead."""
    rep = _run(compute_care_gap_detector(
        fhir_bundle=_EMPTY_BUNDLE,
        patient_age=None,
    ))
    assert rep.abstain_recommended is False
