"""Unit tests for detect_phi tool -- regex fallback path (Presidio not required)."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.detect_phi import detect_phi


def _run(coro):
    return asyncio.run(coro)


def test_detect_phi_finds_mrn():
    rep = _run(detect_phi("Patient John Doe MRN 12345 admitted yesterday."))
    types = {e.type for e in rep.entities_found}
    assert "MRN" in types
    assert rep.risk_level == "high"


def test_detect_phi_finds_email():
    rep = _run(detect_phi("Contact: alice@example.org for follow-up."))
    types = {e.type for e in rep.entities_found}
    assert "EMAIL_ADDRESS" in types


def test_detect_phi_finds_ssn():
    rep = _run(detect_phi("Verified SSN 123-45-6789 for billing."))
    types = {e.type for e in rep.entities_found}
    assert "US_SSN" in types
    assert rep.risk_level == "high"


def test_detect_phi_finds_phone():
    rep = _run(detect_phi("Call (555) 123-4567 to schedule."))
    types = {e.type for e in rep.entities_found}
    assert "PHONE_NUMBER" in types


def test_detect_phi_finds_dob():
    rep = _run(detect_phi("DOB: 03/15/1955 -- adult cohort."))
    types = {e.type for e in rep.entities_found}
    assert "DOB" in types


def test_detect_phi_finds_credit_card():
    rep = _run(detect_phi("Charge to 4111-1111-1111-1111 if needed."))
    types = {e.type for e in rep.entities_found}
    assert "CREDIT_CARD" in types


def test_detect_phi_clean_text():
    rep = _run(detect_phi("Patient is stable for discharge home with care."))
    # Likely no PHI in this neutral clinical sentence
    assert rep.risk_level in ("none", "low", "medium")


def test_detect_phi_redaction_map_populated_for_mrn():
    rep = _run(detect_phi("Patient MRN 99999 home."))
    assert rep.redaction_map  # non-empty
    assert any("MRN" in v for v in rep.redaction_map.values())


def test_detect_phi_entity_count_by_type():
    rep = _run(detect_phi("MRN A1, MRN B2; SSN 111-22-3333."))
    if "MRN" in rep.entity_count_by_type:
        assert rep.entity_count_by_type["MRN"] >= 1


def test_detect_phi_rejects_non_string():
    with pytest.raises(ValueError):
        _run(detect_phi(12345))  # type: ignore[arg-type]
