"""CMP-4 -- PHI redaction stress test.

Property-based + adversarial assertions on:
  1. `a2a_agent.audit._redact` -- strips named PHI keys at any nesting depth
  2. `a2a_agent.audit._hash`   -- never embeds raw PHI in its output (hex only)
  3. Defense-in-depth: even "weird" nested structures (lists of dicts of
     lists of dicts) preserve the redaction guarantee
  4. Case-insensitivity of key matching

The synthesized PHI uses canary tokens (e.g. CANARY_SSN_***) so any leak
is unmissable. We assert no canary survives in the post-redaction payload
or in the hash.
"""
from __future__ import annotations

import json
import string

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from a2a_agent.audit import _hash, _redact


# Keys the audit module is required to strip
_PHI_KEYS = [
    "name", "given", "family", "birthdate", "dob",
    "ssn", "mrn", "address", "phone", "email",
    "patient_name", "first_name", "last_name",
]


# Canary tokens -- distinctive enough that any leak is detectable
_CANARY_PHI_VALUES = [
    "CANARY_SSN_123-45-6789",
    "CANARY_MRN_8675309",
    "CANARY_NAME_JOHN_DOE",
    "CANARY_PHONE_555-0100",
    "CANARY_DOB_1959-08-21",
    "CANARY_EMAIL_jdoe@example.test",
    "CANARY_ADDR_742 Evergreen Terrace",
]


# ─────────────────────── Hypothesis strategies ───────────────────────

_safe_alphabet = string.ascii_letters + string.digits + " _-"
_non_phi_key = st.text(alphabet=_safe_alphabet, min_size=1,
                          max_size=12).filter(
    lambda s: s.lower() not in _PHI_KEYS)
_phi_key = st.sampled_from(_PHI_KEYS)
_canary_value = st.sampled_from(_CANARY_PHI_VALUES)


def _safe_value_strategy(depth: int):
    """Generate a non-PHI scalar / list / dict (capped depth)."""
    if depth <= 0:
        return st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(alphabet=_safe_alphabet, max_size=10),
            st.booleans(),
            st.none(),
        )
    return st.one_of(
        st.integers(),
        st.text(alphabet=_safe_alphabet, max_size=10),
        st.lists(_safe_value_strategy(depth - 1), max_size=4),
        st.dictionaries(
            keys=_non_phi_key,
            values=_safe_value_strategy(depth - 1),
            max_size=4,
        ),
    )


def _phi_dict_strategy(depth: int = 3):
    """Build a dict that ALWAYS contains at least one canary PHI value
    nested at some depth, embedded in otherwise non-PHI structure."""
    if depth <= 0:
        return st.fixed_dictionaries({"name": _canary_value})

    return st.dictionaries(
        keys=st.one_of(_non_phi_key, _phi_key),
        values=st.one_of(_canary_value, _safe_value_strategy(depth - 1)),
        min_size=1, max_size=5,
    ).filter(_payload_has_canary)


def _payload_has_canary(payload) -> bool:
    blob = json.dumps(payload, default=str, sort_keys=True)
    return any(c in blob for c in _CANARY_PHI_VALUES)


# ─────────────────────── Invariants on _redact ───────────────────────

@given(_phi_dict_strategy())
@settings(max_examples=100, deadline=None,
            suppress_health_check=[HealthCheck.too_slow,
                                       HealthCheck.filter_too_much])
def test_top_level_phi_keys_redacted(payload):
    """Any value under a known PHI key at the top level is replaced with
    [REDACTED]; no canary survives at that key."""
    redacted = _redact(payload)
    for key in _PHI_KEYS:
        if key in redacted:
            assert redacted[key] == "[REDACTED]", \
                f"Key {key!r} retained value {redacted[key]!r}"


@given(_phi_dict_strategy())
@settings(max_examples=100, deadline=None,
            suppress_health_check=[HealthCheck.too_slow,
                                       HealthCheck.filter_too_much])
def test_no_canary_under_phi_keys_after_redact(payload):
    """No canary string ever survives under a PHI-named key."""
    redacted = _redact(payload)

    def _walk(node, parent_key: str | None = None):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, k)
        elif isinstance(node, list):
            for v in node:
                _walk(v, parent_key)
        elif isinstance(node, str):
            if parent_key and parent_key.lower() in _PHI_KEYS:
                assert node == "[REDACTED]", \
                    f"PHI canary survived under {parent_key!r}: {node!r}"

    _walk(redacted)


# ─────────────────────── Adversarial nested structures ───────────────────────

def test_phi_nested_in_list_of_dicts_redacted():
    payload = {
        "results": [
            {"name": "CANARY_NAME_JOHN_DOE", "score": 0.9},
            {"ssn": "CANARY_SSN_123-45-6789", "label": "ok"},
        ],
    }
    redacted = _redact(payload)
    blob = json.dumps(redacted)
    assert "CANARY_NAME_JOHN_DOE" not in blob
    assert "CANARY_SSN_123-45-6789" not in blob
    assert blob.count("[REDACTED]") >= 2


def test_phi_at_depth_5_redacted():
    payload = {"a": {"b": {"c": {"d": {"e": {"name": "CANARY_NAME_JOHN_DOE"}}}}}}
    blob = json.dumps(_redact(payload))
    assert "CANARY_NAME_JOHN_DOE" not in blob


def test_unicode_canary_redacted():
    payload = {"name": "Иван Иванов 患者 \U0001F480"}
    blob = json.dumps(_redact(payload), ensure_ascii=False)
    assert "Иван" not in blob


def test_phi_key_case_variants_redacted():
    """The current implementation uses .lower() -- so NAME and Name should
    also be stripped. Document this behavior with a test."""
    for variant in ("Name", "NAME", "nAmE"):
        payload = {variant: "CANARY_NAME_JOHN_DOE"}
        blob = json.dumps(_redact(payload))
        assert "CANARY_NAME_JOHN_DOE" not in blob, \
            f"Failed for variant {variant!r}"


# ─────────────────────── Hash invariants ───────────────────────

@given(_phi_dict_strategy())
@settings(max_examples=100, deadline=None,
            suppress_health_check=[HealthCheck.too_slow,
                                       HealthCheck.filter_too_much])
def test_hash_format_is_hex(payload):
    """Hash output is always sha256:<64 hex chars> -- never a raw PHI string."""
    h = _hash(payload)
    assert h.startswith("sha256:")
    suffix = h[len("sha256:"):]
    assert len(suffix) == 64
    assert all(c in "0123456789abcdef" for c in suffix)


@given(_phi_dict_strategy())
@settings(max_examples=50, deadline=None,
            suppress_health_check=[HealthCheck.too_slow,
                                       HealthCheck.filter_too_much])
def test_hash_does_not_contain_canaries(payload):
    """No canary substring should ever appear in the hash output."""
    h = _hash(_redact(payload))
    for canary in _CANARY_PHI_VALUES:
        assert canary not in h


def test_hash_of_redacted_differs_from_hash_of_raw():
    """Asserting that redaction actually changes the hash (sanity)."""
    payload = {"name": "CANARY_NAME_JOHN_DOE", "x": 1}
    h_raw = _hash(payload)
    h_red = _hash(_redact(payload))
    assert h_raw != h_red


def test_hash_stable_across_calls():
    """Same input -> same hash, regardless of insertion order in the dict."""
    a = {"x": 1, "y": 2, "name": "X"}
    b = {"name": "X", "y": 2, "x": 1}
    assert _hash(a) == _hash(b)


def test_empty_payload_hash_is_canonical():
    assert _hash(None) == "sha256:empty"


# ─────────────────────── Property: redact is idempotent ───────────────────────

@given(_phi_dict_strategy())
@settings(max_examples=50, deadline=None,
            suppress_health_check=[HealthCheck.too_slow,
                                       HealthCheck.filter_too_much])
def test_redact_is_idempotent(payload):
    once = _redact(payload)
    twice = _redact(once)
    assert json.dumps(once, sort_keys=True, default=str) == \
           json.dumps(twice, sort_keys=True, default=str)


# ─────────────────────── Adversarial: detect_phi end-to-end ───────────────────────

def test_detect_phi_handles_arbitrary_strings_without_crash():
    """The detect_phi tool must never raise on any string input."""
    import asyncio
    from mcp_server.tools.detect_phi import detect_phi
    inputs = [
        "",
        "no PHI here",
        "Pt John Doe MRN 12345 DOB 1955-04-12 555-867-5309",
        "Patient \U0001F480 with phone 555-0100",
        "<script>alert('xss')</script>",
        "''; DROP TABLE patients; --",
        "A" * 5000,
    ]
    for s in inputs:
        result = asyncio.run(detect_phi(s))
        assert result is not None
        assert hasattr(result, "entities_found")
