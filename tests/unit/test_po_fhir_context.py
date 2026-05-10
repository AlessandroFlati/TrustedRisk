"""Unit tests for the Prompt Opinion A2A FHIR-context extension reader.

Covers:
  - Extension URI is the canonical Prompt Opinion one
  - camelCase metadata payload parsed correctly
  - Optional fields are optional
  - Malformed / missing-required fields rejected with ValueError
  - bind/release populates and clears the shared FHIRContext ContextVar
"""

from __future__ import annotations

import pytest

from a2a_agent.po_fhir_context import (
    PO_FHIR_CONTEXT_URI,
    POFhirContextPayload,
    bind_po_fhir_context,
    extract_po_fhir_context,
    release_po_fhir_context,
    to_fhir_context,
)
from mcp_server.sharp.headers import FHIRContext, get_fhir_context


# ─────────────────────── Extension URI ───────────────────────

def test_extension_uri_is_prompt_opinion_canonical():
    assert PO_FHIR_CONTEXT_URI == (
        "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"
    )


# ─────────────────────── extract_po_fhir_context ───────────────────────

def test_extract_returns_none_when_metadata_absent():
    assert extract_po_fhir_context(None) is None
    assert extract_po_fhir_context({}) is None


def test_extract_returns_none_when_extension_not_declared():
    metadata = {"some_other_extension": {"foo": "bar"}}
    assert extract_po_fhir_context(metadata) is None


def test_extract_full_payload():
    metadata = {
        PO_FHIR_CONTEXT_URI: {
            "fhirUrl": "https://hapi.example.org/fhir",
            "fhirToken": "tok-123",
            "patientId": "Patient/abc",
            "fhirRefreshToken": "refresh-xyz",
            "fhirRefreshTokenUrl": "https://idp.example.org/refresh",
        }
    }
    p = extract_po_fhir_context(metadata)
    assert isinstance(p, POFhirContextPayload)
    assert p.fhir_url == "https://hapi.example.org/fhir"
    assert p.fhir_token == "tok-123"
    assert p.patient_id == "Patient/abc"
    assert p.fhir_refresh_token == "refresh-xyz"
    assert p.fhir_refresh_token_url == "https://idp.example.org/refresh"


def test_extract_minimal_payload_only_url_required():
    metadata = {PO_FHIR_CONTEXT_URI: {"fhirUrl": "https://hapi.example.org/fhir"}}
    p = extract_po_fhir_context(metadata)
    assert p.fhir_url == "https://hapi.example.org/fhir"
    assert p.fhir_token is None
    assert p.patient_id is None
    assert p.fhir_refresh_token is None
    assert p.fhir_refresh_token_url is None


def test_extract_payload_must_be_object():
    metadata = {PO_FHIR_CONTEXT_URI: "not-an-object"}
    with pytest.raises(ValueError, match="must be an object"):
        extract_po_fhir_context(metadata)


def test_extract_missing_fhirUrl_rejected():
    metadata = {PO_FHIR_CONTEXT_URI: {"fhirToken": "t"}}
    with pytest.raises(ValueError, match="fhirUrl"):
        extract_po_fhir_context(metadata)


def test_extract_empty_fhirUrl_rejected():
    metadata = {PO_FHIR_CONTEXT_URI: {"fhirUrl": ""}}
    with pytest.raises(ValueError, match="fhirUrl"):
        extract_po_fhir_context(metadata)


def test_extract_optional_field_must_be_non_empty_string_when_present():
    metadata = {
        PO_FHIR_CONTEXT_URI: {
            "fhirUrl": "https://x.example/fhir",
            "fhirToken": "",   # empty string forbidden when present
        }
    }
    with pytest.raises(ValueError, match="fhirToken"):
        extract_po_fhir_context(metadata)


# ─────────────────────── to_fhir_context ───────────────────────

def test_to_fhir_context_maps_camel_to_snake():
    payload = POFhirContextPayload(
        fhir_url="https://hapi.example.org/fhir",
        fhir_token="tok",
        patient_id="Patient/x",
        fhir_refresh_token="r-tok",
        fhir_refresh_token_url="https://idp.example.org/refresh",
    )
    ctx = to_fhir_context(payload)
    assert isinstance(ctx, FHIRContext)
    assert ctx.server_url == "https://hapi.example.org/fhir"
    assert ctx.access_token == "tok"
    assert ctx.patient_id == "Patient/x"
    assert ctx.refresh_token == "r-tok"
    assert ctx.refresh_token_url == "https://idp.example.org/refresh"


def test_to_fhir_context_no_token_uses_empty_string():
    """SHARP supports the no-token case; the empty string is the semantic
    carrier, not None."""
    payload = POFhirContextPayload(
        fhir_url="https://hapi.example.org/fhir", fhir_token=None,
    )
    ctx = to_fhir_context(payload)
    assert ctx.access_token == ""


# ─────────────────────── bind / release ───────────────────────

def test_bind_and_release_populates_context_var():
    payload = POFhirContextPayload(
        fhir_url="https://hapi.example.org/fhir",
        fhir_token="tok",
        patient_id="Patient/y",
    )
    token = bind_po_fhir_context(payload)
    try:
        ctx = get_fhir_context()
        assert ctx.server_url == "https://hapi.example.org/fhir"
        assert ctx.patient_id == "Patient/y"
    finally:
        release_po_fhir_context(token)
    # After release, the ContextVar must be empty
    with pytest.raises(LookupError):
        get_fhir_context()
