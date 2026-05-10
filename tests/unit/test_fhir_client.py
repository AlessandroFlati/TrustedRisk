"""Unit tests for the FHIR client wrappers."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.fhir.client import resolve_patient_id
from mcp_server.sharp.headers import FHIRContext, _fhir_ctx


def _run(coro):
    return asyncio.run(coro)


def _ctx(patient_id: str | None):
    return FHIRContext(server_url="http://hapi/fhir", access_token="tok", patient_id=patient_id)


def test_resolve_patient_id_explicit_wins():
    token = _fhir_ctx.set(_ctx("from-header"))
    try:
        assert _run(resolve_patient_id("from-arg")) == "from-arg"
    finally:
        _fhir_ctx.reset(token)


def test_resolve_patient_id_falls_back_to_header():
    token = _fhir_ctx.set(_ctx("from-header"))
    try:
        assert _run(resolve_patient_id(None)) == "from-header"
    finally:
        _fhir_ctx.reset(token)


def test_resolve_patient_id_strips_whitespace():
    token = _fhir_ctx.set(_ctx(None))
    try:
        assert _run(resolve_patient_id("  pt-1  ")) == "pt-1"
    finally:
        _fhir_ctx.reset(token)


def test_resolve_patient_id_raises_when_missing_everywhere():
    token = _fhir_ctx.set(_ctx(None))
    try:
        with pytest.raises(ValueError, match="patient_id is required"):
            _run(resolve_patient_id(None))
    finally:
        _fhir_ctx.reset(token)


def test_resolve_patient_id_raises_on_empty_string():
    token = _fhir_ctx.set(_ctx(""))
    try:
        with pytest.raises(ValueError, match="patient_id is required"):
            _run(resolve_patient_id(""))
    finally:
        _fhir_ctx.reset(token)


def test_get_fhir_client_no_fhirpy(monkeypatch):
    """If fhirpy is not installed, get_fhir_client raises a clear error."""
    import sys
    from mcp_server.fhir import client as cli

    # Force ImportError by removing fhirpy from path
    monkeypatch.setitem(sys.modules, "fhirpy", None)
    token = _fhir_ctx.set(_ctx("pt-1"))
    try:
        with pytest.raises(RuntimeError, match="fhirpy not installed"):
            _run(cli.get_fhir_client())
    finally:
        _fhir_ctx.reset(token)


def test_to_dict_serialize_path():
    """_to_dict prefers .serialize() if available."""
    from mcp_server.fhir.client import _to_dict

    class WithSerialize:
        def serialize(self):
            return {"resourceType": "Patient", "id": "x"}

    assert _to_dict(WithSerialize()) == {"resourceType": "Patient", "id": "x"}


def test_to_dict_dict_passthrough():
    from mcp_server.fhir.client import _to_dict
    assert _to_dict({"resourceType": "Patient", "id": "y"}) == {"resourceType": "Patient", "id": "y"}


def test_to_dict_unknown_returns_empty():
    from mcp_server.fhir.client import _to_dict

    class Opaque:
        pass

    assert _to_dict(Opaque()) == {}
