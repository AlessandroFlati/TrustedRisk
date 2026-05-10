"""Tests for the fhir_helpers test fixture utilities.

These tests verify that bind_in_memory_fhir correctly patches
mcp_server.fhir.client so production code receives the in-memory
bundle rather than making an HTTP request.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.fixtures.fhir_helpers import bind_in_memory_fhir, load_fixture


# ------------------------------------------------------------------ helpers


def _run(coro):
    return asyncio.run(coro)


def _minimal_bundle(patient_id: str = "pt-test") -> dict:
    return {
        "resourceType": "Bundle",
        "entry": [
            {"resource": {
                "resourceType": "Patient",
                "id": patient_id,
                "birthDate": "1960-01-01",
                "gender": "male",
            }},
        ],
    }


# ------------------------------------------------------------------ load_fixture


def test_load_fixture_minimal_chf_patient():
    """The shipped fixture file loads and has the expected structure."""
    bundle = load_fixture("minimal_chf_patient")
    assert bundle["resourceType"] == "Bundle"
    types = {e["resource"]["resourceType"] for e in bundle["entry"]}
    assert "Patient" in types
    assert "Condition" in types
    assert "MedicationRequest" in types


def test_load_fixture_missing_raises():
    with pytest.raises(FileNotFoundError, match="FHIR fixture not found"):
        load_fixture("nonexistent_fixture_xyz")


# ------------------------------------------------------------------ bind_in_memory_fhir


def test_bind_in_memory_fhir_patches_fetch():
    """fetch_patient_bundle must return the supplied bundle inside the block."""
    bundle = _minimal_bundle("pt-alpha")

    async def _inner():
        from mcp_server.fhir.client import fetch_patient_bundle
        return await fetch_patient_bundle("Patient/pt-alpha")

    with bind_in_memory_fhir(bundle, patient_id="Patient/pt-alpha"):
        result = _run(_inner())
    assert result == bundle


def test_bind_in_memory_fhir_patches_resolve():
    """resolve_patient_id must return the supplied patient_id inside the block."""
    bundle = _minimal_bundle("pt-beta")

    async def _inner():
        from mcp_server.fhir.client import resolve_patient_id
        return await resolve_patient_id()

    with bind_in_memory_fhir(bundle, patient_id="Patient/pt-beta"):
        pid = _run(_inner())
    assert pid == "Patient/pt-beta"


def test_bind_in_memory_fhir_restores_after_block():
    """The original functions must be restored after the context manager exits."""
    from mcp_server.fhir.client import (
        fetch_patient_bundle as original_fetch,
        resolve_patient_id as original_resolve,
    )
    bundle = _minimal_bundle()
    with bind_in_memory_fhir(bundle):
        pass
    from mcp_server.fhir.client import (
        fetch_patient_bundle as restored_fetch,
        resolve_patient_id as restored_resolve,
    )
    assert restored_fetch is original_fetch
    assert restored_resolve is original_resolve


def test_bind_in_memory_fhir_explicit_pid_overrides_default():
    """resolve_patient_id(explicit=...) should return the explicit value."""
    bundle = _minimal_bundle("pt-gamma")

    async def _inner():
        from mcp_server.fhir.client import resolve_patient_id
        return await resolve_patient_id(explicit="Patient/override")

    with bind_in_memory_fhir(bundle, patient_id="Patient/default"):
        pid = _run(_inner())
    assert pid == "Patient/override"


def test_bind_in_memory_fhir_rejects_non_dict():
    with pytest.raises(TypeError, match="bundle must be a dict"):
        with bind_in_memory_fhir(["not", "a", "dict"]):  # type: ignore[arg-type]
            pass


def test_load_fixture_missing_error_message():
    """Error message must mention the expected path."""
    with pytest.raises(FileNotFoundError, match="tests/fixtures/fhir"):
        load_fixture("totally_missing")
