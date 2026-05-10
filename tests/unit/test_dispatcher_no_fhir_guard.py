"""Tests for dispatcher no-FHIR-context guard.

The dispatcher must abstain (return an error DispatchResult, never raise)
when called without a fhir_server_url in metadata. This covers both
_run_workflow and _run_specialist_route paths.
"""

from __future__ import annotations

import asyncio

import pytest

from apps.orchestrator.dispatcher import DispatchResult, dispatch


def _run(coro):
    return asyncio.run(coro)


# ------------------------------------------------------------------ workflow path


def test_dispatch_workflow_without_fhir_context_returns_error():
    """A matched workflow prompt with no metadata must abstain, not raise."""
    result = _run(dispatch("sepsis workup", metadata=None))
    assert isinstance(result, DispatchResult)
    assert result.error is not None
    assert "no_fhir_context" in result.error


def test_dispatch_workflow_with_empty_metadata_returns_error():
    result = _run(dispatch("discharge planning", metadata={}))
    assert result.error is not None
    assert "no_fhir_context" in result.error


def test_dispatch_workflow_with_metadata_but_no_url_returns_error():
    result = _run(dispatch("sepsis workup", metadata={"patient_id": "Patient/x"}))
    assert result.error is not None
    assert "no_fhir_context" in result.error


def test_dispatch_returns_abstain_flag_in_output_dump():
    """output_dump must carry abstain_recommended=True for no-context abstains."""
    result = _run(dispatch("discharge planning", metadata=None))
    assert result.output_dump is not None
    assert result.output_dump.get("abstain_recommended") is True


# ------------------------------------------------------------------ specialist path
# Use prompts that uniquely match a single specialist route, not also a workflow,
# to avoid the fanout path which wraps sub-results under a parent DispatchResult
# with error=None.


def test_dispatch_specialist_without_fhir_context_returns_error():
    """A matched specialist route (unique match) with no metadata must abstain."""
    # "cpt" only matches trustedrisk-coder CPT route, not any workflow
    result = _run(dispatch("cpt", metadata=None))
    assert isinstance(result, DispatchResult)
    assert result.error is not None
    assert "no_fhir_context" in result.error


def test_dispatch_specialist_with_empty_metadata_returns_error():
    # "icd10" only matches trustedrisk-coder ICD-10 route
    result = _run(dispatch("icd10", metadata={}))
    assert result.error is not None
    assert "no_fhir_context" in result.error


def test_dispatch_fanout_sub_results_carry_no_fhir_error():
    """When a prompt fans out to workflow+specialist, all sub-results must abstain."""
    # "news2 deterioration" fans out to inpatient_deterioration_response + specialist
    result = _run(dispatch("news2 deterioration", metadata=None))
    assert isinstance(result, DispatchResult)
    # fanout result itself has no top-level error but sub-results do
    if result.target_kind == "fanout":
        assert result.sub_results is not None
        for sub in result.sub_results:
            assert sub.error is not None
            assert "no_fhir_context" in sub.error
    else:
        # single result: must also have the error
        assert result.error is not None
        assert "no_fhir_context" in result.error


# ------------------------------------------------------------------ discovery path (no FHIR needed)


def test_dispatch_discovery_returns_catalog_when_no_match():
    """Discovery does not require FHIR context (no tool is invoked)."""
    result = _run(dispatch("hello who are you", metadata=None))
    assert result.target_kind == "discovery"
    assert result.error is None


# ------------------------------------------------------------------ unknown workflow id


def test_run_workflow_unknown_id_returns_unknown_error():
    """An unrecognised workflow id must return error='unknown_workflow'."""
    from apps.orchestrator.dispatcher import _run_workflow
    result = _run(_run_workflow("nonexistent_workflow_xyz", metadata={
        "fhir_server_url": "http://localhost:8080",
        "patient_id": "Patient/x",
    }))
    assert result.error == "unknown_workflow"
