"""Phase 3.1 -- A2A INPUT_REQUIRED lifecycle tests.

Verifies that the patient resolver and the DDx ranker promote
"ambiguous-but-recoverable" failures to a structured ClarificationRequest
+ task_state="input_required" instead of just abstaining. The BYO
orchestrator on the Prompt Opinion platform reads task_state, asks the
user the clarification question, and re-runs the tool with the
extended context.
"""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.differential_diagnosis_ranker import (
    compute_differential_diagnosis_ranker,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── DDx ranker INPUT_REQUIRED ───────────────────────

def test_ddx_unsupported_complaint_emits_clarification():
    """Free-text complaint that doesn't map to the DDx table promotes
    to INPUT_REQUIRED with a candidates list."""
    out = _run(compute_differential_diagnosis_ranker(
        chief_complaint="just feeling tired and run down lately",
    ))
    assert out.abstain_recommended is True
    assert out.task_state == "input_required"
    assert out.clarification_request is not None
    cr = out.clarification_request
    assert cr.expected_answer_kind == "narrower_diagnosis"
    assert cr.candidates  # non-empty
    assert cr.question


def test_ddx_supported_complaint_completes_normally():
    out = _run(compute_differential_diagnosis_ranker(
        chief_complaint="chest pain",
    ))
    assert out.task_state == "completed"
    assert out.clarification_request is None
    assert out.abstain_recommended is False


def test_ddx_empty_complaint_still_abstains_no_clarification():
    """Empty complaint is unsalvageable -- abstain without INPUT_REQUIRED.

    A clarification asking 'what is your chief complaint?' could be
    added in the future, but for now empty input is treated as a
    client contract violation.
    """
    out = _run(compute_differential_diagnosis_ranker(chief_complaint=""))
    assert out.abstain_recommended is True
    # No clarification -- the empty-input branch leaves task_state at
    # the default
    assert out.task_state == "completed"


# Resolver INPUT_REQUIRED behaviour is verified by the existing
# `test_ambiguous_match_abstains` in `test_conversational_resolver.py`,
# which uses the proper FHIR-context monkeypatch fixture. The test
# there asserts task_state="input_required" + candidates list.


# ─────────────────────── Schema invariants ───────────────────────

def test_clarification_request_question_is_non_empty():
    out = _run(compute_differential_diagnosis_ranker(
        chief_complaint="weird sensation in my elbow",
    ))
    if out.clarification_request:
        assert out.clarification_request.question.strip()


def test_clarification_candidates_match_expected_answer_kind():
    """When expected_answer_kind enumerates options, candidates list
    must be populated."""
    out = _run(compute_differential_diagnosis_ranker(
        chief_complaint="weird sensation in my elbow",
    ))
    cr = out.clarification_request
    if cr and cr.expected_answer_kind in (
            "patient_id", "narrower_diagnosis", "specific_drug_name"):
        assert cr.candidates is not None
        assert len(cr.candidates) > 0


def test_completed_results_carry_no_clarification():
    """All `task_state == "completed"` results must have
    `clarification_request is None` -- INPUT_REQUIRED and completed are
    mutually exclusive."""
    out = _run(compute_differential_diagnosis_ranker(
        chief_complaint="abdominal pain",
    ))
    assert out.task_state == "completed"
    assert out.clarification_request is None
