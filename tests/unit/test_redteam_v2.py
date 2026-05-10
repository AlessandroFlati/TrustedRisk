"""Phase 10.4 -- Adversarial evaluation suite v2 unit tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from a2a_agent.redteam_v2 import (
    load_corpus,
    run_redteam_corpus,
)
from shared.schemas import RedTeamPrompt


def _run(coro):
    return asyncio.run(coro)


CORPUS_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "redteam" / "redteam_corpus_v2.json"
)


# ─────────────────────────────────────────────────────────────────────
# Corpus health
# ─────────────────────────────────────────────────────────────────────

def test_corpus_loads_and_has_at_least_100_prompts():
    prompts = load_corpus(CORPUS_PATH)
    assert len(prompts) >= 100


def test_corpus_covers_all_required_categories():
    prompts = load_corpus(CORPUS_PATH)
    categories = {p.category for p in prompts}
    required = {
        "prompt_injection", "phi_exfiltration", "jailbreak",
        "hallucination_trigger", "citation_fabrication", "bias_probe",
        "ood_input", "multilingual_evasion", "encoding_obfuscation",
        "tool_misuse",
    }
    assert required <= categories


def test_corpus_has_critical_severity_phi_cases():
    """PHI exfiltration must have at least one critical-severity probe."""
    prompts = load_corpus(CORPUS_PATH)
    critical_phi = [
        p for p in prompts
        if p.category == "phi_exfiltration" and p.severity == "critical"
    ]
    assert critical_phi, "Need at least one critical PHI probe"


def test_corpus_prompt_ids_are_unique():
    raw = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    ids = [p["prompt_id"] for p in raw["prompts"]]
    assert len(ids) == len(set(ids))


# ─────────────────────────────────────────────────────────────────────
# Harness behaviour -- synthetic targets
# ─────────────────────────────────────────────────────────────────────

async def _safe_target(payload: str) -> dict:
    """Target that always returns a benign empty schema -- should pass
    every prompt that doesn't require a forbidden-substring leak."""
    return {"summary": "received", "len": len(payload)}


async def _leaky_target(payload: str) -> dict:
    """Target that echoes the payload verbatim -- guaranteed to leak
    every forbidden substring."""
    return {"echo": payload}


async def _crashing_target(payload: str) -> dict:
    raise RuntimeError("simulated crash")


def test_safe_target_passes_most_cases():
    rep = _run(run_redteam_corpus(
        CORPUS_PATH, _safe_target, target_label="safe-stub",
    ))
    assert rep.overall_pass_rate >= 0.95


def test_leaky_target_fails_on_phi_critical():
    rep = _run(run_redteam_corpus(
        CORPUS_PATH, _leaky_target, target_label="leaky-stub",
    ))
    # Critical PHI cases should ALL fail since the target echoes the
    # SSN / MRN / phone payloads verbatim.
    crit_phi = [
        c for c in rep.cases
        if c.category == "phi_exfiltration" and c.severity == "critical"
    ]
    assert all(not c.passed for c in crit_phi)


def test_leaky_target_posture_is_fail():
    rep = _run(run_redteam_corpus(
        CORPUS_PATH, _leaky_target, target_label="leaky-stub",
    ))
    assert rep.posture == "fail"


def test_crashing_target_marks_cases_as_crash():
    rep = _run(run_redteam_corpus(
        CORPUS_PATH, _crashing_target, target_label="crashing-stub",
    ))
    crashes = [c for c in rep.cases if c.observed_outcome == "crash"]
    assert len(crashes) == rep.n_cases
    assert rep.posture == "fail"


def test_pass_rate_aggregates_align_with_per_case_results():
    rep = _run(run_redteam_corpus(
        CORPUS_PATH, _safe_target, target_label="safe-stub",
    ))
    n_pass = sum(1 for c in rep.cases if c.passed)
    assert rep.n_passed == n_pass
    assert abs(rep.overall_pass_rate - n_pass / rep.n_cases) < 1e-6


def test_pass_rate_by_category_covers_every_seen_category():
    rep = _run(run_redteam_corpus(
        CORPUS_PATH, _safe_target, target_label="safe-stub",
    ))
    seen_cats = {c.category for c in rep.cases}
    assert set(rep.pass_rate_by_category.keys()) == seen_cats


def test_in_memory_corpus_works():
    """Bypass corpus file -- pass a list of RedTeamPrompts directly."""
    prompts = [
        RedTeamPrompt(
            prompt_id="X-001", category="prompt_injection",
            severity="low", payload="hello",
            expected_outcome="execute_safely",
            forbidden_substrings=["hello"],   # echoes hello -> fails
        ),
    ]
    rep = _run(run_redteam_corpus(
        prompts, _leaky_target, target_label="x", corpus_id="x",
    ))
    assert rep.n_cases == 1
    assert rep.cases[0].passed is False
