"""Phase 12.6 B1 -- `compute_causal_treatment_effect` MCP-tool wrapper."""

from __future__ import annotations

import asyncio
import random

import pytest

from mcp_server.tools.causal_inference import (
    compute_causal_treatment_effect,
)


def _run(coro):
    return asyncio.run(coro)


def _synthetic_cohort(*, n: int = 600, true_ate: float = -0.10,
                              seed: int = 7):
    """Small synthetic cohort with one binary treatment + binary outcome
    + two confounders. The data-generating process exposes the true
    ATE so refutations have ground truth to recover."""
    rng = random.Random(seed)
    rows: list[dict] = []
    for _ in range(n):
        age = rng.randint(40, 90)
        comorbidity_count = rng.randint(0, 6)
        # Treatment biased by age + comorbidity
        prop = 1.0 / (1.0 + pow(2.71828,
                                       -(0.04 * (age - 60) +
                                         0.20 * comorbidity_count)))
        treated = 1 if rng.random() < prop else 0
        # Outcome = baseline + ATE × treatment + noise
        base = 0.20 + 0.005 * (age - 60) + 0.03 * comorbidity_count
        p_outcome = max(0.0, min(1.0, base + true_ate * treated))
        outcome = 1 if rng.random() < p_outcome else 0
        rows.append({
            "patient_id": f"pt-{rng.randint(0, 999_999)}",
            "treatment": treated, "outcome": outcome,
            "age": age, "comorbidity_count": comorbidity_count,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
# Smoke
# ─────────────────────────────────────────────────────────────────────

def test_compute_causal_treatment_effect_runs_on_synthetic_cohort():
    cohort = _synthetic_cohort(n=400)
    rep = _run(compute_causal_treatment_effect(
        cohort=cohort,
        treatment="treatment", outcome="outcome",
        confounders=["age", "comorbidity_count"],
        run_refutations=False,
    ))
    assert rep.n_treated > 0 and rep.n_untreated > 0
    assert rep.naive_difference_in_means is not None


def test_estimate_recovers_negative_ate_within_tolerance():
    cohort = _synthetic_cohort(n=2000, true_ate=-0.10, seed=42)
    rep = _run(compute_causal_treatment_effect(
        cohort=cohort,
        treatment="treatment", outcome="outcome",
        confounders=["age", "comorbidity_count"],
        run_refutations=False,
    ))
    assert rep.ate_point is not None
    # ATE point estimate should be within 0.06 of the true -0.10 on n=2000
    assert abs(rep.ate_point - (-0.10)) < 0.06


def test_empty_cohort_raises():
    with pytest.raises(ValueError):
        _run(compute_causal_treatment_effect(
            cohort=[], treatment="t", outcome="o",
        ))


def test_unknown_column_raises():
    cohort = _synthetic_cohort(n=200)
    with pytest.raises(ValueError):
        _run(compute_causal_treatment_effect(
            cohort=cohort, treatment="not_a_column", outcome="outcome",
        ))


def test_single_treatment_value_raises():
    cohort = _synthetic_cohort(n=200)
    for r in cohort:
        r["treatment"] = 0   # collapse to one arm
    with pytest.raises(ValueError):
        _run(compute_causal_treatment_effect(
            cohort=cohort, treatment="treatment", outcome="outcome",
            confounders=["age"],
        ))


# ─────────────────────────────────────────────────────────────────────
# Bundle wiring
# ─────────────────────────────────────────────────────────────────────

def test_causal_tool_registered_in_research_design_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_causal_treatment_effect" in BUNDLES["research_design"]


def test_causal_tool_module_listed_in_dunder_all():
    import mcp_server.tools as t
    assert "causal_inference" in t.__all__
