"""Phase 9.5 -- N-of-1 trial designer tests."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.n_of_1_trial_design import (
    compute_n_of_1_trial_design,
)


def _run(coro):
    return asyncio.run(coro)


def test_default_design_is_ABAB_with_4_blocks():
    out = _run(compute_n_of_1_trial_design(
        intervention_label="PT for chronic LBP",
        outcome_label="self-reported pain 0-10",
    ))
    assert out.design_type == "ABAB"
    assert out.n_blocks == 4
    assert out.randomization_sequence == ["A", "B", "A", "B"]


def test_total_duration_matches_blocks_x_block_days():
    out = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        block_duration_days=21, n_blocks=6,
        design_type="ABABAB",
    ))
    assert out.total_duration_days == 21 * 6
    assert out.n_blocks == 6


def test_randomized_block_design_uses_seed_for_reproducibility():
    a = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        design_type="randomized_block", n_blocks=6, seed=42,
    ))
    b = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        design_type="randomized_block", n_blocks=6, seed=42,
    ))
    assert a.randomization_sequence == b.randomization_sequence


def test_randomized_block_design_balanced_AB_count():
    out = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        design_type="randomized_block", n_blocks=8, seed=1,
    ))
    n_a = out.randomization_sequence.count("A")
    n_b = out.randomization_sequence.count("B")
    assert n_a == n_b


def test_minimum_detectable_effect_decreases_with_more_blocks():
    short = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        design_type="ABAB", n_blocks=4,
        n_observations_per_block=14,
    ))
    long = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        design_type="ABABAB", n_blocks=6,
        n_observations_per_block=14,
    ))
    assert long.minimum_detectable_effect < short.minimum_detectable_effect


def test_carryover_estimate_grows_with_half_life():
    short = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        intervention_half_life_days=1, block_duration_days=14,
    ))
    long = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
        intervention_half_life_days=10, block_duration_days=14,
    ))
    assert long.expected_carryover_periods >= short.expected_carryover_periods


def test_analysis_plan_has_required_steps():
    out = _run(compute_n_of_1_trial_design(
        intervention_label="x", outcome_label="y",
    ))
    step_ids = {s.step_id for s in out.analysis_plan}
    assert "exclude_carryover" in step_ids
    assert "paired_t_test" in step_ids


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        _run(compute_n_of_1_trial_design(
            intervention_label="x", outcome_label="y",
            block_duration_days=0,
        ))


def test_research_design_bundle_present():
    from mcp_server.tools import BUNDLES
    assert "research_design" in BUNDLES
    assert "compute_n_of_1_trial_design" in BUNDLES["research_design"]
