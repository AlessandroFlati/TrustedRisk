"""Phase 17.AX - Active learning acquisition tests."""

from __future__ import annotations

import pytest

from a2a_agent.active_learning import (
    ActiveLearningReport,
    select_top_k_by_committee, select_top_k_to_label,
)


# ─────────────────────────────────────────────────────────────────────
# Single-model acquisitions
# ─────────────────────────────────────────────────────────────────────

def test_least_confidence_picks_uniform_distribution_first():
    probs = [
        [0.99, 0.005, 0.005],   # very confident
        [0.34, 0.33, 0.33],     # uncertain - top by least_conf
        [0.7, 0.2, 0.1],
    ]
    rep = select_top_k_to_label(
        unlabelled_probs=probs, strategy="least_confidence", k=1,
    )
    assert rep.picks[0].instance_index == 1


def test_margin_picks_smallest_top1_top2_gap():
    probs = [
        [0.55, 0.45, 0.0],   # margin = 0.10 - smallest
        [0.7, 0.2, 0.1],
        [0.95, 0.04, 0.01],
    ]
    rep = select_top_k_to_label(
        unlabelled_probs=probs, strategy="margin", k=1,
    )
    assert rep.picks[0].instance_index == 0


def test_entropy_picks_uniform_distribution_first():
    probs = [
        [0.99, 0.01],
        [0.5, 0.5],          # max entropy
        [0.7, 0.3],
    ]
    rep = select_top_k_to_label(
        unlabelled_probs=probs, strategy="entropy", k=1,
    )
    assert rep.picks[0].instance_index == 1


def test_top_k_returns_k_picks_sorted_by_rank():
    probs = [
        [0.6, 0.4],
        [0.5, 0.5],
        [0.9, 0.1],
        [0.55, 0.45],
    ]
    rep = select_top_k_to_label(
        unlabelled_probs=probs, strategy="entropy", k=3,
    )
    assert len(rep.picks) == 3
    ranks = [p.rank for p in rep.picks]
    assert ranks == sorted(ranks)


def test_select_rejects_empty_pool():
    with pytest.raises(ValueError):
        select_top_k_to_label(
            unlabelled_probs=[], strategy="entropy", k=1,
        )


def test_select_rejects_zero_k():
    with pytest.raises(ValueError):
        select_top_k_to_label(
            unlabelled_probs=[[0.5, 0.5]],
            strategy="entropy", k=0,
        )


def test_select_rejects_unknown_strategy_for_single_model_path():
    with pytest.raises(ValueError):
        select_top_k_to_label(
            unlabelled_probs=[[0.5, 0.5]],
            strategy="query_by_committee",
            k=1,
        )


# ─────────────────────────────────────────────────────────────────────
# Query-by-committee
# ─────────────────────────────────────────────────────────────────────

def test_committee_picks_high_disagreement_first():
    pool = [
        # Strong agreement on class 0
        [[0.95, 0.05], [0.94, 0.06], [0.96, 0.04]],
        # High disagreement on the top class probability
        [[0.95, 0.05], [0.10, 0.90], [0.55, 0.45]],
        # Strong agreement on class 1
        [[0.10, 0.90], [0.05, 0.95], [0.08, 0.92]],
    ]
    rep = select_top_k_by_committee(
        unlabelled_committee_probs=pool, k=1,
    )
    assert rep.picks[0].instance_index == 1
    assert rep.strategy == "query_by_committee"


def test_committee_rejects_empty_pool():
    with pytest.raises(ValueError):
        select_top_k_by_committee(
            unlabelled_committee_probs=[], k=1,
        )


def test_committee_rejects_zero_k():
    with pytest.raises(ValueError):
        select_top_k_by_committee(
            unlabelled_committee_probs=[
                [[0.5, 0.5], [0.5, 0.5]]
            ], k=0,
        )


# ─────────────────────────────────────────────────────────────────────
# Schema invariants
# ─────────────────────────────────────────────────────────────────────

def test_report_round_trip_through_pydantic():
    rep = select_top_k_to_label(
        unlabelled_probs=[[0.6, 0.4], [0.5, 0.5]],
        strategy="entropy", k=2,
    )
    payload = rep.model_dump(mode="json")
    rebuilt = ActiveLearningReport.model_validate(payload)
    assert rebuilt.k_selected == rep.k_selected


def test_top_k_capped_when_pool_smaller_than_k():
    rep = select_top_k_to_label(
        unlabelled_probs=[[0.6, 0.4]],
        strategy="entropy", k=5,
    )
    assert rep.k_selected == 1
    assert len(rep.picks) == 1
