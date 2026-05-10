"""Phase 17.AS - Causal forest tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.causal_forest import (
    CausalForest, CausalForestReport,
    fit_causal_forest, overall_ate, predict_cate,
    report_causal_forest,
)


def _synthetic_heterogeneous_treatment(
    n: int = 400, seed: int = 7,
) -> tuple[list[list[float]], list[int], list[float]]:
    """Generate (X, T, Y) where the treatment effect depends on x_0:
       - For x_0 < 0: tau(x) = 0.0
       - For x_0 >= 0: tau(x) = 0.5
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    T: list[int] = []
    Y: list[float] = []
    for _ in range(n):
        x0 = rng.gauss(0, 1)
        x1 = rng.gauss(0, 1)
        t = rng.choice([0, 1])
        tau = 0.5 if x0 >= 0 else 0.0
        y = 0.2 * x0 + 0.1 * x1 + tau * t + rng.gauss(0, 0.3)
        X.append([x0, x1])
        T.append(t)
        Y.append(y)
    return X, T, Y


# ─────────────────────────────────────────────────────────────────────
# Forest fitting + prediction
# ─────────────────────────────────────────────────────────────────────

def test_forest_fits_and_predicts_finite_cate():
    X, T, Y = _synthetic_heterogeneous_treatment(n=200, seed=1)
    forest = fit_causal_forest(
        covariates=X, treatments=T, outcomes=Y,
        n_trees=20, max_depth=3, min_leaf_n=10, seed=1,
    )
    cate = predict_cate(forest, [0.5, 0.0])
    assert isinstance(cate, float)
    assert -10.0 < cate < 10.0


def test_forest_recovers_heterogeneity_sign():
    """High x_0 -> larger CATE than low x_0."""
    X, T, Y = _synthetic_heterogeneous_treatment(n=600, seed=2)
    forest = fit_causal_forest(
        covariates=X, treatments=T, outcomes=Y,
        n_trees=80, max_depth=4, min_leaf_n=15, seed=2,
    )
    cate_high = predict_cate(forest, [1.5, 0.0])
    cate_low = predict_cate(forest, [-1.5, 0.0])
    assert cate_high > cate_low


def test_overall_ate_close_to_average_of_per_query_cates():
    X, T, Y = _synthetic_heterogeneous_treatment(n=300, seed=3)
    forest = fit_causal_forest(
        covariates=X, treatments=T, outcomes=Y,
        n_trees=40, max_depth=3, min_leaf_n=10, seed=3,
    )
    a = overall_ate(forest, X)
    avg = sum(predict_cate(forest, x) for x in X) / len(X)
    assert abs(a - avg) < 1e-6


def test_forest_rejects_misaligned():
    with pytest.raises(ValueError):
        fit_causal_forest(
            covariates=[[0.0]], treatments=[0, 1], outcomes=[1.0],
        )


def test_forest_rejects_invalid_treatment_value():
    with pytest.raises(ValueError):
        fit_causal_forest(
            covariates=[[0.0], [0.0]],
            treatments=[0, 2], outcomes=[1.0, 1.0],
        )


def test_forest_rejects_empty():
    with pytest.raises(ValueError):
        fit_causal_forest(
            covariates=[], treatments=[], outcomes=[],
        )


# ─────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────

def test_report_carries_query_cates():
    X, T, Y = _synthetic_heterogeneous_treatment(n=200, seed=4)
    forest = fit_causal_forest(
        covariates=X, treatments=T, outcomes=Y,
        n_trees=20, max_depth=3, min_leaf_n=10, seed=4,
    )
    rep = report_causal_forest(
        forest,
        queries={
            "low_x0": [-1.5, 0.0],
            "high_x0": [1.5, 0.0],
        },
        covariates=X,
    )
    assert "low_x0" in rep.cate_by_query
    assert "high_x0" in rep.cate_by_query


def test_report_round_trip_through_pydantic():
    X, T, Y = _synthetic_heterogeneous_treatment(n=100, seed=5)
    forest = fit_causal_forest(
        covariates=X, treatments=T, outcomes=Y,
        n_trees=10, max_depth=2, min_leaf_n=8, seed=5,
    )
    rep = report_causal_forest(forest)
    payload = rep.model_dump(mode="json")
    rebuilt = CausalForestReport.model_validate(payload)
    assert rebuilt.n_train == rep.n_train


def test_forest_serialises_round_trip():
    X, T, Y = _synthetic_heterogeneous_treatment(n=100, seed=6)
    forest = fit_causal_forest(
        covariates=X, treatments=T, outcomes=Y,
        n_trees=5, max_depth=2, min_leaf_n=8, seed=6,
    )
    payload = forest.model_dump(mode="json")
    rebuilt = CausalForest.model_validate(payload)
    assert len(rebuilt.trees) == len(forest.trees)
