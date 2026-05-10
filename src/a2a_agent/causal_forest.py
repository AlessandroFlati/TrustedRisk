"""Phase 17.AS - Causal forest for conditional ATE (Athey & Wager 2019).

Athey, Tibshirani, Wager 2019 "Generalized Random Forests" (Annals
of Statistics) introduced the *causal forest* - a forest where each
leaf estimates the conditional average treatment effect (CATE) on
its locally-similar subgroup.

This is a minimal pure-Python implementation:

  - **Tree growth**: at each split, choose the feature + threshold
    that maximises the *heterogeneity* of the resulting child CATEs
    (i.e. the difference in treatment effects between left and
    right children).
  - **Honest splitting**: subsample is split into a "splitting"
    half (used to choose splits) and an "estimation" half (used to
    compute the CATE in each leaf). This honesty principle is what
    makes the forest's CATE estimates asymptotically Normal
    (Wager & Athey 2018).
  - **Forest**: ``n_trees`` trees on bootstrap subsamples; CATE for
    a query point is the average of per-tree CATEs of the leaves
    that contain it.

Pure-Python deterministic. No NumPy.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Tree node
# ─────────────────────────────────────────────────────────────────────


class _CFTreeNode(BaseModel):
    is_leaf: bool
    feature_index: int | None = None
    split_threshold: float | None = None
    left: "_CFTreeNode | None" = None
    right: "_CFTreeNode | None" = None
    leaf_cate: float | None = None
    leaf_n: int = 0


_CFTreeNode.model_rebuild()


def _ate_for_indices(
    indices: list[int],
    treatments: list[int],
    outcomes: list[float],
) -> float:
    """ATE on a subset: mean(Y | T=1) - mean(Y | T=0)."""
    if not indices:
        return 0.0
    y_t1 = [outcomes[i] for i in indices if treatments[i] == 1]
    y_t0 = [outcomes[i] for i in indices if treatments[i] == 0]
    if not y_t1 or not y_t0:
        return 0.0
    return sum(y_t1) / len(y_t1) - sum(y_t0) / len(y_t0)


def _grow_tree(
    splitting: list[int],
    estimation: list[int],
    covariates: list[list[float]],
    treatments: list[int],
    outcomes: list[float],
    *,
    max_depth: int,
    min_leaf_n: int,
    rng: random.Random,
    depth: int = 0,
) -> _CFTreeNode:
    if (
        depth >= max_depth
        or len(splitting) < 2 * min_leaf_n
        or len(estimation) < min_leaf_n
    ):
        cate = _ate_for_indices(estimation, treatments, outcomes)
        return _CFTreeNode(
            is_leaf=True, leaf_cate=cate, leaf_n=len(estimation),
        )
    p = len(covariates[0]) if covariates else 0
    n = len(splitting)
    best_score = -1.0
    best_feat = -1
    best_thr = 0.0
    best_left: list[int] = []
    best_right: list[int] = []
    # Try a small random subset of features (mtry)
    n_try = max(1, int(round(math.sqrt(p))))
    feature_candidates = rng.sample(range(p), min(n_try, p))
    for f in feature_candidates:
        # Sort splitting indices by feature value
        sorted_idx = sorted(
            splitting, key=lambda i: covariates[i][f]
        )
        for q in (0.25, 0.5, 0.75):
            cut = sorted_idx[int(q * (n - 1))]
            thr = covariates[cut][f]
            left = [i for i in splitting if covariates[i][f] <= thr]
            right = [i for i in splitting if covariates[i][f] > thr]
            if len(left) < min_leaf_n or len(right) < min_leaf_n:
                continue
            score = abs(
                _ate_for_indices(left, treatments, outcomes)
                - _ate_for_indices(right, treatments, outcomes)
            )
            if score > best_score:
                best_score = score
                best_feat = f
                best_thr = thr
                best_left = left
                best_right = right
    if best_feat < 0:
        cate = _ate_for_indices(estimation, treatments, outcomes)
        return _CFTreeNode(
            is_leaf=True, leaf_cate=cate, leaf_n=len(estimation),
        )
    est_left = [
        i for i in estimation if covariates[i][best_feat] <= best_thr
    ]
    est_right = [
        i for i in estimation if covariates[i][best_feat] > best_thr
    ]
    return _CFTreeNode(
        is_leaf=False,
        feature_index=best_feat,
        split_threshold=best_thr,
        left=_grow_tree(
            best_left, est_left, covariates,
            treatments, outcomes,
            max_depth=max_depth, min_leaf_n=min_leaf_n,
            rng=rng, depth=depth + 1,
        ),
        right=_grow_tree(
            best_right, est_right, covariates,
            treatments, outcomes,
            max_depth=max_depth, min_leaf_n=min_leaf_n,
            rng=rng, depth=depth + 1,
        ),
    )


def _tree_predict(
    tree: _CFTreeNode, x: list[float],
) -> float:
    if tree.is_leaf:
        return tree.leaf_cate or 0.0
    if tree.feature_index is None or tree.split_threshold is None:
        return tree.leaf_cate or 0.0
    if x[tree.feature_index] <= tree.split_threshold:
        return _tree_predict(tree.left, x) if tree.left else 0.0
    return _tree_predict(tree.right, x) if tree.right else 0.0


# ─────────────────────────────────────────────────────────────────────
# Forest
# ─────────────────────────────────────────────────────────────────────


class CausalForestReport(BaseModel):
    n_train: int
    n_trees: int
    feature_names: list[str]
    overall_ate: float
    cate_by_query: dict[str, float] = Field(default_factory=dict)
    rationale: str


class CausalForest(BaseModel):
    feature_names: list[str]
    trees: list[_CFTreeNode]
    n_train: int


def fit_causal_forest(
    *,
    covariates: list[list[float]],
    treatments: list[int],
    outcomes: list[float],
    feature_names: list[str] | None = None,
    n_trees: int = 50,
    max_depth: int = 4,
    min_leaf_n: int = 10,
    subsample_fraction: float = 0.5,
    honest_split: float = 0.5,
    seed: int = 7,
) -> CausalForest:
    """Fit an honest causal forest."""
    n = len(covariates)
    if n == 0:
        raise ValueError("covariates cannot be empty")
    if not (n == len(treatments) == len(outcomes)):
        raise ValueError(
            "covariates, treatments, outcomes must align")
    if not all(t in (0, 1) for t in treatments):
        raise ValueError("treatments must be 0 or 1")
    p = len(covariates[0])
    feature_names = feature_names or [f"x{i}" for i in range(p)]
    if len(feature_names) != p:
        raise ValueError(
            "feature_names length must match covariate width")
    rng = random.Random(seed)
    trees: list[_CFTreeNode] = []
    sub_size = max(2 * min_leaf_n + 2,
                   int(subsample_fraction * n))
    for _ in range(n_trees):
        idx = rng.sample(range(n), min(sub_size, n))
        rng.shuffle(idx)
        cut = max(min_leaf_n, int(honest_split * len(idx)))
        splitting = idx[:cut]
        estimation = idx[cut:]
        if len(estimation) < min_leaf_n:
            estimation = splitting
        tree = _grow_tree(
            splitting, estimation, covariates,
            treatments, outcomes,
            max_depth=max_depth, min_leaf_n=min_leaf_n, rng=rng,
        )
        trees.append(tree)
    return CausalForest(
        feature_names=list(feature_names),
        trees=trees, n_train=n,
    )


def predict_cate(forest: CausalForest, x: list[float]) -> float:
    """Forest CATE = mean of per-tree leaf CATEs."""
    if not forest.trees:
        return 0.0
    return sum(_tree_predict(t, x) for t in forest.trees) \
        / len(forest.trees)


def overall_ate(
    forest: CausalForest, covariates: list[list[float]],
) -> float:
    if not covariates:
        return 0.0
    return sum(predict_cate(forest, x) for x in covariates) \
        / len(covariates)


def report_causal_forest(
    forest: CausalForest, *,
    queries: dict[str, list[float]] | None = None,
    covariates: list[list[float]] | None = None,
) -> CausalForestReport:
    queries = queries or {}
    cate_by_query = {
        k: round(predict_cate(forest, x), 6)
        for k, x in queries.items()
    }
    overall = (
        round(overall_ate(forest, covariates), 6)
        if covariates is not None else 0.0
    )
    return CausalForestReport(
        n_train=forest.n_train,
        n_trees=len(forest.trees),
        feature_names=list(forest.feature_names),
        overall_ate=overall,
        cate_by_query=cate_by_query,
        rationale=(
            f"Causal forest over {forest.n_train} subjects + "
            f"{len(forest.trees)} honest trees; overall ATE "
            f"{overall:.4f}."
        ),
    )
