"""Phase 17.AX - Active learning acquisition functions.

Pool-based active learning over a labelled + unlabelled set. Four
classical acquisition functions:

  - **least_confidence** - score = 1 - max_y P(y|x). Pick the
    instance whose top class is least confident.
  - **margin** - score = -(P(y_top1|x) - P(y_top2|x)). Pick the
    instance whose top-1 vs top-2 gap is smallest.
  - **entropy** - score = -sum_y P(y|x) log P(y|x). Pick the
    highest-entropy instance.
  - **query_by_committee** (Seung 1992) - score = variance of the
    top class probability across an ensemble. Pick the instance
    where the ensemble disagrees most.

Returns the top-k instance indices to label next + their scores.
Pure-Python deterministic.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal

from pydantic import BaseModel, Field


_Strategy = Literal[
    "least_confidence", "margin", "entropy", "query_by_committee",
]


class ActiveLearningPick(BaseModel):
    instance_index: int
    score: float
    rank: int


class ActiveLearningReport(BaseModel):
    strategy: _Strategy
    n_unlabelled: int
    k_selected: int
    picks: list[ActiveLearningPick]
    rationale: str


# ─────────────────────────────────────────────────────────────────────
# Acquisition score functions
# ─────────────────────────────────────────────────────────────────────


def _least_confidence(probs: list[float]) -> float:
    return 1.0 - max(probs)


def _margin(probs: list[float]) -> float:
    sorted_p = sorted(probs, reverse=True)
    if len(sorted_p) < 2:
        return -1.0
    return -(sorted_p[0] - sorted_p[1])


def _entropy(probs: list[float]) -> float:
    out = 0.0
    for p in probs:
        if p > 0:
            out -= p * math.log(p)
    return out


def _committee_disagreement(
    probs_per_member: list[list[float]],
) -> float:
    """Variance of the top-class probability across committee
    members. Robust + simple disagreement measure."""
    if not probs_per_member:
        return 0.0
    top_probs = [max(p) for p in probs_per_member]
    mean = sum(top_probs) / len(top_probs)
    var = (
        sum((tp - mean) ** 2 for tp in top_probs)
        / len(top_probs)
    )
    return var


# ─────────────────────────────────────────────────────────────────────
# Top-k selection
# ─────────────────────────────────────────────────────────────────────


def select_top_k_to_label(
    *,
    unlabelled_probs: list[list[float]],
    strategy: _Strategy = "entropy",
    k: int = 5,
) -> ActiveLearningReport:
    """Single-model strategies (least_confidence / margin / entropy)
    take a probability per class per instance."""
    if strategy not in (
        "least_confidence", "margin", "entropy",
    ):
        raise ValueError(
            f"strategy {strategy!r} requires the committee "
            "selector"
        )
    if not unlabelled_probs:
        raise ValueError("unlabelled_probs cannot be empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    score_fn = {
        "least_confidence": _least_confidence,
        "margin": _margin,
        "entropy": _entropy,
    }[strategy]
    scored: list[tuple[int, float]] = []
    for i, probs in enumerate(unlabelled_probs):
        scored.append((i, score_fn(probs)))
    # Higher score = more useful to label
    scored.sort(key=lambda t: (-t[1], t[0]))
    top = scored[:k]
    picks = [
        ActiveLearningPick(
            instance_index=idx, score=round(score, 6), rank=r,
        )
        for r, (idx, score) in enumerate(top)
    ]
    return ActiveLearningReport(
        strategy=strategy,
        n_unlabelled=len(unlabelled_probs),
        k_selected=len(picks), picks=picks,
        rationale=(
            f"Selected top-{len(picks)} of {len(unlabelled_probs)} "
            f"unlabelled instances under `{strategy}`."
        ),
    )


def select_top_k_by_committee(
    *,
    unlabelled_committee_probs: list[list[list[float]]],
    k: int = 5,
) -> ActiveLearningReport:
    """Query-by-committee acquisition. ``unlabelled_committee_probs``
    is shape (n_instances, n_committee, n_classes)."""
    if not unlabelled_committee_probs:
        raise ValueError("unlabelled_committee_probs cannot be empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    scored: list[tuple[int, float]] = []
    for i, members in enumerate(unlabelled_committee_probs):
        scored.append((i, _committee_disagreement(members)))
    scored.sort(key=lambda t: (-t[1], t[0]))
    top = scored[:k]
    picks = [
        ActiveLearningPick(
            instance_index=idx, score=round(score, 6), rank=r,
        )
        for r, (idx, score) in enumerate(top)
    ]
    return ActiveLearningReport(
        strategy="query_by_committee",
        n_unlabelled=len(unlabelled_committee_probs),
        k_selected=len(picks), picks=picks,
        rationale=(
            f"Query-by-committee top-{len(picks)} on "
            f"{len(unlabelled_committee_probs)} unlabelled "
            f"instances; disagreement = variance of top-class "
            f"probability across committee members."
        ),
    )
