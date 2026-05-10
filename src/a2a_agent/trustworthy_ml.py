"""Phase 16.F2 - Trustworthy-ML formalisms.

Three pure-Python tools that close the research-grade gap:

  1. **Split-conformal multi-class prediction sets** (Vovk et al. 2005,
     Romano et al. 2020). Given calibration scores and a desired
     coverage 1-alpha, produces prediction sets per test instance and
     verifies marginal coverage.

  2. **Per-subgroup calibration tension** (Pleiss et al. 2017).
     Demonstrates the impossibility of simultaneous calibration +
     equalised generalised FPR/FNR across subgroups for a non-trivial
     classifier. Returns the per-subgroup calibration error + the gap
     that the impossibility theorem says must exist.

  3. **Selective classification risk-coverage curve** (El-Yaniv & Wiener
     2010). Given a sequence of (prediction, confidence, true_label)
     triples, sweeps the abstain threshold tau, and returns the
     coverage(tau) -> risk(tau) curve. The 'optimal-stopping' point
     minimises risk*(1-coverage).

Pure-Python, deterministic, stdlib-only. No NumPy/SciPy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Split-conformal multi-class
# ─────────────────────────────────────────────────────────────────────


class ConformalMultiClassReport(BaseModel):
    n_calibration: int = Field(ge=0)
    n_test: int = Field(ge=0)
    target_coverage: float = Field(ge=0.0, le=1.0)
    quantile_threshold: float
    empirical_coverage_test: float = Field(ge=0.0, le=1.0)
    average_set_size: float = Field(ge=0.0)
    prediction_sets: list[list[str]]
    rationale: str


def split_conformal_multiclass(
    *,
    cal_probabilities: list[dict[str, float]],
    cal_true_labels: list[str],
    test_probabilities: list[dict[str, float]],
    test_true_labels: list[str] | None = None,
    target_coverage: float = 0.90,
) -> ConformalMultiClassReport:
    """Split-conformal multi-class with the LAC nonconformity score
    (Romano et al. 2020): score(x, y) = 1 - p_hat(y | x).

    Args:
        cal_probabilities: per-instance class probability dicts on the
            held-out calibration set.
        cal_true_labels: true labels for the calibration instances.
        test_probabilities: per-instance class probability dicts on
            the test instances.
        test_true_labels: optional - when provided we report empirical
            coverage on the test set.
        target_coverage: 1 - alpha (default 0.90 = 90% coverage).

    Returns:
        ConformalMultiClassReport with quantile + per-instance
        prediction sets + empirical coverage.
    """
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be in (0,1)")
    if len(cal_probabilities) != len(cal_true_labels):
        raise ValueError("cal_probabilities and cal_true_labels must be aligned")
    if len(cal_probabilities) == 0:
        raise ValueError("calibration set cannot be empty")

    # Compute calibration nonconformity scores
    cal_scores: list[float] = []
    for probs, y in zip(cal_probabilities, cal_true_labels):
        p_y = float(probs.get(y, 0.0))
        cal_scores.append(1.0 - p_y)

    # The conformal quantile uses (n+1)*(1-alpha)/n with rank
    n_cal = len(cal_scores)
    q_index = math.ceil((n_cal + 1) * target_coverage) - 1
    q_index = min(max(q_index, 0), n_cal - 1)
    cal_scores_sorted = sorted(cal_scores)
    quantile_threshold = cal_scores_sorted[q_index]

    # Build prediction sets on the test instances
    prediction_sets: list[list[str]] = []
    for probs in test_probabilities:
        pset: list[str] = sorted(
            [c for c, p in probs.items() if (1.0 - p) <= quantile_threshold]
        )
        if not pset and probs:
            pset = [max(probs.items(), key=lambda kv: kv[1])[0]]
        prediction_sets.append(pset)

    # Empirical coverage if true labels supplied
    if test_true_labels is not None:
        if len(test_true_labels) != len(test_probabilities):
            raise ValueError(
                "test_true_labels must align with test_probabilities")
        n_covered = sum(
            1 for pset, y in zip(prediction_sets, test_true_labels)
            if y in pset
        )
        empirical_coverage = (
            n_covered / len(test_true_labels)
            if test_true_labels else 0.0
        )
    else:
        empirical_coverage = 0.0

    avg_set_size = (
        sum(len(p) for p in prediction_sets) / len(prediction_sets)
        if prediction_sets else 0.0
    )

    return ConformalMultiClassReport(
        n_calibration=n_cal,
        n_test=len(test_probabilities),
        target_coverage=target_coverage,
        quantile_threshold=round(quantile_threshold, 6),
        empirical_coverage_test=round(empirical_coverage, 6),
        average_set_size=round(avg_set_size, 4),
        prediction_sets=prediction_sets,
        rationale=(
            f"Split-conformal multi-class with LAC score "
            f"(1 - p_hat(y|x)). Quantile q="
            f"{quantile_threshold:.4f} from n={n_cal} calibration "
            f"scores; target coverage {target_coverage*100:.0f}% "
            f"(empirical {empirical_coverage*100:.2f}%)."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Per-subgroup calibration tension (Pleiss 2017)
# ─────────────────────────────────────────────────────────────────────


class SubgroupCalibrationReport(BaseModel):
    n: int
    subgroup_calibration_error: dict[str, float]
    subgroup_generalised_fpr: dict[str, float]
    subgroup_generalised_fnr: dict[str, float]
    impossibility_witnessed: bool
    rationale: str


def per_subgroup_calibration_tension(
    *,
    predictions: list[float],
    labels: list[int],
    subgroup_labels: list[str],
    n_bins: int = 10,
) -> SubgroupCalibrationReport:
    """Demonstrate the Pleiss et al. 2017 impossibility:
    for any non-trivial classifier (where subgroup base rates differ),
    you cannot simultaneously have (a) calibration within each subgroup
    and (b) equal generalised FPR + FNR across subgroups.

    The function returns per-subgroup ECE + generalised FPR (mean
    predicted score on negatives) + generalised FNR
    (1 - mean predicted score on positives), and flags
    impossibility_witnessed=True when the FPR and FNR gaps exceed
    a small numerical tolerance simultaneously with non-trivial
    base-rate variation.
    """
    if not predictions:
        raise ValueError("predictions cannot be empty")
    if not (len(predictions) == len(labels) == len(subgroup_labels)):
        raise ValueError(
            "predictions / labels / subgroup_labels must align")

    by_sg: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for p, y, sg in zip(predictions, labels, subgroup_labels):
        by_sg[sg].append((p, y))

    sg_ece: dict[str, float] = {}
    sg_gfpr: dict[str, float] = {}
    sg_gfnr: dict[str, float] = {}
    base_rates: dict[str, float] = {}
    for sg, rows in by_sg.items():
        if not rows:
            continue
        # ECE
        bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
        for p, y in rows:
            idx = min(int(p * n_bins), n_bins - 1)
            bins[idx].append((p, y))
        total = len(rows)
        ece = 0.0
        for b in bins:
            if not b:
                continue
            avg_p = sum(p for p, _ in b) / len(b)
            avg_y = sum(y for _, y in b) / len(b)
            ece += (len(b) / total) * abs(avg_p - avg_y)
        sg_ece[sg] = round(ece, 6)
        # Generalised FPR / FNR (Pleiss 2017 definition)
        negatives = [p for p, y in rows if y == 0]
        positives = [p for p, y in rows if y == 1]
        gfpr = sum(negatives) / len(negatives) if negatives else 0.0
        gfnr = (1.0 - (sum(positives) / len(positives))
                if positives else 0.0)
        sg_gfpr[sg] = round(gfpr, 6)
        sg_gfnr[sg] = round(gfnr, 6)
        base_rates[sg] = (
            len(positives) / total if total else 0.0
        )

    base_rate_values = list(base_rates.values())
    if len(base_rate_values) >= 2:
        base_rate_spread = max(base_rate_values) - min(base_rate_values)
    else:
        base_rate_spread = 0.0
    if len(sg_gfpr) >= 2:
        fpr_spread = max(sg_gfpr.values()) - min(sg_gfpr.values())
        fnr_spread = max(sg_gfnr.values()) - min(sg_gfnr.values())
    else:
        fpr_spread = fnr_spread = 0.0
    impossibility = (
        base_rate_spread > 0.05
        and (fpr_spread > 0.02 or fnr_spread > 0.02)
    )

    return SubgroupCalibrationReport(
        n=len(predictions),
        subgroup_calibration_error=sg_ece,
        subgroup_generalised_fpr=sg_gfpr,
        subgroup_generalised_fnr=sg_gfnr,
        impossibility_witnessed=impossibility,
        rationale=(
            f"Pleiss 2017 tension: with base-rate spread "
            f"{base_rate_spread:.3f} across {len(by_sg)} subgroups, "
            f"and FPR/FNR spreads {fpr_spread:.3f}/"
            f"{fnr_spread:.3f}, "
            f"impossibility_witnessed={impossibility}."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Selective classification risk-coverage (El-Yaniv & Wiener 2010)
# ─────────────────────────────────────────────────────────────────────


class SelectiveClassificationPoint(BaseModel):
    threshold: float
    coverage: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0)


class SelectiveClassificationReport(BaseModel):
    n: int
    n_thresholds: int
    points: list[SelectiveClassificationPoint]
    optimal_threshold: float
    optimal_coverage: float
    optimal_risk: float
    rationale: str


def selective_classification_curve(
    *,
    predictions: list[float],
    confidences: list[float],
    labels: list[int],
    n_thresholds: int = 21,
) -> SelectiveClassificationReport:
    """Sweep the abstain threshold tau over the confidence quantiles
    and return the coverage(tau) -> risk(tau) curve.

    Optimal threshold minimises risk * (1 - coverage) - a balanced
    proxy for "abstain when uncertain, otherwise commit and accept
    the risk".
    """
    if not predictions:
        raise ValueError("predictions cannot be empty")
    if not (len(predictions) == len(confidences) == len(labels)):
        raise ValueError(
            "predictions / confidences / labels must align")
    if n_thresholds < 2:
        raise ValueError("n_thresholds must be >= 2")
    n = len(predictions)
    sorted_conf = sorted(confidences)
    points: list[SelectiveClassificationPoint] = []
    optimum = (1.0, 1.0, 1.0)   # (objective, coverage, risk)
    optimal_threshold = 0.0
    for i in range(n_thresholds):
        q = i / (n_thresholds - 1)
        idx = min(int(q * (n - 1)), n - 1)
        tau = sorted_conf[idx]
        kept = [
            (p, y) for p, c, y in zip(predictions, confidences, labels)
            if c >= tau
        ]
        coverage = len(kept) / n
        if not kept:
            risk = 0.0
        else:
            errors = sum(
                1 for p, y in kept if (p >= 0.5) != bool(y)
            )
            risk = errors / len(kept)
        points.append(SelectiveClassificationPoint(
            threshold=round(tau, 6),
            coverage=round(coverage, 6),
            risk=round(risk, 6),
        ))
        objective = risk * (1.0 - coverage)
        if objective < optimum[0]:
            optimum = (objective, coverage, risk)
            optimal_threshold = tau
    return SelectiveClassificationReport(
        n=n,
        n_thresholds=n_thresholds,
        points=points,
        optimal_threshold=round(optimal_threshold, 6),
        optimal_coverage=round(optimum[1], 6),
        optimal_risk=round(optimum[2], 6),
        rationale=(
            f"Selective classification curve over {n_thresholds} "
            f"confidence quantiles on n={n} predictions. Optimal "
            f"tradeoff at tau={optimal_threshold:.3f}: "
            f"coverage={optimum[1]*100:.1f}%, "
            f"risk={optimum[2]*100:.2f}%."
        ),
    )
