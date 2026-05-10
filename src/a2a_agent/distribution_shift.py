"""Phase 17.N - Distribution shift formalism.

Three pure-Python tools that detect + correct for distribution shift:

  1. **Two-sample Kolmogorov-Smirnov test** for univariate covariate
     shift between a reference (training) sample and a current
     (production) sample. Returns the K-S statistic + an asymptotic
     two-sided p-value.

  2. **Lipton 2018 BBSE label-shift correction** - Black-Box Shift
     Estimation. Given a confusion matrix on the source + the
     observed predicted-label distribution on the target, recover
     the target's label marginals via a small linear solve.

  3. **Energy-based OOD detector** - Liu et al. 2020 "Energy-Based
     Out-of-Distribution Detection". For a logistic / softmax model,
     compute :math:`E(x) = -log sum_k exp(z_k(x))`. Lower E -> in-
     distribution; higher E -> OOD.

Pure-Python, deterministic, stdlib-only.
"""

from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────
# Two-sample Kolmogorov-Smirnov
# ─────────────────────────────────────────────────────────────────────


class KSTestReport(BaseModel):
    n_reference: int = Field(ge=0)
    n_current: int = Field(ge=0)
    ks_statistic: float = Field(ge=0.0, le=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    drift_severity: str
    rationale: str


def _ks_p_value(d: float, n_eff: float) -> float:
    """Asymptotic Kolmogorov-Smirnov two-sided p-value:
    P(D > d) ~= 2 * sum_{k=1..} (-1)^(k-1) exp(-2 k^2 lambda^2)
    where lambda = (sqrt(n_eff) + 0.12 + 0.11/sqrt(n_eff)) * d.
    """
    if d <= 0.0 or n_eff <= 0:
        return 1.0
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d
    j = 1
    p = 0.0
    last = float("inf")
    while j < 101:
        term = 2.0 * ((-1) ** (j - 1)) * math.exp(-2.0 * (j ** 2) * (lam ** 2))
        p += term
        if abs(term) < 1e-10 and abs(term) >= last:
            break
        last = abs(term)
        j += 1
    return max(0.0, min(1.0, p))


def two_sample_ks_test(
    *,
    reference: list[float],
    current: list[float],
) -> KSTestReport:
    """Two-sample Kolmogorov-Smirnov between two univariate samples.

    Drift severity is reported as `"none"` / `"low"` / `"moderate"` /
    `"high"` based on the K-S statistic (0.10 / 0.20 / 0.30 cuts).
    """
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("each sample must have >= 2 observations")
    n_ref, n_cur = len(reference), len(current)
    sorted_ref = sorted(reference)
    sorted_cur = sorted(current)

    # Walk both sorted arrays; CDF difference at each unique value.
    i_ref = i_cur = 0
    max_d = 0.0
    while i_ref < n_ref or i_cur < n_cur:
        v_ref = sorted_ref[i_ref] if i_ref < n_ref else float("inf")
        v_cur = sorted_cur[i_cur] if i_cur < n_cur else float("inf")
        v = min(v_ref, v_cur)
        while i_ref < n_ref and sorted_ref[i_ref] <= v:
            i_ref += 1
        while i_cur < n_cur and sorted_cur[i_cur] <= v:
            i_cur += 1
        cdf_ref = i_ref / n_ref
        cdf_cur = i_cur / n_cur
        d = abs(cdf_ref - cdf_cur)
        if d > max_d:
            max_d = d
    n_eff = n_ref * n_cur / (n_ref + n_cur)
    p = _ks_p_value(max_d, n_eff)
    if max_d < 0.10:
        sev = "none"
    elif max_d < 0.20:
        sev = "low"
    elif max_d < 0.30:
        sev = "moderate"
    else:
        sev = "high"
    return KSTestReport(
        n_reference=n_ref, n_current=n_cur,
        ks_statistic=round(max_d, 6),
        p_value=round(p, 6),
        drift_severity=sev,
        rationale=(
            f"K-S statistic D={max_d:.4f} on n_ref={n_ref} + "
            f"n_cur={n_cur}; asymptotic two-sided p={p:.4f}; "
            f"drift severity tier `{sev}`."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Lipton 2018 BBSE label-shift correction
# ─────────────────────────────────────────────────────────────────────


class BBSELabelShiftReport(BaseModel):
    classes: list[str]
    source_prior: dict[str, float]
    estimated_target_prior: dict[str, float]
    importance_weights: dict[str, float]
    rationale: str


def _solve_linear_2x2(A: list[list[float]], b: list[float]) -> list[float]:
    a, c = A[0]
    d, e = A[1]
    det = a * e - c * d
    if abs(det) < 1e-12:
        raise ValueError("singular confusion matrix")
    return [
        (e * b[0] - c * b[1]) / det,
        (a * b[1] - d * b[0]) / det,
    ]


def _solve_linear_3x3(A: list[list[float]], b: list[float]
                      ) -> list[float]:
    """Cramer's rule for 3x3."""
    def det3(M):
        return (
            M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
            - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
            + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
        )
    D = det3(A)
    if abs(D) < 1e-12:
        raise ValueError("singular confusion matrix")
    out = []
    for col in range(3):
        Mc = [row[:] for row in A]
        for row in range(3):
            Mc[row][col] = b[row]
        out.append(det3(Mc) / D)
    return out


def lipton_bbse_label_shift(
    *,
    classes: list[str],
    source_confusion_matrix: list[list[float]],
    source_prior: dict[str, float],
    target_predicted_distribution: dict[str, float],
) -> BBSELabelShiftReport:
    """Lipton 2018 Black-Box Shift Estimation.

    Suppose source distribution has class prior :math:`p_s(y)` and
    confusion matrix :math:`M_{kj} = P_s(\\hat{y}=k | y=j)`. On the
    target you only see :math:`q(\\hat{y})`. Under the label-shift
    assumption :math:`p_t(x | y) = p_s(x | y)`, the target prior is

        p_t(y) = M^{-1} q(\\hat{y}).

    Args:
        classes: list of class labels (any string).
        source_confusion_matrix: row-stochastic; rows are predicted
            classes, columns are true classes. Must be square in
            ``len(classes)``.
        source_prior: source label prior ``p_s(y)``. Used only for
            computing importance weights ``p_t(y) / p_s(y)``.
        target_predicted_distribution: empirical
            :math:`q(\\hat{y})` on the target.
    """
    n = len(classes)
    if n not in (2, 3):
        raise ValueError(
            "BBSE solver supports 2 or 3 classes (small-n regime)")
    if (len(source_confusion_matrix) != n
        or any(len(r) != n for r in source_confusion_matrix)):
        raise ValueError(
            "source_confusion_matrix must be n x n")
    q = [target_predicted_distribution.get(c, 0.0) for c in classes]
    if abs(sum(q) - 1.0) > 1e-3:
        raise ValueError(
            "target_predicted_distribution must sum to ~1.0")

    M = [list(r) for r in source_confusion_matrix]
    if n == 2:
        sol = _solve_linear_2x2(M, q)
    else:
        sol = _solve_linear_3x3(M, q)
    # Project onto the simplex (clip to [0,1] then renormalize)
    proj = [max(0.0, min(1.0, v)) for v in sol]
    total = sum(proj)
    if total <= 0:
        raise ValueError("solution falls outside simplex completely")
    proj = [v / total for v in proj]

    target_prior = dict(zip(classes, [round(v, 6) for v in proj]))
    weights: dict[str, float] = {}
    for c in classes:
        ps = source_prior.get(c, 0.0)
        if ps > 0:
            weights[c] = round(target_prior[c] / ps, 6)
        else:
            weights[c] = float("inf") if target_prior[c] > 0 else 0.0
    return BBSELabelShiftReport(
        classes=classes,
        source_prior={c: round(source_prior.get(c, 0.0), 6)
                      for c in classes},
        estimated_target_prior=target_prior,
        importance_weights=weights,
        rationale=(
            f"BBSE recovered target prior by inverting the {n}x{n} "
            f"source confusion matrix; importance weights = "
            f"target_prior / source_prior."
        ),
    )


# ─────────────────────────────────────────────────────────────────────
# Energy-based OOD detector (Liu 2020)
# ─────────────────────────────────────────────────────────────────────


class EnergyOODReport(BaseModel):
    n: int
    energy_scores: list[float]
    in_distribution_threshold: float
    n_ood: int
    n_in: int
    rationale: str


def _logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def energy_based_ood_score(
    *,
    logits_per_instance: list[list[float]],
    in_distribution_threshold: float | None = None,
) -> EnergyOODReport:
    """Compute the Liu 2020 energy score
    :math:`E(x) = -\\log\\sum_k \\exp(z_k(x))` per instance and flag
    OOD when ``E(x) > threshold``.

    If ``in_distribution_threshold`` is ``None``, use the 95th
    percentile of the cohort's own energy as the cutoff (a self-
    calibrated threshold suitable for first-pass deployment).
    """
    if not logits_per_instance:
        raise ValueError("logits_per_instance cannot be empty")
    energies = [-_logsumexp(z) for z in logits_per_instance]
    if in_distribution_threshold is None:
        sorted_e = sorted(energies)
        idx = max(0, min(len(sorted_e) - 1,
                          int(0.95 * (len(sorted_e) - 1))))
        threshold = sorted_e[idx]
    else:
        threshold = in_distribution_threshold
    n_ood = sum(1 for e in energies if e > threshold)
    return EnergyOODReport(
        n=len(energies),
        energy_scores=[round(e, 6) for e in energies],
        in_distribution_threshold=round(threshold, 6),
        n_ood=n_ood,
        n_in=len(energies) - n_ood,
        rationale=(
            f"Energy score E(x) = -logsumexp(z(x)) per Liu 2020; "
            f"threshold {threshold:.4f}; {n_ood}/{len(energies)} "
            f"flagged OOD."
        ),
    )
