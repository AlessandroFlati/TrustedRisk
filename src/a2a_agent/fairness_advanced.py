"""Phase 13.2 I1 -- Equalized Odds + Demographic Parity audit.

Extends the existing per-subgroup multiplier audit (LACE-driven, see
`mcp_server/tools/fairness_audit.py`) with the **classification-style**
fairness metrics that regulators actually cite under the EEOC's 4/5ths
rule + the EU AI Act high-risk classifier requirements:

  - **Demographic Parity (DP)**: P(predicted_positive | g_a) ==
    P(predicted_positive | g_b) for every pair of subgroups.
  - **Equalized Odds (EOO)**: TPR + FPR parity across subgroups.
  - **Chi-squared independence test**: outcome ⊥ subgroup.
  - **Bootstrap CI on selection rate**: 95% CI on each subgroup's
    rate so the auditor can see when a gap is significant vs noisy.

References:
  - Hardt M, Price E, Srebro N. *Equality of Opportunity in Supervised
    Learning*. NeurIPS 2016.
  - EEOC Uniform Guidelines on Employee Selection Procedures (1978) --
    4/5ths Rule.
  - Verma S, Rubin J. *Fairness Definitions Explained*. FairWare 2018.
  - Chouldechova A. *Fair Prediction with Disparate Impact*. Big Data 2017.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from shared.schemas import (
    FairnessAdvancedReport,
    SubgroupFairnessMetric,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _bootstrap_ci(
    successes: int, n: int, *, n_bootstrap: int = 1000,
    alpha: float = 0.05, seed: int = 4242,
) -> tuple[float, float]:
    """Bootstrap 95 % CI on a binomial proportion. Falls back to the
    Wilson interval when n is tiny."""
    if n == 0:
        return (0.0, 0.0)
    if n < 5:
        return _wilson_interval(successes, n, alpha=alpha)
    rng = random.Random(seed)
    sample = [1] * successes + [0] * (n - successes)
    rates: list[float] = []
    for _ in range(n_bootstrap):
        boot = [sample[rng.randrange(n)] for _ in range(n)]
        rates.append(sum(boot) / n)
    rates.sort()
    lo = rates[int((alpha / 2) * n_bootstrap)]
    hi = rates[int((1 - alpha / 2) * n_bootstrap) - 1]
    return (lo, hi)


def _wilson_interval(
    successes: int, n: int, *, alpha: float = 0.05,
) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else 1.96
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _chi2_p(observed: list[list[int]]) -> float:
    """Pearson chi-squared p-value on a 2×k contingency table.

    Rows: [n_predicted_positive_per_subgroup,
           n_predicted_negative_per_subgroup].
    Returns p-value via the chi-squared survival function with
    df = (rows-1)(cols-1) = k-1 (for 2 rows). Uses the regularised
    upper-incomplete gamma series (no SciPy dependency).
    """
    n_rows = len(observed)
    n_cols = len(observed[0]) if observed else 0
    if n_rows < 2 or n_cols < 2:
        return 1.0
    row_totals = [sum(row) for row in observed]
    col_totals = [sum(observed[i][j] for i in range(n_rows))
                       for j in range(n_cols)]
    grand = sum(row_totals)
    if grand == 0:
        return 1.0
    chi2 = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            expected = row_totals[i] * col_totals[j] / grand
            if expected > 0:
                diff = observed[i][j] - expected
                chi2 += diff * diff / expected
    df = (n_rows - 1) * (n_cols - 1)
    return _chi2_sf(chi2, df)


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-squared(df). Uses incomplete gamma."""
    if x <= 0:
        return 1.0
    if df <= 0:
        return 1.0
    # P(X > x) = 1 - regularised lower incomplete gamma(df/2, x/2)
    a = df / 2.0
    z = x / 2.0
    return 1.0 - _gamma_lower_reg(a, z)


def _gamma_lower_reg(a: float, x: float) -> float:
    """Regularised lower incomplete gamma via series (small x) /
    continued fraction (large x). Sufficient precision for p-values."""
    if x < 0 or a <= 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < a + 1:
        # Series expansion
        term = 1.0 / a
        total = term
        for n in range(1, 200):
            term *= x / (a + n)
            total += term
            if abs(term) < 1e-12 * abs(total):
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    return 1.0 - q


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def _classify_posture(max_gap: float) -> str:
    if max_gap < 0.05:
        return "fair"
    if max_gap < 0.10:
        return "monitor"
    if max_gap < 0.20:
        return "investigate"
    return "violation"


def compute_fairness_advanced(
    sensitive_attribute: str,
    rows: Sequence[dict],
    *,
    subgroup_field: str = "subgroup",
    predicted_field: str = "predicted_positive",
    actual_field: str = "actual_positive",
    n_bootstrap: int = 1000,
    seed: int = 4242,
) -> FairnessAdvancedReport:
    """Audit demographic-parity + equalized-odds across subgroups.

    Args:
        sensitive_attribute: free-text label of the protected attribute
            (e.g. "race", "sex", "insurance_band").
        rows: list of dicts; each row carries `subgroup` (str),
            `predicted_positive` (0/1), `actual_positive` (0/1).
        subgroup_field/predicted_field/actual_field: override field
            names if the caller's row shape differs.

    Returns:
        FairnessAdvancedReport.
    """
    if len(rows) < 2:
        raise ValueError("rows must have ≥ 2 observations")

    by_sub: dict[str, list[dict]] = {}
    for r in rows:
        g = str(r.get(subgroup_field, ""))
        if not g:
            continue
        by_sub.setdefault(g, []).append(r)

    if len(by_sub) < 2:
        raise ValueError(
            "fairness audit requires ≥ 2 distinct subgroups in rows"
        )

    metrics: list[SubgroupFairnessMetric] = []
    contingency: list[list[int]] = [[], []]   # [predicted_pos, predicted_neg]
    for sub, sub_rows in by_sub.items():
        n = len(sub_rows)
        n_pos = sum(int(bool(r.get(actual_field, 0))) for r in sub_rows)
        n_pred = sum(int(bool(r.get(predicted_field, 0)))
                          for r in sub_rows)
        n_tp = sum(
            1 for r in sub_rows
            if int(bool(r.get(predicted_field, 0)))
            and int(bool(r.get(actual_field, 0)))
        )
        n_fp = sum(
            1 for r in sub_rows
            if int(bool(r.get(predicted_field, 0)))
            and not int(bool(r.get(actual_field, 0)))
        )
        n_neg = n - n_pos
        sel_rate = n_pred / n if n else 0.0
        tpr = (n_tp / n_pos) if n_pos else 0.0
        fpr = (n_fp / n_neg) if n_neg else 0.0
        ci_lo, ci_hi = _bootstrap_ci(
            n_pred, n, n_bootstrap=n_bootstrap, seed=seed,
        )

        metrics.append(SubgroupFairnessMetric(
            subgroup=sub, n=n, n_positive=n_pos,
            n_predicted_positive=n_pred,
            n_true_positive=n_tp, n_false_positive=n_fp,
            selection_rate=round(sel_rate, 4),
            true_positive_rate=round(tpr, 4),
            false_positive_rate=round(fpr, 4),
            selection_rate_ci95_lower=round(ci_lo, 4),
            selection_rate_ci95_upper=round(ci_hi, 4),
        ))
        contingency[0].append(n_pred)
        contingency[1].append(n - n_pred)

    sel_rates = [m.selection_rate for m in metrics]
    tpr_rates = [m.true_positive_rate for m in metrics
                      if m.n_positive > 0]
    fpr_rates = [m.false_positive_rate for m in metrics
                      if (m.n - m.n_positive) > 0]

    dp_max_gap = (max(sel_rates) - min(sel_rates)) if sel_rates else 0.0
    eoo_tpr_gap = (max(tpr_rates) - min(tpr_rates)) if tpr_rates else 0.0
    eoo_fpr_gap = (max(fpr_rates) - min(fpr_rates)) if fpr_rates else 0.0

    chi2_p: float | None
    try:
        chi2_p = _chi2_p(contingency)
    except (ValueError, ZeroDivisionError):
        chi2_p = None

    posture = _classify_posture(
        max(dp_max_gap, eoo_tpr_gap, eoo_fpr_gap),
    )
    rationale = (
        f"Sensitive attribute = {sensitive_attribute!r}; "
        f"{len(metrics)} subgroups; n = {len(rows)}. "
        f"DP max gap {dp_max_gap:.3f}; EOO TPR gap {eoo_tpr_gap:.3f}; "
        f"EOO FPR gap {eoo_fpr_gap:.3f}; "
        f"chi² p = {chi2_p:.4f}. " if chi2_p is not None else (
            f"Sensitive attribute = {sensitive_attribute!r}; "
            f"{len(metrics)} subgroups; n = {len(rows)}. "
            f"DP max gap {dp_max_gap:.3f}; EOO TPR gap {eoo_tpr_gap:.3f}; "
            f"EOO FPR gap {eoo_fpr_gap:.3f}. "
        )
    ) + f"Posture = {posture}."

    return FairnessAdvancedReport(
        sensitive_attribute=sensitive_attribute,
        n_subgroups=len(metrics), n_total=len(rows),
        metrics=metrics,
        demographic_parity_max_gap=round(dp_max_gap, 4),
        equalized_odds_tpr_max_gap=round(eoo_tpr_gap, 4),
        equalized_odds_fpr_max_gap=round(eoo_fpr_gap, 4),
        chi2_independence_p_value=(
            round(chi2_p, 6) if chi2_p is not None else None
        ),
        posture=posture,                                  # type: ignore[arg-type]
        rationale=rationale,
        references=[
            "Hardt M, Price E, Srebro N. Equality of Opportunity in "
            "Supervised Learning. NeurIPS 2016.",
            "Verma S, Rubin J. Fairness Definitions Explained. "
            "FairWare 2018.",
            "Chouldechova A. Fair Prediction with Disparate Impact. "
            "Big Data 2017.",
            "EEOC Uniform Guidelines (1978) -- 4/5ths Rule.",
        ],
    )
