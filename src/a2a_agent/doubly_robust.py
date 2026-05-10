"""Phase 17.AU - Doubly-robust ATE estimators.

Three estimators of the average treatment effect (ATE) for a binary
treatment ``T`` and continuous/binary outcome ``Y``:

  - **IPW** (Horvitz-Thompson 1952 + Rosenbaum-Rubin 1983) - weights
    each observation by the inverse propensity:
        ATE_IPW = mean[T*Y / e(X)] - mean[(1-T)*Y / (1 - e(X))]

  - **G-formula** (a.k.a. outcome regression) - fits mu_1(X), mu_0(X)
    and averages mu_1(X) - mu_0(X) over the population.

  - **AIPW** (Robins, Rotnitzky, Zhao 1994) - augmented IPW; consistent
    if EITHER the propensity OR the outcome model is correct
    ("doubly robust"):
        ATE_AIPW = mean[ T*(Y - mu_1)/e + mu_1
                       - (1-T)*(Y - mu_0)/(1-e) - mu_0 ]

Pure-Python no NumPy. Propensity + outcome models are passed in as
plain ``Callable[[features], probability_or_value]``; we never fit
them here - the caller supplies the cross-fitted predictions
(canonical for double machine learning).
"""

from __future__ import annotations

import math
from typing import Callable, Iterable

from pydantic import BaseModel, Field


_FeatureFn = Callable[[dict[str, float]], float]


class DoublyRobustReport(BaseModel):
    n: int = Field(ge=0)
    ate_ipw: float
    ate_g_formula: float
    ate_aipw: float
    se_aipw: float = Field(ge=0.0)
    n_treated: int = Field(ge=0)
    n_control: int = Field(ge=0)
    propensity_min: float
    propensity_max: float
    rationale: str


def _clip(p: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return max(lo, min(hi, p))


def estimate_ate_doubly_robust(
    *,
    features: list[dict[str, float]],
    treatments: list[int],
    outcomes: list[float],
    propensity_fn: _FeatureFn,
    outcome_fn_treated: _FeatureFn,
    outcome_fn_control: _FeatureFn,
    clip_propensity_lower: float = 0.01,
    clip_propensity_upper: float = 0.99,
) -> DoublyRobustReport:
    """Cross-fit estimators are assumed to live in the supplied
    callables (canonical DML pattern). The function combines them
    into IPW + g-formula + AIPW estimates."""
    n = len(features)
    if n == 0:
        raise ValueError("features cannot be empty")
    if not (n == len(treatments) == len(outcomes)):
        raise ValueError("inputs must align")
    if not all(t in (0, 1) for t in treatments):
        raise ValueError("treatments must be 0 or 1")

    e_clipped: list[float] = []
    psi_aipw: list[float] = []
    ipw_terms: list[float] = []
    g_terms: list[float] = []

    for x, t, y in zip(features, treatments, outcomes):
        e = _clip(
            float(propensity_fn(x)),
            clip_propensity_lower, clip_propensity_upper,
        )
        e_clipped.append(e)
        mu1 = float(outcome_fn_treated(x))
        mu0 = float(outcome_fn_control(x))
        # IPW
        ipw_t = t * y / e
        ipw_c = (1 - t) * y / (1 - e)
        ipw_terms.append(ipw_t - ipw_c)
        # G-formula
        g_terms.append(mu1 - mu0)
        # AIPW
        aug_t = t * (y - mu1) / e + mu1
        aug_c = (1 - t) * (y - mu0) / (1 - e) + mu0
        psi_aipw.append(aug_t - aug_c)

    ate_ipw = sum(ipw_terms) / n
    ate_g = sum(g_terms) / n
    ate_aipw = sum(psi_aipw) / n
    # AIPW influence-function variance estimate (efficient bound)
    var_aipw = (
        sum((psi - ate_aipw) ** 2 for psi in psi_aipw) / (n - 1)
        if n > 1 else 0.0
    )
    se = math.sqrt(var_aipw / n) if var_aipw > 0 else 0.0

    return DoublyRobustReport(
        n=n,
        ate_ipw=round(ate_ipw, 6),
        ate_g_formula=round(ate_g, 6),
        ate_aipw=round(ate_aipw, 6),
        se_aipw=round(se, 6),
        n_treated=sum(1 for t in treatments if t == 1),
        n_control=sum(1 for t in treatments if t == 0),
        propensity_min=round(min(e_clipped), 6),
        propensity_max=round(max(e_clipped), 6),
        rationale=(
            f"Doubly-robust ATE estimation on n={n}; "
            f"IPW={ate_ipw:.4f}, "
            f"g-formula={ate_g:.4f}, "
            f"AIPW={ate_aipw:.4f} (SE={se:.4f}). Robust if either "
            f"propensity or outcome model is correct."
        ),
    )
