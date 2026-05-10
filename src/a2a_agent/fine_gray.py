"""Phase 17.AQ - Survival analysis with competing risks (Fine-Gray).

Fine & Gray 1999 "A Proportional Hazards Model for the
Subdistribution of a Competing Risk". Estimates the *subdistribution
hazard ratio* (SHR) of an event of interest under the presence of
competing events that prevent it - in our case:

  - event of interest: ``readmission`` (event_type = 1)
  - competing event:   ``death_before_readmit`` (event_type = 2)
  - censored:          ``no_event_in_horizon`` (event_type = 0)

Different from Cox's cause-specific hazard:
  - Cox keeps subjects in the risk set only until they have *any*
    event.
  - Fine-Gray keeps subjects with the competing event in the risk
    set forever (with a weight that decays as the censoring
    distribution decays). This matches the cumulative incidence
    function (CIF) directly.

Pure-Python Newton-Raphson on the Fine-Gray weighted partial
likelihood. Returns SHR + cumulative incidence function (CIF).
"""

from __future__ import annotations

import math
from typing import Iterable, Literal

from pydantic import BaseModel, Field


_EventType = Literal[0, 1, 2]


class FineGrayReport(BaseModel):
    n: int
    n_event_of_interest: int
    n_competing: int
    n_censored: int
    coefficients: list[float]
    feature_names: list[str]
    subdistribution_hazard_ratios: dict[str, float]
    log_likelihood: float
    cumulative_incidence_at_horizon: float
    horizon: float
    rationale: str


# ─────────────────────────────────────────────────────────────────────
# Newton-Raphson on the Fine-Gray weighted partial likelihood
# ─────────────────────────────────────────────────────────────────────


def _ipcw_weights(
    durations: list[float], events: list[_EventType],
) -> list[list[float]]:
    """IPCW (inverse-probability-of-censoring) weights per
    observation x time. For our minimal implementation we assume
    administrative censoring at ``max(durations)`` so the weight is
    1 throughout the at-risk window for non-competing subjects, and
    decays after the competing event for competing-event subjects.

    Concretely we build a dictionary of the unique event-of-interest
    times, and for each subject we record their weight at each such
    time:

      - never had any event before t: weight 1
      - had event of interest at exact t: weight 1 (contributes to
        the numerator)
      - had competing event at time s < t: weight decays linearly
        from 1 (at s) to 0 (at max_time); standard IPCW under
        admin censoring.
      - had any event at time s < t (event of interest): weight 0
    """
    n = len(durations)
    max_t = max(durations) if durations else 0.0
    event_times = sorted({
        d for d, e in zip(durations, events) if e == 1
    })
    if not event_times:
        return [[1.0] * n]
    weights: list[list[float]] = [[0.0] * n for _ in event_times]
    for j, t in enumerate(event_times):
        for i, (d_i, e_i) in enumerate(zip(durations, events)):
            if e_i == 1:
                # event of interest
                if d_i == t:
                    weights[j][i] = 1.0
                elif d_i > t:
                    weights[j][i] = 1.0
                else:
                    # event already happened before t -> excluded
                    weights[j][i] = 0.0
            elif e_i == 2:
                # competing event
                if d_i >= t:
                    weights[j][i] = 1.0
                else:
                    decay = (
                        max(0.0, 1.0 - (t - d_i)
                                / max(1e-9, max_t - d_i))
                    )
                    weights[j][i] = decay
            else:
                # censored
                if d_i >= t:
                    weights[j][i] = 1.0
                else:
                    weights[j][i] = 0.0
    return weights


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _solve_linear_system(
    H: list[list[float]], g: list[float],
) -> list[float]:
    """Solve H delta = g via Gaussian elimination (small p)."""
    n = len(g)
    A = [row[:] + [g[i]] for i, row in enumerate(H)]
    for col in range(n):
        pivot = col
        for r in range(col + 1, n):
            if abs(A[r][col]) > abs(A[pivot][col]):
                pivot = r
        if abs(A[pivot][col]) < 1e-12:
            return [0.0] * n
        A[col], A[pivot] = A[pivot], A[col]
        pv = A[col][col]
        for j in range(col, n + 1):
            A[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            f = A[r][col]
            for j in range(col, n + 1):
                A[r][j] -= f * A[col][j]
    return [A[i][n] for i in range(n)]


def fit_fine_gray(
    *,
    durations: list[float],
    events: list[_EventType],
    covariates: list[list[float]],
    feature_names: list[str] | None = None,
    horizon: float | None = None,
    max_iter: int = 30,
    tolerance: float = 1e-6,
) -> FineGrayReport:
    """Fit a Fine-Gray subdistribution-hazards model via IPCW-
    weighted Newton-Raphson on the partial likelihood."""
    n = len(durations)
    if n == 0:
        raise ValueError("durations cannot be empty")
    if not (n == len(events) == len(covariates)):
        raise ValueError(
            "durations, events, covariates must align")
    p = len(covariates[0]) if covariates else 0
    if p == 0:
        raise ValueError("covariates must have >= 1 feature column")
    feature_names = (
        feature_names if feature_names is not None
        else [f"x{i}" for i in range(p)]
    )
    if len(feature_names) != p:
        raise ValueError(
            "feature_names length must match covariate width")

    weights = _ipcw_weights(durations, events)
    event_times = sorted({
        d for d, e in zip(durations, events) if e == 1
    })

    beta = [0.0] * p
    last_ll = float("-inf")

    for _ in range(max_iter):
        score = [0.0] * p
        info = [[0.0] * p for _ in range(p)]
        ll = 0.0
        for j, t in enumerate(event_times):
            w = weights[j]
            # event subject indices at this time
            s_event = [
                i for i, (d_i, e_i) in enumerate(zip(durations, events))
                if e_i == 1 and d_i == t
            ]
            if not s_event:
                continue
            # exp(beta' x_i) per subject in the risk set
            exp_terms = [
                math.exp(_dot(beta, covariates[i]))
                for i in range(n)
            ]
            denom = sum(
                w[i] * exp_terms[i] for i in range(n)
            )
            if denom <= 0:
                continue
            # Numerator: sum over event subjects of x_i
            sum_x_event = [0.0] * p
            for i in s_event:
                for k in range(p):
                    sum_x_event[k] += covariates[i][k]
                ll += _dot(beta, covariates[i]) - math.log(denom)
            mean_x = [0.0] * p
            for i in range(n):
                wi = w[i] * exp_terms[i] / denom
                for k in range(p):
                    mean_x[k] += wi * covariates[i][k]
            d_event = len(s_event)
            for k in range(p):
                score[k] += sum_x_event[k] - d_event * mean_x[k]
            # Information matrix
            for k in range(p):
                for l in range(p):
                    inner = 0.0
                    for i in range(n):
                        wi = w[i] * exp_terms[i] / denom
                        inner += wi * (
                            (covariates[i][k] - mean_x[k])
                            * (covariates[i][l] - mean_x[l])
                        )
                    info[k][l] += d_event * inner
        delta = _solve_linear_system(info, score)
        beta = [beta[k] + delta[k] for k in range(p)]
        if abs(ll - last_ll) < tolerance:
            break
        last_ll = ll

    # Cumulative incidence at horizon
    horizon = horizon if horizon is not None else max(durations)
    cif = 0.0
    s = 1.0   # tracks cumulative survival in subdistribution
    for j, t in enumerate(event_times):
        if t > horizon:
            break
        d_event = sum(
            1 for d_i, e_i in zip(durations, events)
            if e_i == 1 and d_i == t
        )
        w = weights[j]
        exp_terms = [
            math.exp(_dot(beta, covariates[i])) for i in range(n)
        ]
        denom = sum(
            w[i] * exp_terms[i] for i in range(n)
        )
        if denom <= 0:
            continue
        # Hazard increment for the average subject (xb=0)
        h = d_event / denom
        cif += s * h
        s *= (1.0 - h)

    n_event = sum(1 for e in events if e == 1)
    n_comp = sum(1 for e in events if e == 2)
    n_cens = sum(1 for e in events if e == 0)

    shr_dict = {
        feature_names[i]: round(math.exp(beta[i]), 6)
        for i in range(p)
    }

    return FineGrayReport(
        n=n,
        n_event_of_interest=n_event,
        n_competing=n_comp,
        n_censored=n_cens,
        coefficients=[round(b, 6) for b in beta],
        feature_names=list(feature_names),
        subdistribution_hazard_ratios=shr_dict,
        log_likelihood=round(last_ll, 6),
        cumulative_incidence_at_horizon=round(cif, 6),
        horizon=float(horizon),
        rationale=(
            f"Fine-Gray fit on n={n} ({n_event} event, "
            f"{n_comp} competing, {n_cens} censored); "
            f"CIF at horizon {horizon:.2f} = {cif*100:.2f}%."
        ),
    )
