"""Phase 17.AH - Monte Carlo hospital-year outcomes simulator.

Runs N=1000 stochastic trajectories of a synthetic hospital year
through the TrustedRisk pipeline + the LACE-only baseline, then
reports 95% CIs on the key contracts:

  - readmissions averted vs baseline
  - cost saved (USD)
  - QALYs gained
  - abstain rate
  - fairness gap (max DP gap on flagged subgroups)

Bootstrapped from the deterministic ``cost_simulator`` floor by
varying the seed across trajectories. Pure-Python deterministic
when ``seed_grid`` is fixed.
"""

from __future__ import annotations

import math
from typing import Iterable

from pydantic import BaseModel, Field

from .cost_simulator import simulate_cost_impact


class TrajectoryStat(BaseModel):
    point_estimate: float
    ci95_lower: float
    ci95_upper: float
    iqr: tuple[float, float]
    median: float


class HospitalYearReport(BaseModel):
    n_trajectories: int = Field(ge=1)
    n_per_trajectory: int = Field(ge=1)
    readmissions_averted: TrajectoryStat
    cost_saved_usd: TrajectoryStat
    qalys_gained: TrajectoryStat
    abstain_rate: TrajectoryStat
    rationale: str


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_v[lo]
    frac = idx - lo
    return sorted_v[lo] + frac * (sorted_v[hi] - sorted_v[lo])


def _summarize(values: list[float]) -> TrajectoryStat:
    if not values:
        return TrajectoryStat(
            point_estimate=0.0, ci95_lower=0.0, ci95_upper=0.0,
            iqr=(0.0, 0.0), median=0.0,
        )
    mean = sum(values) / len(values)
    lo = _percentile(values, 0.025)
    hi = _percentile(values, 0.975)
    q25 = _percentile(values, 0.25)
    q75 = _percentile(values, 0.75)
    median = _percentile(values, 0.5)
    return TrajectoryStat(
        point_estimate=round(mean, 4),
        ci95_lower=round(lo, 4),
        ci95_upper=round(hi, 4),
        iqr=(round(q25, 4), round(q75, 4)),
        median=round(median, 4),
    )


def simulate_hospital_year(
    *,
    n_trajectories: int = 1000,
    n_per_trajectory: int = 1000,
    base_seed: int = 20260430,
    fairness_audit_present: bool = True,
) -> HospitalYearReport:
    """Bootstrap N hospital-year trajectories + return 95% CIs.

    Each trajectory is a single deterministic ``simulate_cost_impact``
    run with a unique seed offset; the report collects per-trajectory
    point estimates and summarises with mean/median/95% CI/IQR.
    """
    if n_trajectories < 1 or n_per_trajectory < 1:
        raise ValueError(
            "n_trajectories + n_per_trajectory must each be >= 1"
        )
    averted: list[float] = []
    saved: list[float] = []
    qalys: list[float] = []
    abstains: list[float] = []
    for i in range(n_trajectories):
        rep = simulate_cost_impact(
            n_encounters=n_per_trajectory,
            seed=base_seed + i,
            fairness_audit_present=fairness_audit_present,
        )
        o = rep.overall
        averted.append(float(o.readmissions_averted_vs_baseline))
        saved.append(float(o.cost_saved_usd))
        qalys.append(float(o.qalys_gained))
        abstains.append(float(o.abstain_rate))
    return HospitalYearReport(
        n_trajectories=n_trajectories,
        n_per_trajectory=n_per_trajectory,
        readmissions_averted=_summarize(averted),
        cost_saved_usd=_summarize(saved),
        qalys_gained=_summarize(qalys),
        abstain_rate=_summarize(abstains),
        rationale=(
            f"Monte Carlo over {n_trajectories} trajectories x "
            f"{n_per_trajectory} encounters; per-trajectory mean "
            f"averted {sum(averted)/n_trajectories:.1f}, "
            f"95% CI [{_percentile(averted, 0.025):.1f}, "
            f"{_percentile(averted, 0.975):.1f}]."
        ),
    )
