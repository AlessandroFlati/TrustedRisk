"""SIM-1 -- Monte Carlo hospital outcomes simulator.

Given a hospital case-mix and an intervention's evidence-based effect size,
simulates a 12-month rollout and returns:

  - expected events avoided (mean + 95% CI from bootstrap)
  - $ saved (mean + 95% CI)
  - QALY gained (mean + 95% CI when QALY input present)
  - per-segment breakdown
  - 1-way sensitivity sweep over RRR and avoided_event_cost_usd

Math: per-iteration, for each segment we draw two Binomial(n, p) variates --
one with the baseline event probability and one with the post-intervention
probability -- and take the difference. Bootstrap percentiles produce the CI.
This is the Probabilistic Sensitivity Analysis (PSA) pattern from
Briggs et al. PharmacoEconomics 2008;26:781.

Module is dependency-light: numpy only.
"""

from __future__ import annotations

import numpy as np

from shared.schemas import (
    CaseMixSegment,
    HospitalYearSimulation,
    SensitivityRow,
    SimulationSegmentResult,
)


# ─────────────────────── Validation ───────────────────────

def _validate_inputs(
    case_mix: list[CaseMixSegment],
    intervention_rrr: float,
    intervention_cost_per_patient_usd: float,
    avoided_event_cost_usd: float,
    qaly_gained_per_avoided_event: float | None,
    n_iterations: int,
) -> None:
    if not case_mix:
        raise ValueError("case_mix must contain at least one segment.")
    if not (0.0 <= intervention_rrr <= 1.0):
        raise ValueError(
            f"intervention_rrr must be in [0, 1]; got {intervention_rrr}")
    if intervention_cost_per_patient_usd < 0:
        raise ValueError("intervention_cost_per_patient_usd must be >= 0.")
    if avoided_event_cost_usd < 0:
        raise ValueError("avoided_event_cost_usd must be >= 0.")
    if qaly_gained_per_avoided_event is not None and \
            qaly_gained_per_avoided_event < 0:
        raise ValueError("qaly_gained_per_avoided_event must be >= 0.")
    if n_iterations < 1 or n_iterations > 100_000:
        raise ValueError(
            f"n_iterations must be in [1, 100000]; got {n_iterations}")


# ─────────────────────── Single-iteration sampler ───────────────────────

def _simulate_segment_iter(
    rng: np.random.Generator,
    n: int,
    baseline_p: float,
    rrr: float,
    n_iter: int,
) -> np.ndarray:
    """Return n_iter samples of (events_avoided) for this segment."""
    post_p = max(0.0, baseline_p * (1.0 - rrr))
    no_intervention = rng.binomial(n=n, p=baseline_p, size=n_iter)
    with_intervention = rng.binomial(n=n, p=post_p, size=n_iter)
    return no_intervention - with_intervention


# ─────────────────────── Sensitivity ───────────────────────

def _sensitivity_sweep(
    case_mix: list[CaseMixSegment],
    intervention_cost_per_patient_usd: float,
    avoided_event_cost_usd: float,
    base_rrr: float,
    rrr_grid: list[float],
    cost_grid: list[float],
    rng: np.random.Generator,
    n_iter: int,
) -> list[SensitivityRow]:
    rows: list[SensitivityRow] = []
    for rrr in rrr_grid:
        total_avoided = np.zeros(n_iter, dtype=np.float64)
        for seg in case_mix:
            total_avoided += _simulate_segment_iter(
                rng, seg.n_patients_per_year,
                seg.baseline_event_probability, rrr, n_iter)
        events_mean = float(total_avoided.mean())
        cost_mean = events_mean * float(avoided_event_cost_usd)
        rows.append(SensitivityRow(
            parameter="intervention_rrr",
            value=float(rrr),
            events_avoided_mean=events_mean,
            cost_avoided_mean_usd=cost_mean,
        ))

    for cost in cost_grid:
        total_avoided = np.zeros(n_iter, dtype=np.float64)
        for seg in case_mix:
            total_avoided += _simulate_segment_iter(
                rng, seg.n_patients_per_year,
                seg.baseline_event_probability, base_rrr, n_iter)
        events_mean = float(total_avoided.mean())
        cost_mean = events_mean * float(cost)
        rows.append(SensitivityRow(
            parameter="avoided_event_cost_usd",
            value=float(cost),
            events_avoided_mean=events_mean,
            cost_avoided_mean_usd=cost_mean,
        ))
    return rows


# ─────────────────────── Main simulator ───────────────────────

def simulate_hospital_year(
    case_mix: list[CaseMixSegment | dict],
    *,
    intervention_name: str = "Pharmacist-led discharge counseling",
    intervention_relative_risk_reduction: float = 0.25,
    intervention_cost_per_patient_usd: float = 75.0,
    avoided_event_cost_usd: float = 14_000.0,
    qaly_gained_per_avoided_event: float | None = 0.05,
    n_iterations: int = 2_000,
    seed: int = 42,
    rrr_grid: list[float] | None = None,
    cost_grid: list[float] | None = None,
) -> HospitalYearSimulation:
    """Monte Carlo simulation of one year of intervention rollout.

    Args:
        case_mix: list of CaseMixSegment (or dicts coerced).
        intervention_name: free-text label for the intervention.
        intervention_relative_risk_reduction: RRR (0..1).
        intervention_cost_per_patient_usd: per-patient intervention cost.
        avoided_event_cost_usd: cost per avoided event.
        qaly_gained_per_avoided_event: when set, the simulator returns
            QALY gained + cost-per-QALY.
        n_iterations: Monte Carlo iterations (default 2000 -- fast, low-noise).
        seed: numpy RNG seed for reproducibility.
        rrr_grid: optional sensitivity grid for RRR (defaults to ±10pp).
        cost_grid: optional sensitivity grid for avoided-event cost
            (defaults to ±50%).

    Returns:
        HospitalYearSimulation with point + 95% CI + sensitivity rows.
    """
    coerced: list[CaseMixSegment] = []
    for s in case_mix:
        if isinstance(s, dict):
            coerced.append(CaseMixSegment.model_validate(s))
        else:
            coerced.append(s)
    case_mix = coerced

    _validate_inputs(case_mix,
                       intervention_relative_risk_reduction,
                       intervention_cost_per_patient_usd,
                       avoided_event_cost_usd,
                       qaly_gained_per_avoided_event,
                       n_iterations)

    rng = np.random.default_rng(seed)

    cohort_size = sum(s.n_patients_per_year for s in case_mix)

    # Per-segment Monte Carlo
    segments_results: list[SimulationSegmentResult] = []
    total_events_per_iter = np.zeros(n_iterations, dtype=np.float64)
    for seg in case_mix:
        avoided = _simulate_segment_iter(
            rng, seg.n_patients_per_year,
            seg.baseline_event_probability,
            intervention_relative_risk_reduction,
            n_iterations,
        )
        total_events_per_iter += avoided.astype(np.float64)
        events_mean = float(avoided.mean())
        events_ci = (float(np.percentile(avoided, 2.5)),
                       float(np.percentile(avoided, 97.5)))
        cost_avoided = avoided.astype(np.float64) * float(avoided_event_cost_usd)
        cost_ci = (float(np.percentile(cost_avoided, 2.5)),
                     float(np.percentile(cost_avoided, 97.5)))
        # Per-segment expected (no MC needed -- closed form)
        exp_no = seg.n_patients_per_year * seg.baseline_event_probability
        exp_with = seg.n_patients_per_year * seg.baseline_event_probability * (
            1.0 - intervention_relative_risk_reduction)

        qaly_mean = qaly_ci = None
        if qaly_gained_per_avoided_event is not None:
            qaly_arr = avoided.astype(np.float64) * \
                float(qaly_gained_per_avoided_event)
            qaly_mean = float(qaly_arr.mean())
            qaly_ci = (float(np.percentile(qaly_arr, 2.5)),
                        float(np.percentile(qaly_arr, 97.5)))

        segments_results.append(SimulationSegmentResult(
            name=seg.name,
            n_patients=seg.n_patients_per_year,
            baseline_event_probability=seg.baseline_event_probability,
            expected_events_no_intervention=exp_no,
            expected_events_with_intervention=exp_with,
            events_avoided_mean=events_mean,
            events_avoided_ci95=events_ci,
            cost_avoided_mean_usd=float(cost_avoided.mean()),
            cost_avoided_ci95_usd=cost_ci,
            qaly_gained_mean=qaly_mean,
            qaly_gained_ci95=qaly_ci,
        ))

    # Aggregate totals
    total_events_mean = float(total_events_per_iter.mean())
    total_events_ci = (
        float(np.percentile(total_events_per_iter, 2.5)),
        float(np.percentile(total_events_per_iter, 97.5)),
    )
    total_intervention_cost = cohort_size * float(
        intervention_cost_per_patient_usd)
    total_cost_avoided_arr = total_events_per_iter * float(
        avoided_event_cost_usd)
    total_cost_avoided_mean = float(total_cost_avoided_arr.mean())
    total_cost_avoided_ci = (
        float(np.percentile(total_cost_avoided_arr, 2.5)),
        float(np.percentile(total_cost_avoided_arr, 97.5)),
    )
    net_cost_arr = total_intervention_cost - total_cost_avoided_arr
    net_cost_mean = float(net_cost_arr.mean())
    net_cost_ci = (
        float(np.percentile(net_cost_arr, 2.5)),
        float(np.percentile(net_cost_arr, 97.5)),
    )

    total_qaly_mean = total_qaly_ci = cost_per_qaly_mean = None
    if qaly_gained_per_avoided_event is not None:
        qaly_arr = total_events_per_iter * float(
            qaly_gained_per_avoided_event)
        total_qaly_mean = float(qaly_arr.mean())
        total_qaly_ci = (
            float(np.percentile(qaly_arr, 2.5)),
            float(np.percentile(qaly_arr, 97.5)),
        )
        if total_qaly_mean > 0:
            cost_per_qaly_mean = net_cost_mean / total_qaly_mean

    if rrr_grid is None:
        rrr_grid = sorted({max(0.0, intervention_relative_risk_reduction - 0.10),
                            intervention_relative_risk_reduction,
                            min(1.0, intervention_relative_risk_reduction + 0.10)})
    if cost_grid is None:
        cost_grid = [avoided_event_cost_usd * 0.5,
                       avoided_event_cost_usd,
                       avoided_event_cost_usd * 1.5]

    sensitivity = _sensitivity_sweep(
        case_mix=case_mix,
        intervention_cost_per_patient_usd=intervention_cost_per_patient_usd,
        avoided_event_cost_usd=avoided_event_cost_usd,
        base_rrr=intervention_relative_risk_reduction,
        rrr_grid=rrr_grid,
        cost_grid=cost_grid,
        rng=np.random.default_rng(seed + 1),
        n_iter=max(500, n_iterations // 4),
    )

    return HospitalYearSimulation(
        intervention_name=intervention_name,
        n_iterations=n_iterations,
        seed=seed,
        cohort_size_per_year=cohort_size,
        intervention_relative_risk_reduction=intervention_relative_risk_reduction,
        intervention_cost_per_patient_usd=intervention_cost_per_patient_usd,
        avoided_event_cost_usd=avoided_event_cost_usd,
        qaly_gained_per_avoided_event=qaly_gained_per_avoided_event,
        segments=segments_results,
        total_events_avoided_mean=total_events_mean,
        total_events_avoided_ci95=total_events_ci,
        total_intervention_cost_usd=total_intervention_cost,
        total_avoided_event_cost_mean_usd=total_cost_avoided_mean,
        total_avoided_event_cost_ci95_usd=total_cost_avoided_ci,
        net_cost_mean_usd=net_cost_mean,
        net_cost_ci95_usd=net_cost_ci,
        total_qaly_gained_mean=total_qaly_mean,
        total_qaly_gained_ci95=total_qaly_ci,
        cost_per_qaly_mean_usd=cost_per_qaly_mean,
        sensitivity=sensitivity,
    )
