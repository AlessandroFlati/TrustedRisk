"""Phase 17.AQ - Fine-Gray competing risks tests."""

from __future__ import annotations

import random

import pytest

from a2a_agent.fine_gray import FineGrayReport, fit_fine_gray


def _synthetic_competing_risks(
    n: int = 300, seed: int = 7,
) -> tuple[list[float], list[int], list[list[float]]]:
    rng = random.Random(seed)
    durations: list[float] = []
    events: list[int] = []
    covariates: list[list[float]] = []
    for _ in range(n):
        x = rng.gauss(0, 1)
        # event of interest hazard ~ exp(0.6 x)
        # competing hazard constant
        t_event = rng.expovariate(0.05 * math.exp(0.6 * x))
        t_comp = rng.expovariate(0.04)
        t_cens = 30.0
        if t_event < t_comp and t_event < t_cens:
            t, e = t_event, 1
        elif t_comp < t_cens:
            t, e = t_comp, 2
        else:
            t, e = t_cens, 0
        durations.append(t)
        events.append(e)
        covariates.append([x])
    return durations, events, covariates


import math


# ─────────────────────────────────────────────────────────────────────
# Fitting
# ─────────────────────────────────────────────────────────────────────

def test_fine_gray_returns_report_with_correct_counts():
    d, e, x = _synthetic_competing_risks(n=200, seed=1)
    rep = fit_fine_gray(
        durations=d, events=e, covariates=x,
        feature_names=["x1"],
    )
    assert rep.n == 200
    assert (
        rep.n_event_of_interest + rep.n_competing
        + rep.n_censored == 200
    )


def test_fine_gray_recovers_positive_coefficient():
    d, e, x = _synthetic_competing_risks(n=400, seed=2)
    rep = fit_fine_gray(
        durations=d, events=e, covariates=x,
        feature_names=["x1"],
    )
    # True coefficient on x is +0.6 -> should fit positive
    assert rep.coefficients[0] > 0.0


def test_fine_gray_subdistribution_hazard_ratio_equals_exp_beta():
    d, e, x = _synthetic_competing_risks(n=200, seed=3)
    rep = fit_fine_gray(
        durations=d, events=e, covariates=x,
        feature_names=["x1"],
    )
    assert (
        abs(rep.subdistribution_hazard_ratios["x1"]
            - math.exp(rep.coefficients[0])) < 1e-3
    )


def test_fine_gray_cif_in_unit_interval():
    d, e, x = _synthetic_competing_risks(n=200, seed=4)
    rep = fit_fine_gray(
        durations=d, events=e, covariates=x,
        feature_names=["x1"],
    )
    assert 0.0 <= rep.cumulative_incidence_at_horizon <= 1.0


def test_fine_gray_rejects_misaligned():
    with pytest.raises(ValueError):
        fit_fine_gray(
            durations=[1.0, 2.0],
            events=[1],
            covariates=[[0.0], [1.0]],
        )


def test_fine_gray_rejects_empty():
    with pytest.raises(ValueError):
        fit_fine_gray(
            durations=[], events=[], covariates=[],
        )


def test_fine_gray_rejects_zero_features():
    with pytest.raises(ValueError):
        fit_fine_gray(
            durations=[1.0], events=[1], covariates=[[]],
        )


def test_fine_gray_round_trip_through_pydantic():
    d, e, x = _synthetic_competing_risks(n=100, seed=5)
    rep = fit_fine_gray(
        durations=d, events=e, covariates=x,
        feature_names=["x1"],
    )
    payload = rep.model_dump(mode="json")
    rebuilt = FineGrayReport.model_validate(payload)
    assert rebuilt.n == rep.n


def test_fine_gray_with_competing_risks_yields_lower_cif_than_no_competing():
    """Adding competing events to the cohort should lower the CIF
    of the event of interest at the same horizon."""
    rng = random.Random(13)
    d_no, e_no, x_no = [], [], []
    d_w, e_w, x_w = [], [], []
    for _ in range(300):
        x = rng.gauss(0, 1)
        # 'no competing' cohort: only event of interest + censoring
        t_evt = rng.expovariate(0.05 * math.exp(0.6 * x))
        if t_evt < 30:
            d_no.append(t_evt); e_no.append(1)
        else:
            d_no.append(30.0); e_no.append(0)
        x_no.append([x])
        # 'with competing' cohort: same draws + competing event at
        # rate 0.04
        t_comp = rng.expovariate(0.04)
        if t_evt < t_comp and t_evt < 30:
            d_w.append(t_evt); e_w.append(1)
        elif t_comp < 30:
            d_w.append(t_comp); e_w.append(2)
        else:
            d_w.append(30.0); e_w.append(0)
        x_w.append([x])
    rep_no = fit_fine_gray(
        durations=d_no, events=e_no, covariates=x_no,
        feature_names=["x1"], horizon=30.0,
    )
    rep_w = fit_fine_gray(
        durations=d_w, events=e_w, covariates=x_w,
        feature_names=["x1"], horizon=30.0,
    )
    assert (
        rep_w.cumulative_incidence_at_horizon
        <= rep_no.cumulative_incidence_at_horizon + 0.05
    )
