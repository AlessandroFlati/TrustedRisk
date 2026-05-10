"""Phase 17.M - Causal + counterfactual depth tests."""

from __future__ import annotations

import math

import pytest

from a2a_agent.causal_depth import (
    FrontDoorReport, RosenbaumSensitivityReport,
    WachterCounterfactualReport, WaldIVReport,
    _logistic_score, find_min_distance_counterfactual,
    front_door_adjustment, rosenbaum_gamma_bound,
    wald_iv_estimate,
)


# ─────────────────────────────────────────────────────────────────────
# Wachter min-distance counterfactual
# ─────────────────────────────────────────────────────────────────────

def test_wachter_yields_counterfactual_with_higher_score():
    """LACE-like: low-risk patient has score < 0.5, find the
    minimum modification that pushes the score above 0.5."""
    coeffs = {"L": 0.2, "A": 0.1, "C": 0.1, "E": 0.1}
    intercept = -3.0
    features = {"L": 1.0, "A": 0.0, "C": 0.0, "E": 0.0}
    res = find_min_distance_counterfactual(
        features=features, coefficients=coeffs,
        intercept=intercept,
        target_score_at_least=0.5,
    )
    assert res.counterfactual_score > res.original_score
    assert res.delta_l1 > 0


def test_wachter_converges_when_target_already_met():
    coeffs = {"x": 1.0}
    res = find_min_distance_counterfactual(
        features={"x": 5.0}, coefficients=coeffs,
        intercept=0.0, target_score_at_least=0.5,
    )
    # already at 5.0 -> sigmoid > 0.5 -> immediate convergence
    assert res.converged
    assert res.delta_l1 == 0.0


def test_wachter_respects_feature_min_max():
    coeffs = {"x": 1.0}
    res = find_min_distance_counterfactual(
        features={"x": 0.0}, coefficients=coeffs,
        intercept=-2.0, target_score_at_least=0.6,
        feature_max={"x": 1.0},
    )
    # x clipped to [0, 1] -> sigmoid(-2 + 1) = 0.27 < 0.6 ->
    # cannot converge
    assert res.counterfactual_features["x"] <= 1.0 + 1e-6


def test_wachter_rejects_empty_coefficients():
    with pytest.raises(ValueError):
        find_min_distance_counterfactual(
            features={"x": 1.0}, coefficients={},
            intercept=0.0,
        )


def test_wachter_target_must_be_in_open_unit():
    with pytest.raises(ValueError):
        find_min_distance_counterfactual(
            features={"x": 0.0}, coefficients={"x": 1.0},
            intercept=0.0, target_score_at_least=1.0,
        )


# ─────────────────────────────────────────────────────────────────────
# Rosenbaum sensitivity
# ─────────────────────────────────────────────────────────────────────

def test_rosenbaum_returns_gamma_at_least_one():
    rep = rosenbaum_gamma_bound(
        n_discordant_pairs=20, s_treatment_better=18,
    )
    assert rep.gamma_upper_bound >= 1.0


def test_rosenbaum_observed_p_low_when_overwhelming_majority():
    rep = rosenbaum_gamma_bound(
        n_discordant_pairs=20, s_treatment_better=18,
    )
    # 18 of 20 successes under p=0.5 is highly significant
    assert rep.observed_p_value < 0.001


def test_rosenbaum_higher_majority_gives_higher_gamma():
    a = rosenbaum_gamma_bound(
        n_discordant_pairs=30, s_treatment_better=18,
    )
    b = rosenbaum_gamma_bound(
        n_discordant_pairs=30, s_treatment_better=27,
    )
    assert b.gamma_upper_bound >= a.gamma_upper_bound


def test_rosenbaum_rejects_zero_pairs():
    with pytest.raises(ValueError):
        rosenbaum_gamma_bound(
            n_discordant_pairs=0, s_treatment_better=0,
        )


def test_rosenbaum_rejects_s_out_of_range():
    with pytest.raises(ValueError):
        rosenbaum_gamma_bound(
            n_discordant_pairs=10, s_treatment_better=11,
        )


# ─────────────────────────────────────────────────────────────────────
# Wald IV
# ─────────────────────────────────────────────────────────────────────

def test_wald_iv_recovers_known_effect():
    """Z is a valid IV: directly causes D, only affects Y through D.
    True ATE = 0.5."""
    import random
    rng = random.Random(7)
    Y, D, Z = [], [], []
    for _ in range(2000):
        z = rng.choice([0, 1])
        d = z   # perfect compliance
        # Y = 0.5 * D + epsilon, no other Z dependency
        y = 0.5 * d + rng.gauss(0, 0.5)
        Y.append(y); D.append(d); Z.append(z)
    rep = wald_iv_estimate(
        outcome=Y, treatment=D, instrument=Z)
    assert abs(rep.iv_ate_estimate - 0.5) < 0.1


def test_wald_iv_rejects_zero_relevance():
    Y = [1.0, 0.0, 1.0]
    D = [1.0, 0.0, 1.0]
    Z = [0.0, 0.0, 0.0]   # no variation
    with pytest.raises(ValueError):
        wald_iv_estimate(
            outcome=Y, treatment=D, instrument=Z)


def test_wald_iv_rejects_misaligned():
    with pytest.raises(ValueError):
        wald_iv_estimate(
            outcome=[1.0], treatment=[0.0, 1.0],
            instrument=[0.0],
        )


def test_wald_iv_rejects_short_input():
    with pytest.raises(ValueError):
        wald_iv_estimate(
            outcome=[1.0], treatment=[0.0],
            instrument=[1.0],
        )


# ─────────────────────────────────────────────────────────────────────
# Front-door adjustment
# ─────────────────────────────────────────────────────────────────────

def test_front_door_returns_probability_per_treatment():
    # Pearl's smoking-tar-cancer setup, simplified.
    # X: smoke {yes, no}, M: tar {yes, no}, Y: cancer {yes, no}
    p_m_given_x = {
        "smoke": {"tar": 0.95, "no_tar": 0.05},
        "no_smoke": {"tar": 0.05, "no_tar": 0.95},
    }
    p_y_given_xm = {
        "smoke": {"tar": {"cancer": 0.95, "no_cancer": 0.05},
                   "no_tar": {"cancer": 0.10, "no_cancer": 0.90}},
        "no_smoke": {"tar": {"cancer": 0.85, "no_cancer": 0.15},
                      "no_tar": {"cancer": 0.10, "no_cancer": 0.90}},
    }
    p_x = {"smoke": 0.4, "no_smoke": 0.6}
    rep = front_door_adjustment(
        p_m_given_x=p_m_given_x,
        p_y_given_xm=p_y_given_xm,
        p_x=p_x,
        treatment_values=["smoke", "no_smoke"],
        outcome_value="cancer",
    )
    # P(cancer | do(smoke)) > P(cancer | do(no_smoke))
    assert rep.p_y_do_x["smoke"] > rep.p_y_do_x["no_smoke"]
    assert 0.0 <= rep.p_y_do_x["smoke"] <= 1.0


def test_front_door_rejects_missing_treatment():
    with pytest.raises(ValueError):
        front_door_adjustment(
            p_m_given_x={"x1": {"m1": 1.0}},
            p_y_given_xm={"x1": {"m1": {"y1": 0.5}}},
            p_x={"x1": 1.0},
            treatment_values=["nonexistent"],
            outcome_value="y1",
        )


def test_front_door_rejects_empty_treatment_values():
    with pytest.raises(ValueError):
        front_door_adjustment(
            p_m_given_x={"x": {"m": 1.0}},
            p_y_given_xm={"x": {"m": {"y": 1.0}}},
            p_x={"x": 1.0},
            treatment_values=[], outcome_value="y",
        )


# ─────────────────────────────────────────────────────────────────────
# Logistic helper
# ─────────────────────────────────────────────────────────────────────

def test_logistic_score_in_unit_interval():
    score = _logistic_score(
        {"x": 1.0, "y": -2.0},
        {"x": 0.5, "y": 1.0}, intercept=0.0,
    )
    assert 0.0 <= score <= 1.0


def test_logistic_score_zero_logit_is_half():
    score = _logistic_score({}, {"x": 1.0}, intercept=0.0)
    assert abs(score - 0.5) < 1e-9
