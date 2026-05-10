"""Calibration drift monitor (SCI-3).

Reads recent DecisionCards from the MemoryStore and computes a
sliding-window calibration metric that approximates the model's true ECE
on the receiving cohort.

Constraint: at runtime we typically don't have ground-truth outcomes
(actual readmission Yes/No) for the cards we just emitted -- outcomes
arrive 30-60 days later via post-discharge follow-up data. Therefore the
DRIFT signal here is **distributional**, not outcome-based:

  - **Mean predicted probability drift**: how the predicted-mean of recent
    cards compares to a baseline (the calibration cohort's mean from
    coefficients.json `selection_rationale`).
  - **LACE distribution drift**: KS-style distance between recent
    LACE histograms and the training LACE histogram.
  - **Abstain rate drift**: fraction of recent cards that fired an abstain
    trigger; used as a leading indicator of population shift.
  - **CI width drift**: mean predicted CI width over recent cards (wider
    CIs -> either calibration genuinely uncertain or the lookup table
    being applied to feature combinations under-represented at training).

When ground-truth outcomes ARE available (e.g. via a feedback channel),
the monitor can ALSO compute the post-hoc ECE on the windowed sample --
the API exposes this when `outcomes_observed` is provided.

References:
  Ovadia Y et al. Can You Trust Your Model's Uncertainty? Evaluating
    Predictive Uncertainty Under Dataset Shift. NeurIPS 2019.
  Quiñonero-Candela J et al. Dataset Shift in Machine Learning. MIT 2009.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from .memory import MemoryStore


# ─────────────────────── Baseline reference ───────────────────────
# Pulled from coefficients.json's selection_rationale and validation
# evidence. Production deployments would re-fit these from training stats.

_TRAINING_BASELINE: dict[str, Any] = {
    "mean_predicted_probability": 0.110,    # Synthea cohort mean
    "lace_histogram": {                       # Probability mass per LACE bucket
        "0-2": 0.05, "3-5": 0.18, "6-9": 0.40,
        "10-12": 0.25, "13-19": 0.12,
    },
    "ci_width_p50": 0.06,
    "abstain_rate": 0.08,                    # ~8% expected abstain at training
}


# ─────────────────────── Drift thresholds ───────────────────────

_THRESHOLDS = {
    "mean_prob_drift_warn": 0.05,      # ±5pp mean shift = warn
    "mean_prob_drift_alert": 0.10,     # ±10pp = alert
    "ks_distance_warn": 0.15,           # LACE histogram KS distance
    "ks_distance_alert": 0.25,
    "abstain_rate_warn": 0.20,          # >20% abstain rate
    "abstain_rate_alert": 0.35,
    "ci_width_drift_warn": 0.04,        # CI width inflation
    "ci_width_drift_alert": 0.08,
    "ece_warn": 0.08,                    # post-hoc when outcomes observed
    "ece_alert": 0.15,
}


@dataclass
class DriftSignal:
    """One drift signal with its tier."""
    name: str
    value: float
    baseline: float
    tier: str           # "ok" / "warn" / "alert"
    detail: str


@dataclass
class CalibrationDriftReport:
    """Aggregate drift report over a sliding window."""
    window_label: str
    window_n_cards: int
    window_start: datetime | None
    window_end: datetime | None
    overall_tier: str   # "ok" / "warn" / "alert"
    signals: list[DriftSignal] = field(default_factory=list)
    ece_post_hoc: float | None = None     # only populated when outcomes are given
    auroc_post_hoc: float | None = None
    n_outcomes_observed: int = 0
    rationale: str = ""


# ─────────────────────── KS distance ───────────────────────

def _lace_to_bucket(lace: int) -> str:
    if lace <= 2: return "0-2"
    if lace <= 5: return "3-5"
    if lace <= 9: return "6-9"
    if lace <= 12: return "10-12"
    return "13-19"


def _ks_distance(observed_pmf: dict[str, float],
                  expected_pmf: dict[str, float]) -> float:
    """Maximum cumulative-distribution gap between two histograms over a
    shared bucket order. (KS-style for ordinal histograms.)"""
    buckets = ["0-2", "3-5", "6-9", "10-12", "13-19"]
    cum_obs = 0.0
    cum_exp = 0.0
    max_gap = 0.0
    for b in buckets:
        cum_obs += observed_pmf.get(b, 0.0)
        cum_exp += expected_pmf.get(b, 0.0)
        max_gap = max(max_gap, abs(cum_obs - cum_exp))
    return max_gap


# ─────────────────────── ECE post-hoc ───────────────────────

def _ece(predictions: list[float], outcomes: list[int],
          n_bins: int = 10) -> float:
    if not predictions:
        return 0.0
    pred = np.asarray(predictions, dtype=float)
    obs = np.asarray(outcomes, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(pred)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (pred >= lo) & (pred <= hi)
        else:
            mask = (pred >= lo) & (pred < hi)
        n_in = int(mask.sum())
        if n_in == 0:
            continue
        gap = abs(pred[mask].mean() - obs[mask].mean())
        ece += (n_in / n_total) * gap
    return float(ece)


def _auroc(predictions: list[float], outcomes: list[int]) -> float:
    if not predictions:
        return float("nan")
    pred = np.asarray(predictions, dtype=float)
    obs = np.asarray(outcomes, dtype=int)
    if obs.sum() == 0 or obs.sum() == len(obs):
        return float("nan")
    order = np.argsort(-pred)
    o = obs[order]
    n_pos = int(o.sum())
    n_neg = len(o) - n_pos
    tps = np.cumsum(o); fps = np.cumsum(1 - o)
    tpr = tps / n_pos; fpr = fps / n_neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))


# ─────────────────────── Main monitor ───────────────────────

def compute_drift_report(
    store: MemoryStore | None = None,
    window_hours: int = 24 * 7,
    outcomes_observed: dict[str, int] | None = None,
    now: datetime | None = None,
) -> CalibrationDriftReport:
    """Build a drift report from the MemoryStore over the past `window_hours`.

    Args:
        store: MemoryStore instance. Uses default path if None.
        window_hours: rolling-window duration (default 7 days).
        outcomes_observed: optional dict mapping request_id -> observed
            30-day-readmission outcome (0/1). When provided, the report
            includes post-hoc ECE + AUROC.
        now: override "now" for deterministic tests.

    Returns:
        CalibrationDriftReport with per-signal tiers + overall tier.
    """
    if store is None:
        store = MemoryStore()
    if now is None:
        now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_hours)

    # Pull the cards within the window
    with store._connect() as conn:
        rows = conn.execute(
            """SELECT card_json, written_at_iso FROM decision_cards
               WHERE written_at_iso >= ?
               ORDER BY written_at_iso ASC""",
            (cutoff.isoformat(),),
        ).fetchall()

    from shared.schemas import DecisionCard
    cards: list[tuple[DecisionCard, datetime]] = []
    for r in rows:
        try:
            c = DecisionCard.model_validate_json(r["card_json"])
            ts = datetime.fromisoformat(r["written_at_iso"])
            cards.append((c, ts))
        except Exception:
            continue

    n = len(cards)
    if n == 0:
        return CalibrationDriftReport(
            window_label=f"past_{window_hours}h",
            window_n_cards=0, window_start=cutoff, window_end=now,
            overall_tier="ok", signals=[],
            rationale=f"No DecisionCards written in the past {window_hours}h.",
        )

    # Compute per-card features
    probs: list[float] = []
    laces: list[int] = []
    ci_widths: list[float] = []
    abstain_count = 0
    request_ids: list[str] = []
    for c, _ in cards:
        request_ids.append(c.audit.request_id)
        if c.reasoning is not None:
            r = c.reasoning.risk_estimate
            probs.append(r.probability_mean)
            laces.append(r.lace_raw_score)
            ci_widths.append(r.probability_ci_width)
        if c.abstain:
            abstain_count += 1

    signals: list[DriftSignal] = []

    if probs:
        mean_pred = float(np.mean(probs))
        delta = mean_pred - _TRAINING_BASELINE["mean_predicted_probability"]
        if abs(delta) >= _THRESHOLDS["mean_prob_drift_alert"]:
            tier = "alert"
        elif abs(delta) >= _THRESHOLDS["mean_prob_drift_warn"]:
            tier = "warn"
        else:
            tier = "ok"
        signals.append(DriftSignal(
            name="mean_predicted_probability",
            value=mean_pred,
            baseline=_TRAINING_BASELINE["mean_predicted_probability"],
            tier=tier,
            detail=f"Δ {delta:+.3f} vs training baseline.",
        ))

    if laces:
        observed_pmf: dict[str, float] = {}
        for l in laces:
            observed_pmf[_lace_to_bucket(l)] = (
                observed_pmf.get(_lace_to_bucket(l), 0.0) + 1.0
            )
        for k in observed_pmf:
            observed_pmf[k] /= len(laces)
        ks = _ks_distance(observed_pmf, _TRAINING_BASELINE["lace_histogram"])
        if ks >= _THRESHOLDS["ks_distance_alert"]:
            tier = "alert"
        elif ks >= _THRESHOLDS["ks_distance_warn"]:
            tier = "warn"
        else:
            tier = "ok"
        signals.append(DriftSignal(
            name="lace_distribution_ks",
            value=ks, baseline=0.0, tier=tier,
            detail=f"Cumulative-distribution gap vs training LACE histogram.",
        ))

    abstain_rate = abstain_count / n
    if abstain_rate >= _THRESHOLDS["abstain_rate_alert"]:
        tier = "alert"
    elif abstain_rate >= _THRESHOLDS["abstain_rate_warn"]:
        tier = "warn"
    else:
        tier = "ok"
    signals.append(DriftSignal(
        name="abstain_rate", value=abstain_rate,
        baseline=_TRAINING_BASELINE["abstain_rate"], tier=tier,
        detail=f"{abstain_count} of {n} cards fired ≥1 abstain trigger.",
    ))

    if ci_widths:
        mean_ci = float(np.mean(ci_widths))
        delta_ci = mean_ci - _TRAINING_BASELINE["ci_width_p50"]
        if delta_ci >= _THRESHOLDS["ci_width_drift_alert"]:
            tier = "alert"
        elif delta_ci >= _THRESHOLDS["ci_width_drift_warn"]:
            tier = "warn"
        else:
            tier = "ok"
        signals.append(DriftSignal(
            name="ci_width_drift", value=mean_ci,
            baseline=_TRAINING_BASELINE["ci_width_p50"], tier=tier,
            detail=f"Mean predicted CI width over the window.",
        ))

    # Post-hoc ECE / AUROC when outcomes available
    ece_val: float | None = None
    auroc_val: float | None = None
    n_outcomes = 0
    if outcomes_observed and probs:
        matched_probs: list[float] = []
        matched_outcomes: list[int] = []
        for rid, p in zip(request_ids, probs):
            if rid in outcomes_observed:
                matched_probs.append(p)
                matched_outcomes.append(int(outcomes_observed[rid]))
        n_outcomes = len(matched_probs)
        if n_outcomes >= 5:
            ece_val = _ece(matched_probs, matched_outcomes)
            auroc_val = _auroc(matched_probs, matched_outcomes)
            if ece_val >= _THRESHOLDS["ece_alert"]:
                tier = "alert"
            elif ece_val >= _THRESHOLDS["ece_warn"]:
                tier = "warn"
            else:
                tier = "ok"
            signals.append(DriftSignal(
                name="ece_post_hoc", value=ece_val, baseline=0.05,
                tier=tier,
                detail=f"Post-hoc ECE over {n_outcomes} cards with outcomes.",
            ))

    overall_tier = "ok"
    if any(s.tier == "alert" for s in signals):
        overall_tier = "alert"
    elif any(s.tier == "warn" for s in signals):
        overall_tier = "warn"

    rationale = _build_rationale(n, signals, overall_tier, n_outcomes)

    return CalibrationDriftReport(
        window_label=f"past_{window_hours}h",
        window_n_cards=n,
        window_start=cards[0][1] if cards else cutoff,
        window_end=cards[-1][1] if cards else now,
        overall_tier=overall_tier,
        signals=signals,
        ece_post_hoc=ece_val,
        auroc_post_hoc=auroc_val,
        n_outcomes_observed=n_outcomes,
        rationale=rationale,
    )


def _build_rationale(n: int, signals: list[DriftSignal],
                       overall_tier: str, n_outcomes: int) -> str:
    parts = [f"Analyzed {n} DecisionCard(s) in window."]
    flagged = [s for s in signals if s.tier != "ok"]
    if flagged:
        parts.append(
            f"Flagged signals ({len(flagged)}): "
            + ", ".join(f"{s.name}={s.value:.3f} ({s.tier})" for s in flagged)
        )
    else:
        parts.append("No signals breached the warn/alert thresholds.")
    if n_outcomes > 0:
        parts.append(f"Post-hoc outcome match: {n_outcomes} cards.")
    parts.append(f"Overall tier: {overall_tier.upper()}.")
    return " ".join(parts)
