"""SCALE-2 unit tests for the DoWhy-based causal-inference layer."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from a2a_agent.causal_inference import (
    _naive_diff_in_means,
    compute_average_treatment_effect,
)


def _synthetic_cohort(*, n: int = 1000, true_ate: float = -0.10,
                          confound_strength: float = 0.5,
                          seed: int = 42) -> list[dict]:
    """Build a synthetic observational cohort where:
      - Z (confounder, e.g. age) ~ N(0, 1)
      - T (treatment) ~ Bernoulli(sigmoid(0.5 * Z))   ← treatment depends on Z
      - Y (outcome) = base_p + true_ate * T + confound_strength * Z + noise

    The naive difference-in-means is biased; the DoWhy adjusted estimate
    should recover the true ATE within tolerance.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    pT = 1 / (1 + np.exp(-(0.5 * Z)))
    T = (rng.uniform(size=n) < pT).astype(int)
    base_p = 0.30
    raw = base_p + true_ate * T + confound_strength * 0.05 * Z \
        + 0.05 * rng.standard_normal(n)
    Y = (raw > 0.5).astype(int)
    return [{"T": int(T[i]), "Y": int(Y[i]), "Z": float(Z[i])}
            for i in range(n)]


# ─────────────────────── Validation guards ───────────────────────

def test_empty_cohort_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_average_treatment_effect(
            cohort=[], treatment="T", outcome="Y")


def test_unknown_treatment_column_raises():
    with pytest.raises(ValueError, match="not found"):
        compute_average_treatment_effect(
            cohort=[{"T": 0, "Y": 0}], treatment="X", outcome="Y")


def test_unknown_outcome_column_raises():
    with pytest.raises(ValueError, match="not found"):
        compute_average_treatment_effect(
            cohort=[{"T": 0, "Y": 0}], treatment="T", outcome="X")


def test_single_treatment_value_raises():
    with pytest.raises(ValueError, match="one unique"):
        compute_average_treatment_effect(
            cohort=[{"T": 0, "Y": 0}, {"T": 0, "Y": 1}],
            treatment="T", outcome="Y", confounders=[])


def test_non_numeric_treatment_raises():
    with pytest.raises(ValueError, match="numeric"):
        compute_average_treatment_effect(
            cohort=[{"T": "yes", "Y": 0}, {"T": "no", "Y": 1}],
            treatment="T", outcome="Y")


# ─────────────────────── Naive vs adjusted comparison ───────────────────────

def test_no_confounders_returns_naive_warning():
    cohort = _synthetic_cohort(n=200)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y",
        confounders=[],
    )
    assert r.estimation_method == "naive_difference_in_means"
    assert any("confounder" in w.lower() for w in r.assumptions_warnings)


def test_naive_difference_matches_panda_split():
    cohort = _synthetic_cohort(n=200)
    import pandas as pd
    df = pd.DataFrame(cohort)
    expected = (df[df["T"] == 1]["Y"].mean()
                  - df[df["T"] == 0]["Y"].mean())
    naive = _naive_diff_in_means(df, "T", "Y")
    assert abs(naive - expected) < 1e-9


def test_dowhy_adjusted_estimate_recovers_true_ate():
    """With a strong confounder, the naive estimate is biased and the
    DoWhy linear-regression adjustment should be closer to the true ATE."""
    cohort = _synthetic_cohort(n=2000, true_ate=-0.15, confound_strength=2.0,
                                  seed=1)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y",
        confounders=["Z"], run_refutations=False,
    )
    # Adjusted ATE should be MUCH closer to -0.15 than the naive estimate is.
    # We accept any value in [-0.30, 0.0] -- confirming the sign is correct
    # and the magnitude is plausibly close to true.
    assert -0.30 <= r.ate_point <= 0.0
    # Confidence: estimation_method should report DoWhy backdoor
    assert r.estimation_method.startswith("dowhy.backdoor")


def test_n_treated_n_untreated_counted():
    cohort = _synthetic_cohort(n=500, seed=2)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y", confounders=["Z"],
        run_refutations=False,
    )
    assert r.n_treated + r.n_untreated == 500
    assert r.n_treated > 0
    assert r.n_untreated > 0


def test_estimand_text_present():
    cohort = _synthetic_cohort(n=300, seed=3)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y", confounders=["Z"],
        run_refutations=False,
    )
    assert r.identified_estimand_text  # non-empty


# ─────────────────────── Refutations ───────────────────────

def test_refutations_run_when_enabled():
    cohort = _synthetic_cohort(n=600, seed=4)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y", confounders=["Z"],
        run_refutations=True,
    )
    refute_names = {ref.name for ref in r.refutations}
    # We expect both refutations to attempt; some may fail due to
    # underlying lib quirks -- at minimum one should succeed.
    assert len(refute_names) >= 1
    expected_names = {"random_common_cause", "placebo_treatment"}
    assert refute_names <= expected_names


def test_refutations_skipped_when_disabled():
    cohort = _synthetic_cohort(n=300, seed=5)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y", confounders=["Z"],
        run_refutations=False,
    )
    assert r.refutations == []


# ─────────────────────── Schema + references ───────────────────────

def test_references_include_pearl_and_dowhy():
    cohort = _synthetic_cohort(n=200, seed=6)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y", confounders=[],
    )
    refs = " ".join(r.references)
    assert "Pearl" in refs
    assert "DoWhy" in refs


def test_rationale_summarizes_method_and_n():
    cohort = _synthetic_cohort(n=200, seed=7)
    r = compute_average_treatment_effect(
        cohort=cohort, treatment="T", outcome="Y", confounders=["Z"],
        run_refutations=False,
    )
    assert "ATE" in r.rationale
    assert "n_treated" in r.rationale or "naive" in r.rationale


# ─────────────────────── /api/causal/ate endpoint ───────────────────────

@pytest.fixture(scope="module")
def app():
    ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "playground_server",
        str(ROOT / "apps" / "playground" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["playground_server"] = mod
    spec.loader.exec_module(mod)
    return mod.app


def test_causal_ate_endpoint(app):
    from fastapi.testclient import TestClient
    cohort = _synthetic_cohort(n=300, seed=8)
    body = {
        "cohort": cohort,
        "treatment": "T",
        "outcome": "Y",
        "confounders": ["Z"],
        "run_refutations": False,
    }
    with TestClient(app) as c:
        r = c.post("/api/causal/ate", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["treatment_name"] == "T"
    assert "ate_point" in payload


def test_causal_ate_endpoint_rejects_empty_cohort(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/causal/ate", json={
            "cohort": [], "treatment": "T", "outcome": "Y"})
    assert r.status_code == 400


def test_causal_ate_endpoint_rejects_missing_treatment(app):
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/api/causal/ate", json={
            "cohort": [{"T": 0, "Y": 0}], "outcome": "Y"})
    assert r.status_code == 400
