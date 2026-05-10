"""Phase 15.B3 -- Tests for the subgroup_audit builder."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "build_subgroup_audit.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_subgroup_audit", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_subgroup_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────
# Cohort generation + composability
# ─────────────────────────────────────────────────────────────────────

def test_build_cohort_returns_n_rows():
    mod = _load_builder()
    rng = random.Random(7)
    cohort = mod._build_cohort(rng, n=500)
    assert len(cohort) == 500


def test_each_row_has_demographics():
    mod = _load_builder()
    rng = random.Random(8)
    cohort = mod._build_cohort(rng, n=200)
    for r in cohort:
        for k in ("age", "age_band", "sex", "race",
                   "ethnicity", "insurance_type", "language",
                   "lace", "outcome", "p_post"):
            assert k in r
        assert 0 <= r["p_post"] <= 1
        assert r["outcome"] in (0, 1)


def test_cohort_is_seeded_deterministic():
    mod = _load_builder()
    a = mod._build_cohort(random.Random(99), n=200)
    b = mod._build_cohort(random.Random(99), n=200)
    assert [r["lace"] for r in a] == [r["lace"] for r in b]
    assert [r["outcome"] for r in a] == [r["outcome"] for r in b]


# ─────────────────────────────────────────────────────────────────────
# Aggregation + gap calculations
# ─────────────────────────────────────────────────────────────────────

def test_aggregate_subgroup_emits_one_entry_per_value():
    mod = _load_builder()
    rng = random.Random(10)
    cohort = mod._build_cohort(rng, n=2000)
    by_race = mod._aggregate_subgroup(cohort, "race")
    assert set(by_race.keys()) <= set(mod._RACE_MULT.keys())
    for v, m in by_race.items():
        assert m["n"] >= 1
        assert 0 <= m["mean_predicted_rate"] <= 1
        assert 0 <= m["tpr_at_thr"] <= 1


def test_eoo_dp_gaps_reference_value_marker():
    mod = _load_builder()
    rng = random.Random(11)
    cohort = mod._build_cohort(rng, n=1500)
    by_race = mod._aggregate_subgroup(cohort, "race")
    gaps = mod._eoo_dp_gaps(by_race, "white")
    assert "white" in gaps
    assert gaps["white"]["reference_value"] == 1.0
    assert gaps["white"]["eoo_gap_vs_reference"] == 0.0
    for v, g in gaps.items():
        assert "eoo_gap_vs_reference" in g
        assert "dp_gap_vs_reference" in g


def test_eoo_dp_gaps_returns_empty_for_unknown_reference():
    mod = _load_builder()
    rng = random.Random(12)
    cohort = mod._build_cohort(rng, n=500)
    by_race = mod._aggregate_subgroup(cohort, "race")
    assert mod._eoo_dp_gaps(by_race, "this_value_does_not_exist") == {}


# ─────────────────────────────────────────────────────────────────────
# DP noise mechanism
# ─────────────────────────────────────────────────────────────────────

def test_noised_int_returns_non_negative():
    mod = _load_builder()
    rng = random.Random(13)
    for raw in (0, 5, 100, 10000):
        for _ in range(20):
            assert mod._noised_int(raw, 1.0, rng) >= 0


def test_noised_int_centred_on_raw_in_aggregate():
    mod = _load_builder()
    rng = random.Random(14)
    raw = 1000
    samples = [mod._noised_int(raw, 1.0, rng) for _ in range(2000)]
    avg = sum(samples) / len(samples)
    # Laplace(0,1) noise on a count of 1000 stays within ~30 on average
    assert abs(avg - raw) < 30


# ─────────────────────────────────────────────────────────────────────
# Calibration metrics
# ─────────────────────────────────────────────────────────────────────

def test_ece_zero_for_perfect_calibration():
    mod = _load_builder()
    pairs = [(0.5, 1), (0.5, 0)] * 500
    assert mod._ece(pairs) < 0.01


def test_brier_at_zero_for_certain_correct():
    mod = _load_builder()
    pairs = [(1.0, 1)] * 100 + [(0.0, 0)] * 100
    assert mod._brier(pairs) == 0.0


# ─────────────────────────────────────────────────────────────────────
# End-to-end artefact (uses the on-disk artefact written by main())
# ─────────────────────────────────────────────────────────────────────

ARTEFACT = ROOT / "docs" / "fairness" / "subgroup_audit.json"


@pytest.mark.skipif(not ARTEFACT.exists(),
                    reason="run scripts/build_subgroup_audit.py first")
def test_artefact_has_six_subgroup_axes():
    art = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert set(art["subgroups"].keys()) == {
        "age_band", "sex", "race", "ethnicity",
        "insurance_type", "language",
    }


@pytest.mark.skipif(not ARTEFACT.exists(),
                    reason="run scripts/build_subgroup_audit.py first")
def test_artefact_includes_dp_published_segments():
    art = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert art["differential_privacy"]["mechanism"] == "Laplace"
    assert art["differential_privacy"]["epsilon"] == 1.0
    n_segs = sum(len(v) for v in art["dp_published_for_release"].values())
    assert n_segs >= 1


@pytest.mark.skipif(not ARTEFACT.exists(),
                    reason="run scripts/build_subgroup_audit.py first")
def test_artefact_cohort_size_one_hundred_thousand():
    art = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert art["cohort_n"] == 100_000


@pytest.mark.skipif(not ARTEFACT.exists(),
                    reason="run scripts/build_subgroup_audit.py first")
def test_regulatory_pack_picks_up_fairness_artefact():
    """When the subgroup_audit.json artefact is on disk the regulatory
    pack's fairness section flips to artefact_present=True."""
    sys.path.insert(0, str(ROOT / "src"))
    from a2a_agent.regulatory_pack import build_regulatory_pack
    pack = build_regulatory_pack(project_root=ROOT)
    fairness = next(
        s for s in pack.sections if s.title.startswith("Fairness")
    )
    assert fairness.artefact_present is True
    assert pack.overall_artefact_coverage == 1.0
