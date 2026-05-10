"""Phase 15.A4 -- E2E showcase v7 smoke test."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "e2e_showcase_v7.py"


def _load_v7():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("e2e_showcase_v7", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e2e_showcase_v7"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Per-scenario invariants
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "factory,expected_min_calls",
    [
        ("s1_sepsis_cascade", 5),
        ("s2_cardiology_acs", 5),
        ("s3_heme_onc", 4),
        ("s4_periop_rheum", 5),
        ("s5_neuro_emergency", 5),
        ("s6_ob_peds", 4),
        ("s7_gi_hep_id", 5),
        ("s8_outpatient_panel", 10),
    ],
)
def test_each_scenario_yields_expected_number_of_calls(
    factory: str, expected_min_calls: int,
):
    v7 = _load_v7()
    fn = getattr(v7, factory)
    sc = _run(fn())
    assert len(sc.calls) >= expected_min_calls
    for c in sc.calls:
        assert c.tool.startswith("compute_")
        assert c.bundle != ""
        assert c.latency_ms >= 0


# ─────────────────────────────────────────────────────────────────────
# Bundle coverage across the 8 scenarios
# ─────────────────────────────────────────────────────────────────────

_REQUIRED_BUNDLES = {
    "critical_care",
    "cardiology_depth",
    "heme_onc_depth",
    "rheumatology",
    "peri_op_risk",
    "infectious_disease",
    "gi_hepatology_depth",
    "neurology_depth",
    "ob_peds_advanced",
    "sleep_pain",
    "endocrinology_advanced",
    "transplant",
    "specialty_clinics",
    "model_research",
    "legacy_ehr_parsers",
}


def test_v7_covers_every_phase_13_14_bundle():
    v7 = _load_v7()
    factories = [
        v7.s1_sepsis_cascade, v7.s2_cardiology_acs,
        v7.s3_heme_onc, v7.s4_periop_rheum,
        v7.s5_neuro_emergency, v7.s6_ob_peds,
        v7.s7_gi_hep_id, v7.s8_outpatient_panel,
    ]
    bundles: set[str] = set()
    for f in factories:
        sc = _run(f())
        bundles.update(sc.bundles_touched)
    missing = _REQUIRED_BUNDLES - bundles
    assert not missing, (
        f"v7 doesn't cover Phase 13/14 bundles: {sorted(missing)}"
    )


# ─────────────────────────────────────────────────────────────────────
# HTML output presence (after `python scripts/e2e_showcase_v7.py`)
# ─────────────────────────────────────────────────────────────────────

OUT = ROOT / "docs" / "e2e" / "v7" / "index.html"


@pytest.mark.skipif(not OUT.exists(),
                    reason="run scripts/e2e_showcase_v7.py first")
def test_html_output_exists_and_non_trivial_size():
    assert OUT.stat().st_size > 10_000


@pytest.mark.skipif(not OUT.exists(),
                    reason="run scripts/e2e_showcase_v7.py first")
def test_html_mentions_every_required_bundle():
    text = OUT.read_text(encoding="utf-8")
    for bundle in _REQUIRED_BUNDLES:
        assert bundle in text, f"missing bundle pill: {bundle}"
