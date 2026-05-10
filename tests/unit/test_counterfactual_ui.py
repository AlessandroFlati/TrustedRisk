"""Phase 17.AE - Interactive counterfactual UI smoke test."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "generate_counterfactual_ui.py"
OUT = ROOT / "docs" / "ui" / "counterfactual.html"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "generate_counterfactual_ui", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_counterfactual_ui"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_html_template_includes_lace_sliders():
    mod = _load_script()
    for slider_id in ('id="L"', 'id="A"', 'id="C"', 'id="E"'):
        assert slider_id in mod._HTML


def test_html_template_includes_calibration_lookup():
    mod = _load_script()
    # All five W1 spec_002 buckets must be present in the JS lookup
    for v in ("0.072", "0.103", "0.158", "0.234", "0.327"):
        assert v in mod._HTML


def test_html_template_includes_action_classes():
    mod = _load_script()
    for cls in (
        "a-discharge_home",
        "a-discharge_with_homecare",
        "a-continued_admission",
    ):
        assert cls in mod._HTML


def test_html_template_includes_min_l1_search():
    mod = _load_script()
    assert "findCounterfactual" in mod._HTML
    assert "BOUNDS" in mod._HTML


@pytest.mark.skipif(
    not OUT.exists(),
    reason="run scripts/generate_counterfactual_ui.py first",
)
def test_rendered_html_above_5kb():
    assert OUT.stat().st_size > 5_000


@pytest.mark.skipif(
    not OUT.exists(),
    reason="run scripts/generate_counterfactual_ui.py first",
)
def test_rendered_html_includes_timestamp_marker():
    text = OUT.read_text(encoding="utf-8")
    assert "<!-- generated" in text
