"""Phase 17.R - Story-mode showcase tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "generate_storymode.py"


def _load_storymode():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "generate_storymode", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["generate_storymode"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_six_patients_defined():
    mod = _load_storymode()
    cases = mod._patients()
    assert len(cases) == 6


def test_each_patient_has_unique_case_id():
    mod = _load_storymode()
    cases = mod._patients()
    ids = [c.case_id for c in cases]
    assert len(set(ids)) == len(ids)


def test_each_patient_has_chart_paragraph():
    mod = _load_storymode()
    cases = mod._patients()
    for c in cases:
        assert len(c.chart_text) >= 200


def test_pipeline_runs_for_every_case():
    mod = _load_storymode()
    cases = mod._patients()
    rendered = [mod._render_case(c) for c in cases]
    for r in rendered:
        assert "Multi-agent debate verdict" in r
        assert "Patient-advocate verdict" in r


def test_demographic_diversity_covered():
    mod = _load_storymode()
    cases = mod._patients()
    races = {c.advocate_input.patient_race for c in cases}
    insurances = {c.advocate_input.patient_insurance for c in cases}
    languages = {c.advocate_input.patient_language for c in cases}
    assert "black" in races
    assert "indigenous" in races
    assert "hispanic" in races
    assert "uninsured" in insurances
    assert "medicaid" in insurances
    assert "spanish" in languages
    assert "vietnamese" in languages


_HTML = ROOT / "docs" / "showcase" / "STORYMODE.html"


@pytest.mark.skipif(not _HTML.exists(),
                    reason="run scripts/generate_storymode.py first")
def test_html_output_above_15kb():
    assert _HTML.stat().st_size > 15_000


@pytest.mark.skipif(not _HTML.exists(),
                    reason="run scripts/generate_storymode.py first")
def test_html_mentions_all_six_case_ids():
    text = _HTML.read_text(encoding="utf-8")
    for cid in (f"STORY-{i}" for i in range(1, 7)):
        assert cid in text
