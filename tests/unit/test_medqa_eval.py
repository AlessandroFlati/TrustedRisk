"""Phase 11.8 -- MedQA-USMLE-style eval harness tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from a2a_agent.medqa_eval import (
    MedQAItem,
    list_bench,
    run_medqa,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Bench shape
# ─────────────────────────────────────────────────────────────────────

def test_bench_size_is_at_least_30():
    bench = list_bench()
    assert len(bench) >= 30


def test_every_item_has_4_options_with_unique_letters():
    for item in list_bench():
        assert set(item.options.keys()) == {"A", "B", "C", "D"}
        # No duplicate option text within a single item
        assert len(set(item.options.values())) == 4


def test_every_correct_letter_in_set():
    for item in list_bench():
        assert item.correct in {"A", "B", "C", "D"}


def test_every_item_has_a_category():
    for item in list_bench():
        assert item.category and isinstance(item.category, str)


# ─────────────────────────────────────────────────────────────────────
# Floor accuracy
# ─────────────────────────────────────────────────────────────────────

def test_floor_accuracy_meets_baseline():
    """The deterministic floor must beat random (0.25) by a wide margin
    on this synthetic bench. We require ≥ 0.80."""
    rep = _run(run_medqa(use_llm=False))
    assert rep.floor_accuracy >= 0.80


def test_floor_per_category_no_zero_categories():
    """No category should be worse than random on the floor."""
    rep = _run(run_medqa(use_llm=False))
    for cat, acc in rep.by_category_floor.items():
        assert acc >= 0.25, f"Category {cat!r}: floor accuracy {acc} < 0.25"


def test_run_with_disabled_llm_keeps_llm_accuracy_none(monkeypatch):
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")
    rep = _run(run_medqa(use_llm=True))
    assert rep.llm_accuracy is None


# ─────────────────────────────────────────────────────────────────────
# Output artifact contract
# ─────────────────────────────────────────────────────────────────────

def test_results_artifact_exists_after_driver_run():
    """The driver writes both the JSON and the markdown."""
    json_path = REPO_ROOT / "docs" / "evals" / "medqa_run.json"
    md_path = REPO_ROOT / "docs" / "evals" / "MEDQA_RESULTS.md"
    if not (json_path.exists() and md_path.exists()):
        pytest.skip("Run `python -m a2a_agent.medqa_eval` first")
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    assert raw["n_items"] >= 30
    assert 0.0 <= raw["floor_accuracy"] <= 1.0


def test_in_memory_bench_substitution_works():
    """Caller can pass a custom bench list."""
    custom = [
        MedQAItem(
            qid="X1", category="custom",
            stem="What is 2 + 2 in clinical algebra?",
            options={"A": "3", "B": "4", "C": "5", "D": "6"},
            correct="B",
        ),
    ]
    rep = _run(run_medqa(bench=custom, use_llm=False))
    assert rep.n_items == 1
    # No keyword cues for "custom" -> falls back to ties -> A
    # We don't assert the floor letter; we only assert the run shape.
    assert rep.items[0].item_id == "X1"
