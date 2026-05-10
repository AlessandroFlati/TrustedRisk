"""Phase 14.12 L3 -- Background scheduler tests."""

from __future__ import annotations

import asyncio

import pytest

from a2a_agent.scheduler import (
    JobRun, get_history, list_jobs, register_default_jobs,
    register_job, reset_registry, tick,
)


# ─────────────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────────────

def test_register_job_validates_positive_interval():
    reset_registry()
    with pytest.raises(ValueError):
        register_job("bad", 0, lambda: None)


def test_first_tick_fires_every_registered_job():
    reset_registry()

    async def hello(): return "ok"
    register_job("a", 60, hello)
    register_job("b", 60, hello)
    fired = tick(now=1000.0)
    assert {f.job_name for f in fired} == {"a", "b"}


def test_second_tick_within_interval_does_not_re_fire():
    reset_registry()

    async def hello(): return "ok"
    register_job("a", 60, hello)
    tick(now=1000.0)
    fired_2 = tick(now=1010.0)   # 10s after, interval is 60s
    assert fired_2 == []


def test_tick_after_interval_re_fires():
    reset_registry()

    async def hello(): return "ok"
    register_job("a", 60, hello)
    tick(now=1000.0)
    fired = tick(now=1100.0)    # 100s after
    assert len(fired) == 1
    assert fired[0].job_name == "a"


# ─────────────────────────────────────────────────────────────────────
# Status + error capture
# ─────────────────────────────────────────────────────────────────────

def test_failing_job_marks_status_failed():
    reset_registry()

    async def boom(): raise RuntimeError("kaboom")
    register_job("boom", 60, boom)
    fired = tick(now=1000.0)
    assert fired[0].status == "failed"
    assert "kaboom" in (fired[0].error or "")


def test_success_summary_captured():
    reset_registry()

    async def yields(): return {"ok": 1}
    register_job("y", 60, yields)
    fired = tick(now=1000.0)
    assert fired[0].status == "succeeded"
    assert "1" in (fired[0].output_summary or "")


# ─────────────────────────────────────────────────────────────────────
# History bound
# ─────────────────────────────────────────────────────────────────────

def test_history_grows_per_run():
    reset_registry()

    async def hello(): return "x"
    register_job("h", 1, hello)
    for i in range(5):
        tick(now=1000.0 + i * 2)
    history = get_history("h")
    assert len(history) == 5


def test_history_capped_at_256():
    reset_registry()

    async def hello(): return "x"
    register_job("h", 1, hello)
    for i in range(300):
        tick(now=1000.0 + i * 2)
    assert len(get_history("h")) == 256


# ─────────────────────────────────────────────────────────────────────
# Default registration
# ─────────────────────────────────────────────────────────────────────

def test_register_default_jobs_yields_two_jobs():
    reset_registry()
    register_default_jobs()
    assert set(list_jobs()) == {"care_gap_sweep", "drift_detection"}


def test_default_care_gap_job_runs_to_completion():
    reset_registry()
    register_default_jobs()
    fired = tick(now=1000.0)
    by_name = {f.job_name: f for f in fired}
    cg = by_name["care_gap_sweep"]
    assert cg.status == "succeeded"


def test_default_drift_detection_runs_without_artifact_or_skipped():
    """When the MIMIC artifact is absent the drift job returns
    flag=False; when present it returns the drift number. Either way
    the job must succeed."""
    reset_registry()
    register_default_jobs()
    fired = tick(now=1000.0)
    by_name = {f.job_name: f for f in fired}
    drift = by_name["drift_detection"]
    assert drift.status == "succeeded"
