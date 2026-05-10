"""Phase 14.12 L3 -- Background scheduler.

Lightweight in-process declarative scheduler for periodic care-gap
sweep + drift detection jobs. No threads, no asyncio loop, no
external scheduler -- the runner is invocation-driven: every call to
`tick(now)` advances any due jobs and writes a structured log entry
per run.

Design:
  - Jobs are async callables registered with an interval.
  - State (last_run_at + run history) lives in a module-level dict
    so multiple ticks within one process see consistent state.
  - Run history is capped (default 256 per job) to keep memory bounded.
  - The runner is fully synchronous from the caller's POV (uses
    `asyncio.run` per job) so a cron-driven script can drive it.

Production deployments would swap to APScheduler / systemd timers /
Cloud Scheduler. The interface here is deliberately narrow so that
swap is a 1-day refactor.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class JobRun:
    job_name: str
    started_at: float
    finished_at: float
    duration_seconds: float
    status: str    # "succeeded" | "failed"
    error: str | None = None
    output_summary: str | None = None


@dataclass
class JobSpec:
    name: str
    interval_seconds: float
    callable: Callable[[], Awaitable[Any]]
    last_run_at: float = 0.0
    run_history: list[JobRun] = field(default_factory=list)


_REGISTRY: dict[str, JobSpec] = {}
_HISTORY_CAP = 256


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def register_job(
    name: str, interval_seconds: float,
    fn: Callable[[], Awaitable[Any]],
) -> None:
    """Register a periodic job. Overrides any prior registration."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")
    _REGISTRY[name] = JobSpec(
        name=name, interval_seconds=interval_seconds,
        callable=fn,
    )


def reset_registry() -> None:
    """Test-helper: drop all jobs."""
    _REGISTRY.clear()


def list_jobs() -> list[str]:
    return list(_REGISTRY.keys())


def get_history(name: str) -> list[JobRun]:
    """Return a snapshot of the named job's run history."""
    spec = _REGISTRY.get(name)
    if spec is None:
        return []
    return list(spec.run_history)


def tick(now: float | None = None) -> list[JobRun]:
    """Advance any due jobs. Returns the list of runs that fired
    on this tick. Idempotent -- calling twice in quick succession
    will only fire jobs whose interval has elapsed."""
    now = now if now is not None else time.time()
    fired: list[JobRun] = []
    for spec in _REGISTRY.values():
        if spec.last_run_at == 0:
            # First-ever tick fires every job
            fired.append(_run_job(spec, now))
            continue
        if (now - spec.last_run_at) >= spec.interval_seconds:
            fired.append(_run_job(spec, now))
    return fired


def _run_job(spec: JobSpec, now: float) -> JobRun:
    started = now
    error: str | None = None
    summary: str | None = None
    status = "succeeded"
    try:
        out = asyncio.run(spec.callable())
        summary = repr(out)[:200] if out is not None else None
    except Exception as exc:    # noqa: BLE001
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
    finished = time.time()
    run = JobRun(
        job_name=spec.name, started_at=started, finished_at=finished,
        duration_seconds=round(finished - started, 4),
        status=status, error=error, output_summary=summary,
    )
    spec.last_run_at = now
    spec.run_history.append(run)
    if len(spec.run_history) > _HISTORY_CAP:
        spec.run_history = spec.run_history[-_HISTORY_CAP:]
    return run


# ─────────────────────────────────────────────────────────────────────
# Pre-built jobs (care-gap sweep + drift detection)
# ─────────────────────────────────────────────────────────────────────


async def _care_gap_sweep_job() -> dict[str, Any]:
    """Run the priority ranking against a tiny synthetic cohort summary.

    In production this would pull a fresh cohort from the EHR; the
    scheduled-job hook is what wires this onto the cron tick."""
    from mcp_server.tools.quality_stars import (
        compute_care_gap_priority_ranking,
        compute_quality_measures_aggregate,
    )
    agg = await compute_quality_measures_aggregate(
        measurement_year=2025,
        cohort_summary={
            "BCS": {"numerator": 1400, "denominator": 1968},
            "COL": {"numerator": 3300, "denominator": 5000},
        },
    )
    ranking = await compute_care_gap_priority_ranking(
        aggregate=agg, top_n=5,
    )
    return {
        "n_actions": ranking.n_actions,
        "cumulative_qbp": ranking.cumulative_expected_qbp_dollars,
    }


async def _drift_detection_job() -> dict[str, Any]:
    """Compare the W1 published ECE vs the last MIMIC recal ECE; if
    the drift > 0.02 raise a flag."""
    import json as _json
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parent.parent.parent
    mim = _Path(root) / "data" / "mimic_iv_recalibration.json"
    if not mim.exists():
        return {"flag": False, "reason": "no_mimic_artifact"}
    raw = _json.loads(mim.read_text(encoding="utf-8"))
    ece = raw["metrics_overall"]["ece"]
    drift = abs(ece - 0.0078)
    return {"w1_ece": 0.0078, "current_ece": ece,
                 "drift": round(drift, 4),
                 "flag": drift > 0.02}


def register_default_jobs() -> None:
    """Register the production care-gap sweep + drift detection jobs.

    Tests reset the registry before invoking this if they want a
    clean slate."""
    register_job("care_gap_sweep", 24 * 3600, _care_gap_sweep_job)
    register_job("drift_detection", 6 * 3600, _drift_detection_job)
