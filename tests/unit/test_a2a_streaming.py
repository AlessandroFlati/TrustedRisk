"""Phase 3.3 -- A2A streaming + SSE event helpers tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from a2a_agent import streaming as st


# ─────────────────────── SSE wire-format ───────────────────────

def test_sse_event_format():
    raw = st.sse_event("hello", {"x": 1})
    assert raw.startswith("event: hello\n")
    assert "\ndata: " in raw
    assert raw.endswith("\n\n")
    body_line = next(line for line in raw.splitlines() if line.startswith("data: "))
    assert json.loads(body_line[len("data: "):]) == {"x": 1}


def test_status_update_carries_taskId_state_and_progress():
    raw = st.sse_status_update(
        "WORKING", task_id="t-1", context_id="ctx-1",
        progress_pct=0.42, message="halfway",
    )
    body = json.loads(raw.splitlines()[1][len("data: "):])
    assert body["kind"] == "TaskStatusUpdateEvent"
    assert body["taskId"] == "t-1"
    assert body["contextId"] == "ctx-1"
    assert body["status"]["state"] == "WORKING"
    assert body["status"]["progress_pct"] == 0.42


def test_artifact_update_carries_artifact_payload():
    raw = st.sse_artifact_update(
        {"foo": "bar"}, task_id="t-1", artifact_id="art-1",
    )
    body = json.loads(raw.splitlines()[1][len("data: "):])
    assert body["kind"] == "TaskArtifactUpdateEvent"
    assert body["artifact"]["artifactId"] == "art-1"
    assert body["artifact"]["parts"][0]["data"] == {"foo": "bar"}


def test_done_marker_format():
    raw = st.sse_done(task_id="t-1", context_id="ctx-1")
    body = json.loads(raw.splitlines()[1][len("data: "):])
    assert body["kind"] == "Done"
    assert body["taskId"] == "t-1"
    assert body["contextId"] == "ctx-1"


# ─────────────────────── Streaming generator ───────────────────────

def test_outcomes_stream_emits_progress_and_artifact():
    case_mix = [
        {
            "name": "all-cause readmission",
            "n_patients_per_year": 1000,
            "baseline_event_probability": 0.18,
        },
    ]

    async def collect():
        events: list[str] = []
        async for ev in st.stream_outcomes_simulation(
            case_mix=case_mix,
            intervention_rrr=0.25,
            intervention_cost_per_patient_usd=75.0,
            avoided_event_cost_usd=14000.0,
            n_iterations=200,
            chunk_size=50,
            task_id="t-stream-1",
        ):
            events.append(ev)
        return events

    events = asyncio.run(collect())
    # Each event is a complete SSE block ending in \n\n
    for ev in events:
        assert ev.endswith("\n\n")

    # Categorise events
    statuses = [e for e in events if e.startswith("event: status")]
    artifacts = [e for e in events if e.startswith("event: artifact")]
    dones = [e for e in events if e.startswith("event: done")]

    assert len(statuses) >= 2     # initial WORKING + final COMPLETED at minimum
    assert len(artifacts) == 1    # one artifact = the final result
    assert len(dones) == 1

    # The artifact carries the simulation result
    art_body = json.loads(
        artifacts[0].splitlines()[1][len("data: "):]
    )
    payload = art_body["artifact"]["parts"][0]["data"]
    # The HospitalYearSimulation payload carries cohort + cost fields
    assert "cohort_size_per_year" in payload
    assert "intervention_cost_per_patient_usd" in payload


def test_outcomes_stream_progress_monotonic():
    """Every status event with `progress_pct` must be ≥ the previous one."""
    case_mix = [{
        "name": "x", "n_patients_per_year": 100,
        "baseline_event_probability": 0.1,
    }]

    async def collect_progress():
        out: list[float] = []
        async for ev in st.stream_outcomes_simulation(
            case_mix=case_mix,
            intervention_rrr=0.25,
            intervention_cost_per_patient_usd=75.0,
            avoided_event_cost_usd=14000.0,
            n_iterations=100, chunk_size=20,
        ):
            if ev.startswith("event: status"):
                body = json.loads(ev.splitlines()[1][len("data: "):])
                p = body.get("status", {}).get("progress_pct")
                if p is not None:
                    out.append(p)
        return out

    progress = asyncio.run(collect_progress())
    assert progress == sorted(progress)
    assert progress[-1] == 1.0


def test_outcomes_stream_terminal_state_completed():
    case_mix = [{
        "name": "x", "n_patients_per_year": 100,
        "baseline_event_probability": 0.1,
    }]

    async def collect():
        out: list[str] = []
        async for ev in st.stream_outcomes_simulation(
            case_mix=case_mix,
            intervention_rrr=0.25,
            intervention_cost_per_patient_usd=75.0,
            avoided_event_cost_usd=14000.0,
            n_iterations=100,
        ):
            out.append(ev)
        return out

    events = asyncio.run(collect())
    # Find the last status event
    statuses = [e for e in events if e.startswith("event: status")]
    last = json.loads(statuses[-1].splitlines()[1][len("data: "):])
    assert last["status"]["state"] == "COMPLETED"


# ─────────────────────── HTTP endpoint smoke ───────────────────────

def test_streaming_endpoint_serves_sse():
    """Smoke test: the /api/stream/outcomes-simulation endpoint emits
    a 200 stream with content-type text/event-stream when called with
    valid SHARP context."""
    from starlette.testclient import TestClient
    from mcp_server.server import build_http_app

    app = build_http_app()
    client = TestClient(app)
    headers = {
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
    }
    body = {
        "case_mix": [{
            "name": "x", "n_patients_per_year": 100,
            "baseline_event_probability": 0.1,
        }],
        "intervention_rrr": 0.25,
        "n_iterations": 100, "chunk_size": 20,
    }
    with client.stream("POST", "/api/stream/outcomes-simulation",
                            json=body, headers=headers) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # Read first line block
        first_chunk = next(resp.iter_bytes())
        assert b"event: status" in first_chunk


def test_streaming_endpoint_rejects_missing_case_mix():
    from starlette.testclient import TestClient
    from mcp_server.server import build_http_app

    app = build_http_app()
    client = TestClient(app)
    headers = {
        "X-FHIR-Server-URL": "http://hapi/fhir",
        "X-FHIR-Access-Token": "tok",
    }
    resp = client.post("/api/stream/outcomes-simulation",
                          json={"intervention_rrr": 0.25}, headers=headers)
    assert resp.status_code == 400
    assert resp.json()["error"] == "missing_case_mix"
