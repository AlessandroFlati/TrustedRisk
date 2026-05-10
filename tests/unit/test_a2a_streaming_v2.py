"""Phase 17.AJ - A2A streaming SSE tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from a2a_agent.a2a_streaming import (
    CancellationToken, EventStore, StreamEvent,
    collect_stream, render_sse, stream_decision_card,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# EventStore + SSE rendering
# ─────────────────────────────────────────────────────────────────────

def test_event_store_assigns_monotonic_ids():
    store = EventStore()
    a = store.append("a", {"x": 1})
    b = store.append("b", {"x": 2})
    assert a.id == 0
    assert b.id == 1
    assert store.next_id == 2


def test_event_store_replay_after_filters_by_id():
    store = EventStore()
    for i in range(5):
        store.append(f"e{i}", {"i": i})
    replay = store.replay_after(2)
    assert [e.id for e in replay] == [3, 4]


def test_event_store_max_events_caps_buffer():
    store = EventStore(max_events=3)
    for i in range(10):
        store.append(f"e{i}", {"i": i})
    all_events = store.all_events()
    assert len(all_events) == 3
    assert [e.event for e in all_events] == ["e7", "e8", "e9"]


def test_render_sse_emits_canonical_envelope():
    ev = StreamEvent(id=7, event="risk_estimate",
                     data={"prob": 0.158})
    sse = render_sse(ev)
    assert sse.startswith("id: 7\n")
    assert "event: risk_estimate\n" in sse
    assert "data: {" in sse
    assert sse.endswith("\n\n")


def test_event_store_rejects_zero_max_events():
    with pytest.raises(ValueError):
        EventStore(max_events=0)


# ─────────────────────────────────────────────────────────────────────
# Cancellation
# ─────────────────────────────────────────────────────────────────────

def test_cancellation_token_default_is_not_cancelled():
    token = CancellationToken()
    assert token.is_cancelled is False


def test_cancellation_token_cancel_flips_state():
    token = CancellationToken()
    token.cancel()
    assert token.is_cancelled is True


# ─────────────────────────────────────────────────────────────────────
# stream_decision_card
# ─────────────────────────────────────────────────────────────────────

def test_full_stream_emits_six_canonical_phases():
    store = EventStore()
    chunks = _run(collect_stream(
        stream_decision_card(store=store)
    ))
    assert len(chunks) == 6
    for phase in (
        "plan", "risk_estimate", "fairness_audit",
        "debate", "advocate", "decision_card",
    ):
        assert any(f"event: {phase}" in c for c in chunks)


def test_cancelled_stream_terminates_early():
    store = EventStore()
    token = CancellationToken()

    async def _go():
        out = []
        async for ev in stream_decision_card(
            store=store, token=token,
        ):
            out.append(ev)
            if len(out) == 2:
                token.cancel()
        return out
    chunks = _run(_go())
    assert len(chunks) == 2
    # Store should have only the events that were emitted
    assert len(store.all_events()) == 2


def test_resume_from_last_event_id_replays_buffered_events():
    store = EventStore()
    # Emit the full stream once
    _run(collect_stream(stream_decision_card(store=store)))
    # Now a "reconnecting" client resumes from id 2 -> should
    # receive events 3, 4, 5
    chunks = _run(collect_stream(
        stream_decision_card(store=store, last_event_id=2)
    ))
    assert len(chunks) == 3


def test_resume_from_last_event_id_with_no_new_events_only_replays():
    store = EventStore()
    _run(collect_stream(stream_decision_card(store=store)))
    chunks = _run(collect_stream(
        stream_decision_card(store=store, last_event_id=-1)
    ))
    # All 6 buffered events replayed; nothing new appended
    assert len(chunks) == 6
    assert store.next_id == 6


def test_event_data_is_valid_json():
    store = EventStore()
    chunks = _run(collect_stream(stream_decision_card(store=store)))
    for chunk in chunks:
        data_line = next(
            ln for ln in chunk.splitlines()
            if ln.startswith("data: ")
        )
        payload = data_line[len("data: "):]
        # Must round-trip through json
        json.loads(payload)
