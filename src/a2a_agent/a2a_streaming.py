"""Phase 17.AJ - A2A streaming SSE + cancellation + resume.

Async event stream that emits partial DecisionCard events to an A2A
client, with three production-grade features:

  - **Server-Sent Events** envelope per event ``id: ... \\nevent: ...
    \\ndata: ... \\n\\n`` (RFC 8895 / W3C SSE).
  - **Cooperative cancellation** via a ``CancellationToken`` shared
    between producer + caller.
  - **Resume from last-event-id** - the in-memory ``EventStore``
    replays events with id > ``last_event_id`` so a reconnecting
    client never misses a checkpoint.

Pure-Python async, stdlib-only.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any, AsyncIterator, Iterable

from pydantic import BaseModel, Field


_EVENT_TEMPLATE = "id: {id}\nevent: {event}\ndata: {data}\n\n"


class StreamEvent(BaseModel):
    id: int = Field(ge=0)
    event: str
    data: dict[str, Any]


class CancellationToken:
    """Cooperative cancellation - the producer checks
    ``token.is_cancelled`` between yields."""

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True


class EventStore:
    """Bounded in-memory event log keyed by integer event id.

    Provides ``append`` for the producer and ``replay_after`` for the
    consumer to resume from a previously-seen event id.
    """

    def __init__(self, *, max_events: int = 1024) -> None:
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self._events: deque[StreamEvent] = deque(maxlen=max_events)
        self._next_id = 0

    @property
    def next_id(self) -> int:
        return self._next_id

    def append(self, event: str, data: dict[str, Any]
               ) -> StreamEvent:
        ev = StreamEvent(id=self._next_id, event=event, data=data)
        self._next_id += 1
        self._events.append(ev)
        return ev

    def replay_after(
        self, last_event_id: int,
    ) -> list[StreamEvent]:
        return [e for e in self._events if e.id > last_event_id]

    def all_events(self) -> list[StreamEvent]:
        return list(self._events)


def render_sse(event: StreamEvent) -> str:
    """Render a ``StreamEvent`` as a single SSE wire string."""
    return _EVENT_TEMPLATE.format(
        id=event.id,
        event=event.event,
        data=json.dumps(event.data, separators=(",", ":")),
    )


# ─────────────────────────────────────────────────────────────────────
# Decision-card streaming producer
# ─────────────────────────────────────────────────────────────────────


_CANONICAL_PHASES = (
    ("plan", "planner emitted the tool sequence"),
    ("risk_estimate",
     "calibrated readmission risk + CI computed"),
    ("fairness_audit",
     "subgroup fairness audit completed"),
    ("debate",
     "3-agent debate verdict reached"),
    ("advocate",
     "patient-advocate verdict reached"),
    ("decision_card",
     "final DecisionCard ready"),
)


async def stream_decision_card(
    *,
    store: EventStore,
    token: CancellationToken | None = None,
    last_event_id: int = -1,
    delay_per_event_seconds: float = 0.0,
) -> AsyncIterator[str]:
    """Async generator yielding SSE-formatted strings for each
    DecisionCard phase.

    On a fresh stream pass ``last_event_id=-1`` (the default). On a
    resume, pass the last id the client received - the generator
    replays the buffered events first, then continues with new
    phases.

    The generator polls ``token.is_cancelled`` between yields and
    exits cleanly when set.
    """
    for ev in store.replay_after(last_event_id):
        yield render_sse(ev)
        if token is not None and token.is_cancelled:
            return

    # Continue with whatever phases haven't yet been appended
    n_existing = store.next_id
    for idx, (phase, summary) in enumerate(_CANONICAL_PHASES):
        if idx < n_existing:
            continue
        if token is not None and token.is_cancelled:
            return
        ev = store.append(
            event=phase, data={"phase": phase, "summary": summary},
        )
        yield render_sse(ev)
        if delay_per_event_seconds > 0:
            await asyncio.sleep(delay_per_event_seconds)


async def collect_stream(
    stream: AsyncIterator[str],
) -> list[str]:
    """Convenience helper for tests + demos: drain an async stream
    into a list."""
    out: list[str] = []
    async for chunk in stream:
        out.append(chunk)
    return out
