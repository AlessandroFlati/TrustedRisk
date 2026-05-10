"""TIME-1 -- Event-sourced patient state log.

SQLite-backed append-only log of FHIR resource updates with deterministic
replay. The log captures every Observation / Condition / MedicationRequest
/ Procedure / Encounter as a separate event so the patient's state at any
historical timestamp can be reconstructed by replaying events in order.

Use cases:
  - audit forensics ("what did the chart look like at 2025-04-29 14:00?")
  - reproducibility (replay to a fixed timestamp + verify byte-equivalence)
  - streaming risk recomputation (consume events as they arrive -- see
    `incremental_risk` for TIME-2)

Append-only contract: rows are NEVER updated. Corrections are themselves
new events with a `payload.correction_for` reference.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from shared.schemas import (
    EventLogEntry,
    PatientStateSnapshot,
)


_DB_LOCK = threading.Lock()
_DEFAULT_PATH = "data/event_log.sqlite3"


def _db_path() -> Path:
    return Path(os.environ.get("TRUSTEDRISK_EVENT_LOG_PATH",
                                  _DEFAULT_PATH))


@contextlib.contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        with _DB_LOCK:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_log (
                    event_id TEXT PRIMARY KEY,
                    sequence_number INTEGER NOT NULL,
                    patient_id TEXT NOT NULL,
                    fhir_resource_type TEXT NOT NULL,
                    fhir_resource_id TEXT,
                    payload_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at_iso TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_log_patient_seq
                ON event_log (patient_id, sequence_number)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_event_log_patient_time
                ON event_log (patient_id, created_at_iso)
            """)
            conn.commit()
        yield conn
    finally:
        conn.close()


def _next_sequence(conn: sqlite3.Connection, patient_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(sequence_number), -1) AS max_seq "
        "FROM event_log WHERE patient_id = ?",
        (patient_id,),
    ).fetchone()
    return int(row["max_seq"]) + 1


def append_event(
    *,
    patient_id: str,
    fhir_resource_type: str,
    payload: dict[str, Any],
    fhir_resource_id: str | None = None,
    source: str = "ehr",
    created_at_iso: str | None = None,
) -> EventLogEntry:
    """Append one event to the log. Returns the persisted entry."""
    if not patient_id or not isinstance(patient_id, str):
        raise ValueError("patient_id is required.")
    if not fhir_resource_type or not isinstance(fhir_resource_type, str):
        raise ValueError("fhir_resource_type is required.")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict.")
    event_id = uuid.uuid4().hex
    timestamp = created_at_iso or datetime.now(timezone.utc).isoformat()

    with _conn() as c:
        seq = _next_sequence(c, patient_id)
        c.execute(
            """INSERT INTO event_log
                 (event_id, sequence_number, patient_id, fhir_resource_type,
                  fhir_resource_id, payload_json, source, created_at_iso)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_id, seq, patient_id, fhir_resource_type,
             fhir_resource_id, json.dumps(payload, default=str),
             source, timestamp),
        )
        c.commit()

    return EventLogEntry(
        event_id=event_id, patient_id=patient_id,
        fhir_resource_type=fhir_resource_type,
        fhir_resource_id=fhir_resource_id,
        payload=payload, source=source,
        created_at_iso=timestamp,
        sequence_number=seq,
    )


def list_events(
    patient_id: str,
    limit: int = 200,
    until_iso: str | None = None,
) -> list[EventLogEntry]:
    """Return events for a patient (sorted ascending by sequence_number)."""
    if not patient_id:
        return []
    if limit < 1 or limit > 10_000:
        raise ValueError("limit must be in [1, 10000].")
    with _conn() as c:
        if until_iso:
            rows = c.execute(
                "SELECT * FROM event_log "
                "WHERE patient_id = ? AND created_at_iso <= ? "
                "ORDER BY sequence_number ASC LIMIT ?",
                (patient_id, until_iso, limit),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM event_log WHERE patient_id = ? "
                "ORDER BY sequence_number ASC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
    return [
        EventLogEntry(
            event_id=r["event_id"],
            patient_id=r["patient_id"],
            fhir_resource_type=r["fhir_resource_type"],
            fhir_resource_id=r["fhir_resource_id"],
            payload=json.loads(r["payload_json"]),
            source=r["source"],
            created_at_iso=r["created_at_iso"],
            sequence_number=int(r["sequence_number"]),
        )
        for r in rows
    ]


def replay_to_state(
    patient_id: str,
    until_iso: str | None = None,
) -> PatientStateSnapshot:
    """Replay events in order to reconstruct the patient state.

    Resources of the same type collapse on `id` (later events of the same
    resource_id replace earlier ones). Resources without an id are
    appended (e.g. unindexed Observations).
    """
    events = list_events(patient_id, limit=10_000, until_iso=until_iso)
    by_type_by_id: dict[str, dict[str, dict[str, Any]]] = {}
    by_type_unindexed: dict[str, list[dict[str, Any]]] = {}

    for ev in events:
        rtype = ev.fhir_resource_type
        rid = ev.fhir_resource_id
        if rid:
            by_type_by_id.setdefault(rtype, {})[rid] = ev.payload
        else:
            by_type_unindexed.setdefault(rtype, []).append(ev.payload)

    resources_by_type: dict[str, list[dict[str, Any]]] = {}
    for rtype, entries in by_type_by_id.items():
        resources_by_type[rtype] = list(entries.values())
    for rtype, entries in by_type_unindexed.items():
        resources_by_type.setdefault(rtype, []).extend(entries)

    # Deterministic state hash over the (sorted) materialized resources
    h = hashlib.sha256()
    for rtype in sorted(resources_by_type):
        for resource in resources_by_type[rtype]:
            h.update(json.dumps(resource, sort_keys=True,
                                  default=str).encode("utf-8"))
            h.update(b"|")

    return PatientStateSnapshot(
        patient_id=patient_id,
        n_events_replayed=len(events),
        as_of_iso=until_iso,
        resources_by_type=resources_by_type,
        state_hash=h.hexdigest(),
    )


def truncate_log(patient_id: str | None = None) -> int:
    """Test helper -- wipe the log for a patient (or everything)."""
    with _conn() as c:
        if patient_id:
            cur = c.execute("DELETE FROM event_log WHERE patient_id = ?",
                              (patient_id,))
        else:
            cur = c.execute("DELETE FROM event_log")
        c.commit()
        return cur.rowcount or 0
