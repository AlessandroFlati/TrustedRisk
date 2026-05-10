"""HIPAA-style structured audit logging + reproducibility (AUDIT-1, AUDIT-2).

Two responsibilities:

  1. **Audit log** -- every tool call and DecisionCard emission is recorded
     in a structured append-only JSONL log with:
       tenant_id, request_id, timestamp_utc, actor (user/agent/service),
       tool, input_hash (SHA-256 of redacted input), output_hash,
       fhir_server_url (if any), bundle_used, abstain_triggered,
       agreement (LLM ↔ gate match)

     The log is HIPAA-style in that PHI is hashed (not stored) and the
     record carries enough audit trail to reconstruct who-did-what-when
     without exposing patient data.

  2. **Reproducibility archive** -- for each DecisionCard, a snapshot of
     all inputs (FHIR bundle, coefficients version, tool versions, env
     vars affecting calibration) is stored. Replay a past decision via
     `replay_decision(request_id)`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# ─────────────────────── Audit log path config ───────────────────────

def _audit_log_path() -> Path:
    p = Path(os.environ.get("TRUSTEDRISK_AUDIT_LOG_PATH",
                              "data/audit.jsonl"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _archive_db_path() -> Path:
    p = Path(os.environ.get("TRUSTEDRISK_REPRODUCIBILITY_DB_PATH",
                              "data/reproducibility_archive.sqlite3"))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ─────────────────────── Hashing helpers ───────────────────────

def _hash(payload: Any) -> str:
    """SHA-256 hash of a JSON-serializable payload. Stable across runs."""
    if payload is None:
        return "sha256:empty"
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(blob).hexdigest()}"


def _redact(payload: Any) -> Any:
    """Defensive PHI scrubber for log input. Removes obvious PHI fields
    from a dict before hashing. The hash is computed AFTER redaction so
    even the hash doesn't reveal PHI."""
    if isinstance(payload, dict):
        return {
            k: ("[REDACTED]" if k.lower() in {
                "name", "given", "family", "birthdate", "dob",
                "ssn", "mrn", "address", "phone", "email",
                "patient_name", "first_name", "last_name",
            } else _redact(v))
            for k, v in payload.items()
        }
    if isinstance(payload, list):
        return [_redact(v) for v in payload]
    return payload


# ─────────────────────── Audit event dataclass ───────────────────────

@dataclass
class AuditEvent:
    request_id: str
    timestamp_utc: str
    tenant_id: str
    actor: str               # "agent" / "user" / "cds-hook" / "federation-partner"
    tool: str                # e.g. "compute_readmission_risk", "POST /cds-services/..."
    input_hash: str
    output_hash: str
    bundle_used: str | None
    abstain_triggered: bool
    agreement: str | None    # match / safer / mismatch / n_a
    fhir_server_url: str | None = None
    extras: dict[str, Any] | None = None


def append_audit_event(event: AuditEvent) -> None:
    """Append a single event to the audit log (JSONL append-only)."""
    path = _audit_log_path()
    record = {
        "request_id": event.request_id,
        "timestamp_utc": event.timestamp_utc,
        "tenant_id": event.tenant_id,
        "actor": event.actor,
        "tool": event.tool,
        "input_hash": event.input_hash,
        "output_hash": event.output_hash,
        "bundle_used": event.bundle_used,
        "abstain_triggered": event.abstain_triggered,
        "agreement": event.agreement,
        "fhir_server_url": event.fhir_server_url,
        "extras": event.extras or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_tool_call(*, tool: str, request_id: str, tenant_id: str = "default",
                    actor: str = "agent",
                    input_payload: Any = None, output_payload: Any = None,
                    bundle_used: str | None = None,
                    abstain_triggered: bool = False,
                    agreement: str | None = None,
                    fhir_server_url: str | None = None,
                    extras: dict[str, Any] | None = None) -> AuditEvent:
    """Convenience wrapper: hash the input + output, build the event,
    persist it. Returns the event for the caller to inspect."""
    event = AuditEvent(
        request_id=request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        tenant_id=tenant_id, actor=actor, tool=tool,
        input_hash=_hash(_redact(input_payload)),
        output_hash=_hash(_redact(output_payload)),
        bundle_used=bundle_used,
        abstain_triggered=abstain_triggered,
        agreement=agreement,
        fhir_server_url=fhir_server_url, extras=extras,
    )
    append_audit_event(event)
    return event


def read_audit_log(limit: int | None = None,
                    request_id: str | None = None,
                    tenant_id: str | None = None,
                    abstain_only: bool = False) -> list[dict[str, Any]]:
    """Read events from the audit log with optional filters."""
    path = _audit_log_path()
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if request_id is not None and rec.get("request_id") != request_id:
                continue
            if tenant_id is not None and rec.get("tenant_id") != tenant_id:
                continue
            if abstain_only and not rec.get("abstain_triggered"):
                continue
            events.append(rec)
    if limit is not None:
        events = events[-limit:]
    return events


# ─────────────────────── Reproducibility archive ───────────────────────

_ARCHIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS decision_archive (
    request_id      TEXT PRIMARY KEY,
    archived_at_iso TEXT NOT NULL,
    inputs_json     TEXT NOT NULL,
    outputs_json    TEXT NOT NULL,
    coefficients_version TEXT,
    tool_versions_json   TEXT,
    env_snapshot_json    TEXT
);
"""


@contextmanager
def _archive_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_archive_db_path(), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(_ARCHIVE_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def archive_decision(*, request_id: str,
                       inputs: dict[str, Any],
                       outputs: dict[str, Any],
                       coefficients_version: str | None = None,
                       tool_versions: dict[str, str] | None = None) -> None:
    """Snapshot a DecisionCard's inputs + outputs + coefficient/tool
    versions for later replay."""
    env_snapshot = {
        k: v for k, v in os.environ.items()
        if k.startswith("TRUSTEDRISK_")
    }
    with _archive_conn() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO decision_archive
                 (request_id, archived_at_iso, inputs_json, outputs_json,
                  coefficients_version, tool_versions_json, env_snapshot_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id,
                datetime.now(timezone.utc).isoformat(),
                json.dumps(_redact(inputs), default=str),
                json.dumps(outputs, default=str),
                coefficients_version,
                json.dumps(tool_versions or {}),
                json.dumps(env_snapshot),
            ),
        )


def fetch_archived_decision(request_id: str) -> dict[str, Any] | None:
    with _archive_conn() as conn:
        row = conn.execute(
            "SELECT * FROM decision_archive WHERE request_id = ?",
            (request_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "request_id": row["request_id"],
        "archived_at_iso": row["archived_at_iso"],
        "inputs": json.loads(row["inputs_json"]),
        "outputs": json.loads(row["outputs_json"]),
        "coefficients_version": row["coefficients_version"],
        "tool_versions": json.loads(row["tool_versions_json"] or "{}"),
        "env_snapshot": json.loads(row["env_snapshot_json"] or "{}"),
    }


def list_archived_decisions(limit: int = 50) -> list[dict[str, Any]]:
    with _archive_conn() as conn:
        rows = conn.execute(
            """SELECT request_id, archived_at_iso, coefficients_version
               FROM decision_archive ORDER BY archived_at_iso DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [
        {"request_id": r["request_id"],
         "archived_at_iso": r["archived_at_iso"],
         "coefficients_version": r["coefficients_version"]}
        for r in rows
    ]


# ─────────────────────── Reproducibility check ───────────────────────

@dataclass
class ReproducibilityResult:
    request_id: str
    matched: bool
    archived_outputs_hash: str
    replayed_outputs_hash: str
    archived_coefficients_version: str | None
    current_coefficients_version: str | None
    coefficients_drift: bool       # True if version differs
    detail: str


def check_reproducibility(request_id: str,
                            replay_fn: Any = None) -> ReproducibilityResult:
    """Replay a past DecisionCard and verify byte-identical output.

    Args:
        request_id: the archived decision to replay.
        replay_fn: optional callable (inputs_dict) -> outputs_dict that
            re-runs the pipeline. If None, this function only compares
            archived vs current coefficient version (without re-running),
            useful for a "would this be reproducible today?" pre-flight.

    Returns:
        ReproducibilityResult with `matched` true iff the replay output
        matches the archive byte-for-byte (after JSON normalization).
    """
    archive = fetch_archived_decision(request_id)
    if archive is None:
        return ReproducibilityResult(
            request_id=request_id, matched=False,
            archived_outputs_hash="(not_archived)",
            replayed_outputs_hash="(no_replay)",
            archived_coefficients_version=None,
            current_coefficients_version=_current_coefficients_version(),
            coefficients_drift=True,
            detail=f"No archived decision found for request_id={request_id!r}",
        )

    arch_hash = _hash(archive["outputs"])
    cur_ver = _current_coefficients_version()
    arch_ver = archive["coefficients_version"]
    drift = (cur_ver != arch_ver)

    if replay_fn is None:
        return ReproducibilityResult(
            request_id=request_id, matched=False,
            archived_outputs_hash=arch_hash,
            replayed_outputs_hash="(replay_not_invoked)",
            archived_coefficients_version=arch_ver,
            current_coefficients_version=cur_ver,
            coefficients_drift=drift,
            detail=("Pre-flight check: archive present, replay_fn not "
                     "provided. Coefficient version drift = "
                     f"{drift} (archived={arch_ver}, current={cur_ver})."),
        )

    try:
        replayed = replay_fn(archive["inputs"])
    except Exception as exc:
        return ReproducibilityResult(
            request_id=request_id, matched=False,
            archived_outputs_hash=arch_hash,
            replayed_outputs_hash=f"(replay_error: {type(exc).__name__})",
            archived_coefficients_version=arch_ver,
            current_coefficients_version=cur_ver,
            coefficients_drift=drift,
            detail=f"Replay raised {type(exc).__name__}: {exc}",
        )

    replayed_hash = _hash(replayed)
    matched = (arch_hash == replayed_hash)
    detail = (
        f"Match: {matched}. archived_hash={arch_hash[:24]}..., "
        f"replayed_hash={replayed_hash[:24]}..., "
        f"coefficient_drift={drift}."
    )
    return ReproducibilityResult(
        request_id=request_id, matched=matched,
        archived_outputs_hash=arch_hash,
        replayed_outputs_hash=replayed_hash,
        archived_coefficients_version=arch_ver,
        current_coefficients_version=cur_ver,
        coefficients_drift=drift,
        detail=detail,
    )


def _current_coefficients_version() -> str | None:
    """Read the version field from the live coefficients.json."""
    try:
        path = Path(os.environ.get("TRUSTEDRISK_COEFFICIENTS_PATH",
                                      "data/coefficients.json"))
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8")).get("model_version")
    except Exception:
        return None
