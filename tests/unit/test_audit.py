"""Unit tests for AUDIT-1 + AUDIT-2: audit log + reproducibility archive."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from a2a_agent import audit


@pytest.fixture
def isolated_audit(tmp_path, monkeypatch):
    """Redirect audit log + archive to a temp dir for the test."""
    monkeypatch.setenv("TRUSTEDRISK_AUDIT_LOG_PATH",
                          str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("TRUSTEDRISK_REPRODUCIBILITY_DB_PATH",
                          str(tmp_path / "archive.sqlite3"))
    yield tmp_path


# ─────────────────────── Hash + redact ───────────────────────

def test_hash_deterministic():
    a = audit._hash({"x": 1, "y": [2, 3]})
    b = audit._hash({"y": [2, 3], "x": 1})  # same content, different order
    assert a == b
    assert a.startswith("sha256:")


def test_hash_empty():
    assert audit._hash(None) == "sha256:empty"


def test_redact_phi_fields():
    raw = {"name": "John Doe", "mrn": "1234567",
            "age": 70, "address": "123 Main St",
            "phone": "555-1234", "diagnosis": "CHF"}
    red = audit._redact(raw)
    assert red["name"] == "[REDACTED]"
    assert red["mrn"] == "[REDACTED]"
    assert red["address"] == "[REDACTED]"
    assert red["phone"] == "[REDACTED]"
    # Non-PHI fields preserved
    assert red["age"] == 70
    assert red["diagnosis"] == "CHF"


def test_redact_nested_lists():
    raw = {"patients": [{"name": "Alice"}, {"name": "Bob"}]}
    red = audit._redact(raw)
    assert all(p["name"] == "[REDACTED]" for p in red["patients"])


# ─────────────────────── Audit log append + read ───────────────────────

def test_append_and_read_audit_event(isolated_audit):
    audit.log_tool_call(
        tool="compute_readmission_risk",
        request_id="req-1", tenant_id="tenant-a",
        input_payload={"patient_id": "p1", "name": "Alice"},
        output_payload={"probability_mean": 0.15, "lace": 7},
        bundle_used="core_discharge",
        abstain_triggered=False, agreement="match",
    )
    events = audit.read_audit_log()
    assert len(events) == 1
    e = events[0]
    assert e["request_id"] == "req-1"
    assert e["tenant_id"] == "tenant-a"
    assert e["tool"] == "compute_readmission_risk"
    assert e["bundle_used"] == "core_discharge"
    assert e["agreement"] == "match"
    assert e["input_hash"].startswith("sha256:")
    assert e["output_hash"].startswith("sha256:")


def test_audit_log_redacts_input(isolated_audit):
    """The hashed input should NOT contain the patient name."""
    audit.log_tool_call(
        tool="compute_readmission_risk", request_id="req-redact",
        input_payload={"patient_id": "p1", "name": "John Doe",
                         "mrn": "1234567"},
        output_payload={"probability_mean": 0.10},
    )
    # Hash is over redacted payload -- same hash regardless of PHI content
    h_with_phi = audit._hash(audit._redact({
        "patient_id": "p1", "name": "John Doe", "mrn": "1234567",
    }))
    h_without_phi = audit._hash(audit._redact({
        "patient_id": "p1", "name": "Different", "mrn": "9999999",
    }))
    # Both reduce to "name":"[REDACTED]","mrn":"[REDACTED]" -> same hash
    assert h_with_phi == h_without_phi


def test_filter_by_request_id(isolated_audit):
    audit.log_tool_call(tool="t1", request_id="rid-1")
    audit.log_tool_call(tool="t2", request_id="rid-2")
    audit.log_tool_call(tool="t3", request_id="rid-1")
    events = audit.read_audit_log(request_id="rid-1")
    assert len(events) == 2
    assert all(e["request_id"] == "rid-1" for e in events)


def test_filter_by_tenant(isolated_audit):
    audit.log_tool_call(tool="t1", request_id="rid-1", tenant_id="tenant-a")
    audit.log_tool_call(tool="t2", request_id="rid-2", tenant_id="tenant-b")
    audit.log_tool_call(tool="t3", request_id="rid-3", tenant_id="tenant-a")
    events = audit.read_audit_log(tenant_id="tenant-a")
    assert len(events) == 2


def test_filter_abstain_only(isolated_audit):
    audit.log_tool_call(tool="t1", request_id="r1", abstain_triggered=False)
    audit.log_tool_call(tool="t2", request_id="r2", abstain_triggered=True)
    audit.log_tool_call(tool="t3", request_id="r3", abstain_triggered=True)
    events = audit.read_audit_log(abstain_only=True)
    assert len(events) == 2
    assert all(e["abstain_triggered"] for e in events)


def test_audit_log_jsonl_format(isolated_audit):
    """Verify on-disk format is JSONL (one JSON object per line)."""
    audit.log_tool_call(tool="t1", request_id="r1")
    audit.log_tool_call(tool="t2", request_id="r2")
    log_path = Path(os.environ["TRUSTEDRISK_AUDIT_LOG_PATH"])
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line must be valid JSON


# ─────────────────────── Reproducibility archive ───────────────────────

def test_archive_and_fetch(isolated_audit):
    audit.archive_decision(
        request_id="req-1",
        inputs={"patient_id": "p1", "lace": 7, "name": "Alice"},
        outputs={"probability_mean": 0.15, "action": "home_with_care"},
        coefficients_version="v1.0",
        tool_versions={"readmission_risk": "1.0.0"},
    )
    archived = audit.fetch_archived_decision("req-1")
    assert archived is not None
    assert archived["request_id"] == "req-1"
    assert archived["coefficients_version"] == "v1.0"
    assert archived["outputs"]["action"] == "home_with_care"
    # Inputs were redacted before archival
    assert archived["inputs"]["name"] == "[REDACTED]"
    assert archived["inputs"]["lace"] == 7


def test_fetch_archived_missing_returns_none(isolated_audit):
    assert audit.fetch_archived_decision("never-existed") is None


def test_list_archived_chronological(isolated_audit):
    for i in range(5):
        audit.archive_decision(
            request_id=f"req-{i}", inputs={"x": i},
            outputs={"y": i * 2},
        )
    listed = audit.list_archived_decisions(limit=10)
    assert len(listed) == 5
    # Most recent first
    assert listed[0]["request_id"] == "req-4"
    assert listed[-1]["request_id"] == "req-0"


# ─────────────────────── Reproducibility check ───────────────────────

def test_reproducibility_check_unknown_request(isolated_audit):
    result = audit.check_reproducibility("nonexistent")
    assert result.matched is False
    assert "not_archived" in result.archived_outputs_hash


def test_reproducibility_pre_flight_no_replay(isolated_audit):
    """No replay_fn -> just compares versions."""
    audit.archive_decision(
        request_id="req-1", inputs={"x": 1}, outputs={"y": 2},
        coefficients_version="v0.1",
    )
    result = audit.check_reproducibility("req-1", replay_fn=None)
    assert result.archived_outputs_hash.startswith("sha256:")
    assert result.matched is False  # can't match without replay
    # If current coefficient version == "v0.1" no drift; otherwise drift = True
    # The current version comes from data/coefficients.json
    assert isinstance(result.coefficients_drift, bool)


def test_reproducibility_replay_match(isolated_audit):
    """If the replay_fn returns the same outputs as archived -> matched."""
    archived_outputs = {"action": "discharge_home", "prob": 0.10}
    audit.archive_decision(
        request_id="req-replay",
        inputs={"patient_id": "p1"},
        outputs=archived_outputs,
    )
    def replay(inputs):
        return archived_outputs   # deterministic match
    result = audit.check_reproducibility("req-replay", replay_fn=replay)
    assert result.matched is True


def test_reproducibility_replay_mismatch(isolated_audit):
    """If the replay produces DIFFERENT outputs -> mismatch flagged."""
    audit.archive_decision(
        request_id="req-mismatch",
        inputs={"x": 1},
        outputs={"action": "discharge_home"},
    )
    def replay(inputs):
        return {"action": "snf"}   # different
    result = audit.check_reproducibility("req-mismatch", replay_fn=replay)
    assert result.matched is False
    assert result.archived_outputs_hash != result.replayed_outputs_hash


def test_reproducibility_replay_error(isolated_audit):
    audit.archive_decision(
        request_id="req-err", inputs={"x": 1}, outputs={"y": 2},
    )
    def replay(inputs):
        raise RuntimeError("simulated failure")
    result = audit.check_reproducibility("req-err", replay_fn=replay)
    assert result.matched is False
    assert "replay_error" in result.replayed_outputs_hash


def test_archive_persists_env_snapshot(isolated_audit, monkeypatch):
    """The archive should snapshot TRUSTEDRISK_* env vars for replay context."""
    monkeypatch.setenv("TRUSTEDRISK_FAIRNESS_BASELINE_PATH", "/some/path")
    audit.archive_decision(
        request_id="req-env", inputs={}, outputs={},
    )
    archived = audit.fetch_archived_decision("req-env")
    assert "TRUSTEDRISK_FAIRNESS_BASELINE_PATH" in archived["env_snapshot"]
