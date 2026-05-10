"""Phase 12.2 -- Audit Merkle proof verifier tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from a2a_agent.merkle_audit import (
    build_inclusion_proof,
    compute_merkle_audit_root,
    verify_audit_chain,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _events() -> list[dict]:
    return [
        {"event_id": "evt-001", "ts": "2026-04-30T01:00:00Z",
         "tool": "compute_readmission_risk",
         "patient_hash": "sha256:aaa", "result_hash": "sha256:bbb"},
        {"event_id": "evt-002", "ts": "2026-04-30T01:05:00Z",
         "tool": "compute_decision_utility",
         "patient_hash": "sha256:aaa", "result_hash": "sha256:ccc"},
        {"event_id": "evt-003", "ts": "2026-04-30T01:10:00Z",
         "tool": "detect_phi", "result_hash": "sha256:ddd"},
        {"event_id": "evt-004", "ts": "2026-04-30T01:15:00Z",
         "tool": "compute_fairness_audit",
         "result_hash": "sha256:eee"},
    ]


# ─────────────────────────────────────────────────────────────────────
# Pass case
# ─────────────────────────────────────────────────────────────────────

def test_verify_audit_chain_passes_for_intact_event():
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-002",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=proof.proof_steps,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "verified"
    assert report.recomputed_leaf_hash == proof.leaf_hash
    assert report.recomputed_root == proof.merkle_root


def test_verify_audit_chain_passes_for_first_event():
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-001")
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-001",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=proof.proof_steps,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "verified"


def test_verify_audit_chain_passes_for_last_event():
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-004")
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-004",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=proof.proof_steps,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "verified"


# ─────────────────────────────────────────────────────────────────────
# Tamper detection
# ─────────────────────────────────────────────────────────────────────

def test_tamper_event_detected_when_event_payload_mutated():
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    # Mutate one byte of evt-002 AFTER the proof was built
    tampered = [dict(e) for e in events]
    tampered[1]["result_hash"] = "sha256:ZZZ"   # changed
    report = verify_audit_chain(
        events=tampered,
        target_event_id="evt-002",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=proof.proof_steps,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "tamper_event"
    assert report.recomputed_leaf_hash != report.claimed_leaf_hash


def test_tamper_chain_detected_when_proof_step_mutated():
    """Leaf is intact but a sibling hash is wrong -> reconstruction fails."""
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    # Mutate one of the sibling hashes in the proof
    bad_proof = [s.model_copy() for s in proof.proof_steps]
    bad_proof[0].sibling_hash = "f" * 64
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-002",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=bad_proof,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "tamper_chain"
    assert report.recomputed_leaf_hash == report.claimed_leaf_hash
    assert report.recomputed_root != proof.merkle_root


def test_tamper_chain_detected_when_root_mutated():
    """Same leaf + proof but expected_root flipped -> tamper_chain."""
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    bad_root = "0" * 64
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-002",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=proof.proof_steps,
        expected_merkle_root=bad_root,
    )
    assert report.posture == "tamper_chain"


def test_unknown_event_id():
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-NOT-IN-LEDGER",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=proof.proof_steps,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "unknown_event"


# ─────────────────────────────────────────────────────────────────────
# Coercion: dict-shaped proof_steps work too
# ─────────────────────────────────────────────────────────────────────

def test_dict_shaped_proof_steps_are_accepted():
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-003")
    steps_as_dicts = [s.model_dump() for s in proof.proof_steps]
    report = verify_audit_chain(
        events=events,
        target_event_id="evt-003",
        claimed_leaf_hash=proof.leaf_hash,
        proof_steps=steps_as_dicts,
        expected_merkle_root=proof.merkle_root,
    )
    assert report.posture == "verified"


# ─────────────────────────────────────────────────────────────────────
# CLI smoke
# ─────────────────────────────────────────────────────────────────────

def test_cli_returns_0_on_verified_proof(tmp_path):
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    payload = {
        "events": events,
        "target_event_id": "evt-002",
        "claimed_leaf_hash": proof.leaf_hash,
        "proof_steps": [s.model_dump() for s in proof.proof_steps],
        "expected_merkle_root": proof.merkle_root,
    }
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cmd = [sys.executable,
              str(REPO_ROOT / "scripts" / "verify_audit_chain.py"),
              str(path)]
    env = {"PYTHONPATH": str(REPO_ROOT / "src")}
    import os
    env = {**os.environ, **env}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 0
    assert '"posture": "verified"' in r.stdout


def test_cli_returns_1_on_tampered_proof(tmp_path):
    events = _events()
    proof = build_inclusion_proof(events, target_event_id="evt-002")
    tampered = [dict(e) for e in events]
    tampered[1]["result_hash"] = "tampered"
    payload = {
        "events": tampered,
        "target_event_id": "evt-002",
        "claimed_leaf_hash": proof.leaf_hash,
        "proof_steps": [s.model_dump() for s in proof.proof_steps],
        "expected_merkle_root": proof.merkle_root,
    }
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cmd = [sys.executable,
              str(REPO_ROOT / "scripts" / "verify_audit_chain.py"),
              str(path)]
    import os
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert r.returncode == 1
    assert '"posture": "tamper_event"' in r.stdout
