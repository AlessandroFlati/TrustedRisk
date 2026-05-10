"""Phase 17.U - Federation chained-call demo tests."""

from __future__ import annotations

import pytest

from a2a_agent.federation_chain import (
    FederationChainTrace, run_federation_chain,
)


def _payload(**overrides):
    base = dict(
        patient_id="p-1", encounter_id="e-1", lace=6,
        demographics={
            "age": 70, "race": "white", "insurance": "medicare",
            "language": "english", "n_chronic_meds": 5,
            "has_home_caregiver_available": True,
            "has_transportation": True,
            "fairness_audit_present": True,
        },
    )
    base.update(overrides)
    return base


def test_chain_runs_three_hops_when_clean():
    trace = run_federation_chain(initial_payload=_payload())
    assert trace.n_hops == 3
    assert trace.final_verdict in ("forwarded", "challenged", "blocked")


def test_chain_audit_root_is_64_hex_chars():
    trace = run_federation_chain(initial_payload=_payload())
    assert len(trace.audit_root) == 64
    assert all(c in "0123456789abcdef" for c in trace.audit_root)


def test_chain_id_is_deterministic_for_payload():
    a = run_federation_chain(initial_payload=_payload())
    b = run_federation_chain(initial_payload=_payload())
    assert a.chain_id == b.chain_id
    assert a.audit_root == b.audit_root


def test_hop_audit_hashes_chain_to_audit_root():
    trace = run_federation_chain(initial_payload=_payload())
    for i in range(1, len(trace.hops)):
        assert trace.hops[i].parent_hash == trace.hops[i - 1].audit_hash


def test_chain_blocks_on_advocate_escalate():
    """Patient with no caregiver + discharge_with_homecare -> the
    advocate escalates -> chain blocks."""
    trace = run_federation_chain(initial_payload=_payload(
        lace=8,
        demographics={
            "age": 70, "race": "white", "insurance": "medicare",
            "language": "english", "n_chronic_meds": 5,
            "has_home_caregiver_available": False,
            "has_transportation": True,
            "fairness_audit_present": True,
        },
    ))
    assert trace.final_verdict == "blocked"
    assert "abstain" in trace.final_action.lower()


def test_chain_challenged_on_low_risk_continued_admission_via_high_lace():
    """Use a high-LACE bin so action is continued_admission and
    fairness flagged subgroup with audit triggers a challenge."""
    trace = run_federation_chain(initial_payload=_payload(
        lace=11,
        demographics={
            "age": 60, "race": "black", "insurance": "medicaid",
            "language": "english", "n_chronic_meds": 6,
            "has_home_caregiver_available": True,
            "has_transportation": True,
            "fairness_audit_present": True,
        },
    ))
    assert trace.final_verdict in ("challenged", "blocked")


def test_chain_blocks_when_fairness_audit_missing():
    trace = run_federation_chain(initial_payload=_payload(
        lace=9,
        demographics={
            "age": 60, "race": "black", "insurance": "medicaid",
            "language": "english", "n_chronic_meds": 6,
            "has_home_caregiver_available": True,
            "has_transportation": True,
            "fairness_audit_present": False,
        },
    ))
    assert trace.final_verdict == "blocked"


def test_round_trip_through_pydantic():
    trace = run_federation_chain(initial_payload=_payload())
    payload = trace.model_dump(mode="json")
    rebuilt = FederationChainTrace.model_validate(payload)
    assert rebuilt.audit_root == trace.audit_root
