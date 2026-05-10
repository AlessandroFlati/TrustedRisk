"""Phase 17.U - Federation chained-call orchestrator.

Demonstrates a real A2A specialist-to-specialist call chain:

    discharge-specialist -> pa-agent -> patient-advocate

Each step ingests the prior step's output and contributes its own
audit record. The full chain is covered by a single Merkle-style
hash chain so any tampering surfaces.

Pure-Python deterministic. No HTTP - the simulator routes calls in-
process. Production swaps the registry for an httpx-based A2A client.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from .multi_agent_debate import DebateInput, run_debate
from .patient_advocate import (
    PatientAdvocateInput, evaluate_patient_advocate,
)


_HopVerdict = Literal["forwarded", "challenged", "blocked"]


class ChainHop(BaseModel):
    hop_index: int
    specialist_id: str
    incoming_summary: str
    output_summary: str
    verdict: _HopVerdict
    latency_ms: float
    audit_hash: str
    parent_hash: str


class FederationChainTrace(BaseModel):
    chain_id: str
    n_hops: int = Field(ge=0)
    hops: list[ChainHop]
    final_action: str
    final_verdict: _HopVerdict
    audit_root: str
    rationale: str


def _hash_record(content: dict[str, Any], parent: str) -> str:
    s = json.dumps(content, sort_keys=True) + parent
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────
# Synthetic specialist callables (deterministic floor for the demo)
# ─────────────────────────────────────────────────────────────────────


def _hop_discharge(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], _HopVerdict, str]:
    """Discharge specialist: emits a candidate DecisionCard from the
    LACE-derived risk + proposed action."""
    lace = int(payload.get("lace", 6))
    risk = (
        0.072 if lace <= 2
        else 0.103 if lace <= 5
        else 0.158 if lace <= 9
        else 0.234 if lace <= 12
        else 0.327
    )
    if risk < 0.10:
        action = "discharge_home"
    elif risk < 0.20:
        action = "discharge_with_homecare"
    else:
        action = "continued_admission"
    out = {
        "candidate_decision_card": {
            "patient_id": payload.get("patient_id", "p1"),
            "encounter_id": payload.get("encounter_id", "e1"),
            "recommended_action": action,
            "risk_point_estimate": risk,
            "risk_ci_width": 0.05,
            "lace": lace,
        },
        "demographics": payload.get("demographics", {}),
    }
    return out, "forwarded", (
        f"discharge specialist computed risk {risk:.3f} -> action "
        f"`{action}` (LACE {lace})."
    )


def _hop_pa_agent(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], _HopVerdict, str]:
    """PA agent: confirms whether the proposed disposition is
    payer-coverable + emits a structured pre-auth request when
    needed."""
    card = payload["candidate_decision_card"]
    action = card["recommended_action"]
    pa_required = action in ("discharge_with_homecare",)
    out = {
        **payload,
        "pa_decision": {
            "service_type": (
                "home_health_setup" if pa_required else "no_pa"
            ),
            "pa_required": pa_required,
            "estimated_approval_likelihood": (
                0.78 if pa_required else 1.0
            ),
        },
    }
    summary = (
        f"PA agent: {'pre-auth path opened' if pa_required else 'no PA needed'} "
        f"for action `{action}`."
    )
    return out, "forwarded", summary


def _hop_patient_advocate(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], _HopVerdict, str]:
    """Patient advocate: 5-axis review + final verdict."""
    card = payload["candidate_decision_card"]
    dem = payload.get("demographics", {})
    advocate = evaluate_patient_advocate(PatientAdvocateInput(
        patient_age=int(dem.get("age", 65)),
        patient_race=dem.get("race"),
        patient_insurance=dem.get("insurance"),
        patient_language=dem.get("language", "english"),
        n_chronic_medications=int(dem.get("n_chronic_meds", 5)),
        has_home_caregiver_available=bool(
            dem.get("has_home_caregiver_available", True)
        ),
        has_transportation=bool(
            dem.get("has_transportation", True)
        ),
        fairness_audit_present=bool(
            dem.get("fairness_audit_present", True)
        ),
        recommended_action=card["recommended_action"],
        risk_point_estimate=card["risk_point_estimate"],
    ))
    debate = run_debate(DebateInput(
        recommended_action=card["recommended_action"],
        risk_point_estimate=card["risk_point_estimate"],
        risk_ci_width=card["risk_ci_width"],
        fairness_subgroup=(
            (dem.get("race") or "").lower()
            if (dem.get("race") or "").lower() in (
                "black", "indigenous"
            )
            else (dem.get("insurance") or "").lower()
            if (dem.get("insurance") or "").lower() in (
                "medicaid", "uninsured"
            )
            else None
        ),
        fairness_audit_present=bool(
            dem.get("fairness_audit_present", True)
        ),
    ))
    if (
        advocate.overall_verdict == "escalate"
        or debate.verdict == "force_abstain"
    ):
        verdict: _HopVerdict = "blocked"
    elif (
        advocate.overall_verdict == "challenge"
        or debate.verdict == "revise"
    ):
        verdict = "challenged"
    else:
        verdict = "forwarded"
    out = {
        **payload,
        "advocate_outcome": advocate.model_dump(mode="json"),
        "debate_outcome": debate.model_dump(mode="json"),
    }
    summary = (
        f"patient advocate: {advocate.overall_verdict}; debate: "
        f"{debate.verdict}."
    )
    return out, verdict, summary


_DEFAULT_PIPELINE: list[
    tuple[str, Callable[[dict[str, Any]],
                         tuple[dict[str, Any], _HopVerdict, str]]]
] = [
    ("trustedrisk-discharge", _hop_discharge),
    ("trustedrisk-pa", _hop_pa_agent),
    ("trustedrisk-patient-advocate", _hop_patient_advocate),
]


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def run_federation_chain(
    *,
    initial_payload: dict[str, Any],
    chain_id: str | None = None,
) -> FederationChainTrace:
    """Run the deterministic federation chain over the initial
    payload. Returns a trace with per-hop audit hashes, the chain
    Merkle root, and the final verdict.
    """
    if chain_id is None:
        chain_id = (
            "chain-"
            + hashlib.sha256(
                json.dumps(initial_payload, sort_keys=True)
                .encode("utf-8")
            ).hexdigest()[:16]
        )
    parent = "0" * 64
    payload = dict(initial_payload)
    hops: list[ChainHop] = []
    final_verdict: _HopVerdict = "forwarded"
    for idx, (specialist_id, fn) in enumerate(_DEFAULT_PIPELINE):
        t0 = time.perf_counter()
        new_payload, verdict, summary = fn(payload)
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        record = {
            "hop_index": idx,
            "specialist_id": specialist_id,
            "summary": summary,
            "verdict": verdict,
        }
        audit_hash = _hash_record(record, parent)
        hops.append(ChainHop(
            hop_index=idx,
            specialist_id=specialist_id,
            incoming_summary=str(payload.get("__last_summary__", "init")),
            output_summary=summary,
            verdict=verdict,
            latency_ms=latency_ms,
            audit_hash=audit_hash,
            parent_hash=parent,
        ))
        new_payload["__last_summary__"] = summary
        payload = new_payload
        parent = audit_hash
        if verdict == "blocked":
            final_verdict = "blocked"
            break
        if verdict == "challenged":
            final_verdict = "challenged"

    final_action = (
        payload.get("candidate_decision_card", {})
        .get("recommended_action", "unknown")
    )
    if final_verdict == "blocked":
        final_action = "abstain_pending_advocate_escalation"

    return FederationChainTrace(
        chain_id=chain_id,
        n_hops=len(hops),
        hops=hops,
        final_action=final_action,
        final_verdict=final_verdict,
        audit_root=parent,
        rationale=(
            f"federation chain over {len(_DEFAULT_PIPELINE)} "
            f"specialists; final verdict `{final_verdict}`; "
            f"chain audit-root {parent[:16]}..."
        ),
    )
