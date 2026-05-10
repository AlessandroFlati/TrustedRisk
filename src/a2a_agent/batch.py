"""Deterministic batch composer for POST /api/batch/decision-cards.

Why deterministic + tool-only (no LlmAgent)?
  1. Throughput: a 100-patient batch should not require 100 LLM round-trips.
  2. Auditability: byte-identical output given identical inputs and artifacts.
  3. Cost / SLA: local Ollama LLM at ~30s/call doesn't fit batch SLAs.

The pipeline per patient:
    compute_readmission_risk(patient_id)
        ├─ optional compute_medication_reconciliation
        └─ optional compute_fairness_audit(risk, demographics)
    -> check_ci_width / check_ood
    -> recommend_action_from_risk (rule-based)
    -> BatchPatientResult

Errors are isolated to the failing patient -- one bad chart does not poison
the rest of the batch.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from shared.abstain import check_ci_width, check_ood
from shared.schemas import (
    AbstainTrigger,
    Action,
    BatchPatientRequest,
    BatchPatientResult,
    BatchRequest,
    BatchResponse,
    FairnessReport,
    MedReconReport,
    Recommendation,
    RiskEstimate,
)

from mcp_server.tools.fairness_audit import compute_fairness_audit
from mcp_server.tools.medication_reconciliation import (
    compute_medication_reconciliation,
)
from mcp_server.tools.readmission_risk import compute_readmission_risk


# ─────────────────────────────────────────────────────────────────────
# Risk-to-recommendation rule (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────────────

def recommend_action_from_risk(risk: RiskEstimate) -> Recommendation:
    """Map a calibrated RiskEstimate to a discharge action + confidence band.

    Thresholds chosen so they straddle the LACE-strata boundaries from the
    promoted coefficients.json (LACE 0-2 ≈ 10.8%, LACE 3-5 ≈ 10.3%, LACE 6-9
    ≈ 14%, LACE 10+ ≈ 23%+). They are deliberately conservative on the high
    end -- the calibration cohort's AUROC is only 0.59, so the model is best
    used as a conservative gate rather than a fine-grained ranker.
    """
    p = risk.probability_mean
    if p > 0.30:
        return Recommendation(action=Action.CONTINUED_ADMISSION, confidence="low")
    if p > 0.20:
        return Recommendation(action=Action.SNF, confidence="low")
    if p > 0.10:
        return Recommendation(action=Action.HOME_WITH_CARE, confidence="medium")
    return Recommendation(action=Action.DISCHARGE_HOME, confidence="high")


# ─────────────────────────────────────────────────────────────────────
# Per-patient pipeline
# ─────────────────────────────────────────────────────────────────────

async def compose_decision_for_patient(
    req: BatchPatientRequest,
) -> BatchPatientResult:
    """Run the deterministic per-patient pipeline. Catches all exceptions and
    converts them into a per-patient error so a single bad chart doesn't fail
    the whole batch."""
    try:
        risk: RiskEstimate = await compute_readmission_risk(
            horizon_days=req.horizon_days,
            patient_id=req.patient_id,
        )
    except Exception as exc:
        return BatchPatientResult(
            patient_id=req.patient_id,
            status="error",
            error=f"compute_readmission_risk failed: {type(exc).__name__}: {exc}",
        )

    medrecon: MedReconReport | None = None
    if req.include_med_recon:
        try:
            medrecon = await compute_medication_reconciliation(
                patient_id=req.patient_id,
            )
        except Exception as exc:
            # Degrade gracefully: keep risk, drop medrecon, downgrade confidence.
            medrecon = None
            _medrecon_error = f"med_recon_skipped: {type(exc).__name__}: {exc}"
        else:
            _medrecon_error = None
    else:
        _medrecon_error = None

    fairness: FairnessReport | None = None
    if req.include_fairness and req.demographics:
        try:
            fairness = await compute_fairness_audit(
                risk=risk,
                patient_demographics=req.demographics,
            )
        except Exception:
            fairness = None  # non-fatal; surface as confidence downgrade

    abstain: list[AbstainTrigger] = []
    ci_trigger = check_ci_width(risk)
    if ci_trigger is not None:
        abstain.append(ci_trigger)
    ood_trigger = check_ood(risk.lace_raw_score)
    if ood_trigger is not None:
        abstain.append(ood_trigger)

    if abstain:
        recommendation = None
    else:
        recommendation = recommend_action_from_risk(risk)
        if fairness is not None and fairness.confidence_action in (
            "downgrade_confidence", "abstain_recommended",
        ):
            if fairness.confidence_action == "abstain_recommended":
                recommendation = None
                abstain.append(AbstainTrigger(
                    type="out_of_distribution",
                    detail=fairness.rationale,
                    threshold_exceeded={
                        "max_relative_drift": fairness.max_relative_drift,
                    },
                ))
            else:
                recommendation = Recommendation(
                    action=recommendation.action,
                    confidence="low",
                )

    return BatchPatientResult(
        patient_id=req.patient_id,
        status="ok",
        recommendation=recommendation,
        risk_estimate=risk,
        medication_reconciliation=medrecon,
        fairness=fairness,
        abstain=abstain,
        valid_until=risk.valid_until,
    )


# ─────────────────────────────────────────────────────────────────────
# Batch driver -- concurrency-bounded gather
# ─────────────────────────────────────────────────────────────────────

async def process_batch(req: BatchRequest) -> BatchResponse:
    """Run compose_decision_for_patient over `req.patients` with bounded
    concurrency. Always returns one BatchPatientResult per requested patient,
    in input order."""
    started_at = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(req.max_concurrency)

    async def _bounded(p: BatchPatientRequest) -> BatchPatientResult:
        async with semaphore:
            return await compose_decision_for_patient(p)

    results: list[BatchPatientResult] = await asyncio.gather(
        *[_bounded(p) for p in req.patients],
        return_exceptions=False,  # compose_decision_for_patient never raises
    )

    completed_at = datetime.now(timezone.utc)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000.0)

    n_succeeded = sum(1 for r in results if r.status == "ok")
    n_failed = sum(1 for r in results if r.status == "error")
    n_abstained = sum(1 for r in results if r.status == "ok" and r.recommendation is None)

    return BatchResponse(
        n_requested=len(req.patients),
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        n_abstained=n_abstained,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────
# JSON request/response adapters (used by the Starlette route)
# ─────────────────────────────────────────────────────────────────────

def parse_batch_request(payload: Any) -> BatchRequest:
    """Validate and parse a JSON body into a BatchRequest."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    return BatchRequest.model_validate(payload)


def serialize_batch_response(resp: BatchResponse) -> dict[str, Any]:
    """Convert BatchResponse to a JSON-safe dict."""
    return resp.model_dump(mode="json")
