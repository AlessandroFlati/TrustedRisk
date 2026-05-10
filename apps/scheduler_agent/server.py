"""COMPOSE-2 — TrustedRisk scheduler agent (federated A2A partner).

The scheduler agent transforms a TrustedRisk DecisionCard into a concrete
follow-up visit plan. It plays the role of a specialist scheduling agent so
TrustedRisk itself doesn't have to own scheduling rules + EHR integration.

Inputs:
  POST /a2a/skill/propose_followup_visits
    body = {"decision_card": <DecisionCard dict>,
            "discharge_date": "YYYY-MM-DD"  (optional; default = today UTC)}

Outputs: list of FollowupVisit dicts (visit type, proposed date, modality,
priority, rationale).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
        .venv/Scripts/python.exe -m uvicorn \
        apps.scheduler_agent.server:app --port 8769
"""

from __future__ import annotations

import json
import sys
import uuid
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


app = FastAPI(title="trustedrisk-scheduler-agent (COMPOSE-2)", version="0.1.0")

_AGENT_CARD_PATH = Path(__file__).parent / "agent_card.json"

_RECENT_PROPOSALS: Deque[dict[str, Any]] = deque(maxlen=200)


def _reset_state() -> None:
    """Test helper."""
    _RECENT_PROPOSALS.clear()


# ─────────────────────── Risk-tier policy ───────────────────────
#
# Visit timing follows the AHRQ Re-Engineered Discharge (RED) bundle and
# Hansen NEJM 2011 evidence: high-risk patients need a 7-day touchpoint;
# moderate-risk benefit from 7–14 day follow-up; routine can be 14–30 day.

def _risk_tier(probability_mean: float | None,
                  confidence: str | None) -> str:
    """Map (calibrated risk, confidence) to a follow-up tier."""
    p = probability_mean or 0.0
    if p >= 0.30:
        return "high"
    if p >= 0.15:
        return "moderate"
    if confidence == "low":
        # Low confidence in a low-risk recommendation still warrants
        # an earlier check-in (deference to clinician).
        return "moderate"
    return "routine"


_TIER_TIMING_DAYS = {
    "high": (3, 7),
    "moderate": (7, 14),
    "routine": (14, 30),
}


def _propose_pcp_visit(tier: str, discharge: date) -> dict[str, Any]:
    lo, hi = _TIER_TIMING_DAYS[tier]
    proposed = discharge + timedelta(days=lo)
    deadline = discharge + timedelta(days=hi)
    return {
        "visit_id": uuid.uuid4().hex,
        "visit_type": "primary_care",
        "specialty": None,
        "proposed_date": proposed.isoformat(),
        "deadline_date": deadline.isoformat(),
        "modality": "in_person",
        "priority": tier,
        "rationale": (
            f"AHRQ RED bundle + Hansen NEJM 2011: post-discharge PCP visit "
            f"within {lo}-{hi} days for {tier}-risk discharges to reduce "
            f"30-day readmission."
        ),
    }


# ─────────────────────── Specialist inference from tool trace ───────────────────────

_TOOL_TO_SPECIALTY: dict[str, tuple[str, str]] = {
    "compute_aki_kdigo_stage":
        ("nephrology", "AKI requires nephrology follow-up to confirm renal recovery."),
    "compute_dialysis_initiation_decision":
        ("nephrology", "Dialysis decision requires close nephrology follow-up."),
    "compute_dka_severity":
        ("endocrinology", "Post-DKA endocrinology visit for insulin titration "
                            "and complication screening."),
    "compute_inpatient_glycemic_control":
        ("endocrinology", "Inpatient glycemic adjustments warrant endocrine review."),
    "compute_heart_score":
        ("cardiology", "Recent ACS workup — outpatient cardiology stratification."),
    "compute_acs_disposition_decision":
        ("cardiology", "Recent ACS — cardiology follow-up within 7-14 days."),
    "compute_stroke_severity":
        ("neurology", "Post-stroke neurology follow-up for residual deficits."),
    "compute_stroke_thrombolysis_eligibility":
        ("neurology", "Post-thrombolysis neurology follow-up."),
    "compute_oncology_treatment_response":
        ("oncology", "RECIST follow-up imaging cycle scheduled with oncology."),
    "compute_chemo_dose_adjustment":
        ("oncology", "Chemo cycle scheduled with oncology nurse navigator."),
    "compute_suicide_risk_assessment":
        ("mental_health", "Suicide risk warrants short-interval mental health "
                            "contact (Joint Commission NPSG 15)."),
    "compute_psychiatric_admission_decision":
        ("mental_health", "Psychiatric follow-up to maintain treatment plan continuity."),
    "compute_preeclampsia_assessment":
        ("obstetrics", "Postpartum preeclampsia visit within 3-7 days for BP check."),
    "compute_maternal_early_warning":
        ("obstetrics", "Postpartum obstetric follow-up for residual MEOWS triggers."),
    "compute_falls_risk_morse":
        ("geriatrics", "Falls-risk follow-up — home safety + medication review."),
    "compute_delirium_screening_cam":
        ("geriatrics", "Post-delirium cognitive follow-up."),
    "compute_empiric_antibiotic_selection":
        ("infectious_disease", "Antibiotic stewardship follow-up at completion of course."),
    "compute_antibiotic_de_escalation":
        ("infectious_disease", "ID review at end of antimicrobial course."),
}


def _specialist_visits(card: dict[str, Any], tier: str,
                          discharge: date) -> list[dict[str, Any]]:
    audit = card.get("audit") or {}
    tools = audit.get("tools") or audit.get("tool_trace") or []
    seen: set[str] = set()
    visits: list[dict[str, Any]] = []
    # Specialist visits go a touch later than PCP — their schedules are tighter.
    base_lo, base_hi = _TIER_TIMING_DAYS[tier]
    spec_lo = max(base_lo, 7)
    spec_hi = max(base_hi, 14)

    for entry in tools:
        if isinstance(entry, dict):
            tool_name = entry.get("tool") or entry.get("name") or ""
        elif isinstance(entry, str):
            tool_name = entry
        else:
            continue

        spec = _TOOL_TO_SPECIALTY.get(tool_name)
        if spec is None:
            continue
        specialty, rationale = spec
        if specialty in seen:
            continue
        seen.add(specialty)

        proposed = discharge + timedelta(days=spec_lo)
        deadline = discharge + timedelta(days=spec_hi)
        modality = "telehealth" if specialty == "mental_health" else "in_person"
        visits.append({
            "visit_id": uuid.uuid4().hex,
            "visit_type": "specialist",
            "specialty": specialty,
            "proposed_date": proposed.isoformat(),
            "deadline_date": deadline.isoformat(),
            "modality": modality,
            "priority": tier,
            "rationale": rationale,
        })
    return visits


# ─────────────────────── Card extraction helpers ───────────────────────

def _extract_action(card: dict[str, Any]) -> str | None:
    rec = card.get("recommendation") or {}
    return rec.get("action") if isinstance(rec, dict) else None


def _extract_confidence(card: dict[str, Any]) -> str | None:
    rec = card.get("recommendation") or {}
    return rec.get("confidence") if isinstance(rec, dict) else None


def _extract_risk(card: dict[str, Any]) -> float | None:
    reasoning = card.get("reasoning") or {}
    risk = reasoning.get("risk_estimate") or {}
    p = risk.get("probability_mean")
    if isinstance(p, (int, float)):
        return float(p)
    return None


def _extract_request_id(card: dict[str, Any]) -> str | None:
    audit = card.get("audit") or {}
    rid = audit.get("request_id")
    return rid if isinstance(rid, str) else None


# ─────────────────────── Routes ───────────────────────

@app.get("/.well-known/agent-card.json")
async def agent_card() -> JSONResponse:
    return JSONResponse(content=json.loads(
        _AGENT_CARD_PATH.read_text(encoding="utf-8")))


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(content={
        "status": "ok",
        "recent_proposals": len(_RECENT_PROPOSALS),
    })


@app.post("/a2a/skill/propose_followup_visits")
async def propose_followup_visits(req: Request) -> JSONResponse:
    body = await req.json()
    card = body.get("decision_card")
    if not isinstance(card, dict):
        raise HTTPException(status_code=400,
                              detail="decision_card must be a DecisionCard dict")

    action = _extract_action(card)
    if action is None:
        return JSONResponse(content={
            "plan_id": uuid.uuid4().hex,
            "visits": [],
            "summary": "DecisionCard has no recommendation (abstained); "
                          "no follow-up proposed.",
            "request_id": _extract_request_id(card),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })

    discharge_str = body.get("discharge_date")
    if discharge_str:
        try:
            discharge = date.fromisoformat(discharge_str)
        except ValueError as e:
            raise HTTPException(status_code=400,
                                  detail=f"discharge_date must be ISO-8601: {e}")
    else:
        discharge = datetime.now(timezone.utc).date()

    risk = _extract_risk(card)
    confidence = _extract_confidence(card)
    tier = _risk_tier(risk, confidence)

    visits: list[dict[str, Any]] = []
    if action in ("home_with_care", "discharge_home"):
        # Both home dispositions get a PCP visit. continued_admission and
        # snf are handled by their respective transition coordinators —
        # this scheduler covers outpatient follow-up only.
        visits.append(_propose_pcp_visit(tier, discharge))
    elif action == "snf":
        # SNF transition — single short-interval interface visit
        proposed = discharge + timedelta(days=7)
        visits.append({
            "visit_id": uuid.uuid4().hex,
            "visit_type": "transition_visit",
            "specialty": None,
            "proposed_date": proposed.isoformat(),
            "deadline_date": (discharge + timedelta(days=14)).isoformat(),
            "modality": "in_person",
            "priority": tier,
            "rationale": "SNF disposition — facility transition visit at 7d "
                            "for medication reconciliation + therapy goals.",
        })
    elif action == "continued_admission":
        # Inpatient: no outpatient follow-up needed yet
        visits = []

    visits.extend(_specialist_visits(card, tier, discharge))

    plan = {
        "plan_id": uuid.uuid4().hex,
        "request_id": _extract_request_id(card),
        "discharge_date": discharge.isoformat(),
        "risk_tier": tier,
        "recommendation_action": action,
        "recommendation_confidence": confidence,
        "visits": visits,
        "summary": (
            f"{len(visits)} follow-up visit(s) proposed for {action} "
            f"with {tier} risk tier."
        ),
        "references": [
            "Jack BW, et al. A reengineered hospital discharge program to "
            "decrease rehospitalization: a randomized trial. Ann Intern Med. "
            "2009;150:178-187. (AHRQ RED bundle)",
            "Hansen LO, et al. Interventions to reduce 30-day rehospitalization: "
            "a systematic review. Ann Intern Med. 2011;155:520.",
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _RECENT_PROPOSALS.appendleft(plan)
    return JSONResponse(content=plan)


@app.get("/a2a/skill/list_recent_proposals")
async def list_recent_proposals(limit: int = 20) -> JSONResponse:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400,
                              detail="limit must be in [1, 200]")
    items = list(_RECENT_PROPOSALS)[:limit]
    return JSONResponse(content={"proposals": items, "n": len(items)})


# ─────────────────────── UI ───────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    n = len(_RECENT_PROPOSALS)
    return f"""<!DOCTYPE html>
<html><head><title>trustedrisk-scheduler-agent</title>
<style>body{{font-family:system-ui;margin:32px;max-width:800px;color:#222}}
code{{background:#f3f4f6;padding:2px 6px;border-radius:4px}}</style></head>
<body>
<h1>trustedrisk-scheduler-agent</h1>
<p>COMPOSE-2 federated A2A partner. Recent proposals buffered: <code>{n}</code></p>
<ul>
  <li>POST <code>/a2a/skill/propose_followup_visits</code>
      {{decision_card, discharge_date?}}</li>
  <li>GET <code>/a2a/skill/list_recent_proposals</code></li>
  <li>GET <code>/.well-known/agent-card.json</code></li>
</ul>
</body></html>"""
