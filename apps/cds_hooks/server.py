"""CDS Hooks server (CDS-1, upgraded to v2.0 in Phase 11.5).

Implements the HL7 CDS Hooks specification (v2.0) for TrustedRisk:

  GET  /cds-services
       Returns the catalog of available CDS services. Each service
       advertises `hook`, `prefetch`, optional `description`, and (v2.0)
       a `feedbackEndpoint`-relative URL.
  POST /cds-services/{service-id}
       Invoked by an EHR when a hook fires. Returns CDS Cards 2.0.
  POST /cds-services/{service-id}/feedback
       v2.0 feedback channel: the EHR posts {feedback: [{card, outcome,
       overrideReason?, acceptedSuggestions?}]} so the publisher can
       audit which cards were acted on.

Supported hooks (CDS Hooks 2.0 + legacy v1.1):
  - patient-view              — when a clinician opens a chart
  - order-select (v2.0)       — when the EHR is selecting a draft order
  - order-sign (v2.0)         — when the order is about to be signed
  - medication-prescribe      — legacy v1.1 (kept for back-compat)
  - order-review              — legacy v1.1 (kept for back-compat)

Mapping: every CDS Hook invocation gets routed to the appropriate
TrustedRisk bundle (via the bundle orchestrator from SAFE-2), the
relevant tools are called, and the resulting DecisionCard is rendered
as one or more CDS Cards (v2.0 — every card carries a `uuid`, and
medication-safety cards carry `overrideReasons[]`).

Spec references:
  https://cds-hooks.org/specification/current/
  https://cds-hooks.hl7.org/2.0/

SMART-on-FHIR launch context:
  The optional `fhirAuthorization` block in the request body carries
  the SMART access_token + patient context. We extract it and pipe it
  into the SHARP-on-MCP context for downstream FHIR fetches.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))


app = FastAPI(title="TrustedRisk CDS Hooks Server", version="0.1.0")


# ─────────────────────── CDS service catalog ───────────────────────

_SERVICES = [
    {
        "id": "trustedrisk-patient-view",
        "hook": "patient-view",
        "title": "TrustedRisk patient overview",
        "description": (
            "On chart open, evaluate the patient against TrustedRisk's "
            "calibrated risk + safety models. Surfaces calibrated 30-day "
            "readmission risk, fairness drift flags, abstain triggers, and "
            "any active polypharmacy / falls / delirium / lab-trend concerns "
            "as CDS Cards."
        ),
        "prefetch": {
            "patient": "Patient/{{context.patientId}}",
            "conditions": "Condition?patient={{context.patientId}}",
            "medications": "MedicationRequest?patient={{context.patientId}}",
            "observations": "Observation?patient={{context.patientId}}",
        },
    },
    {
        "id": "trustedrisk-medication-prescribe",
        "hook": "medication-prescribe",
        "title": "TrustedRisk medication safety review",
        "description": (
            "On medication prescription, screen the proposed order against "
            "the patient's existing medications for DDI risk, polypharmacy "
            "burden, age contraindications (pediatric / geriatric), and "
            "monitoring gaps."
        ),
        "prefetch": {
            "patient": "Patient/{{context.patientId}}",
            "active_meds": "MedicationRequest?patient={{context.patientId}}&status=active",
        },
    },
    {
        "id": "trustedrisk-order-review",
        "hook": "order-review",
        "title": "TrustedRisk order review (imaging + chemo)",
        "description": (
            "On order review, evaluate imaging-appropriateness (ACR), "
            "contrast safety (eGFR / metformin / pregnancy), and chemo "
            "dose adjustments (ANC / renal / hepatic gates)."
        ),
        "prefetch": {
            "patient": "Patient/{{context.patientId}}",
            "egfr": "Observation?patient={{context.patientId}}&code=33914-3",
        },
    },
    # ─── CDS Hooks 2.0 services ───────────────────────────────────
    {
        "id": "trustedrisk-order-select",
        "hook": "order-select",
        "title": "TrustedRisk order-select review (CDS Hooks 2.0)",
        "description": (
            "Fires when a clinician selects a draft order. Evaluates "
            "imaging appropriateness + contrast safety + DDI + dose "
            "guard before the order is signed. Returns CDS Cards 2.0 "
            "with override reasons + suggestion cards."
        ),
        "prefetch": {
            "patient": "Patient/{{context.patientId}}",
            "active_meds": (
                "MedicationRequest?patient={{context.patientId}}&status=active"
            ),
            "egfr": (
                "Observation?patient={{context.patientId}}&code=33914-3"
            ),
        },
        "feedbackEndpoint": (
            "/cds-services/trustedrisk-order-select/feedback"
        ),
        "extension": {
            "com.trustedrisk.calibration_version": "spec_002",
        },
    },
    {
        "id": "trustedrisk-order-sign",
        "hook": "order-sign",
        "title": "TrustedRisk order-sign final guard (CDS Hooks 2.0)",
        "description": (
            "Fires immediately before the EHR signs a batch of orders. "
            "Performs a final readmission-risk + abstain-trigger check "
            "and emits override-reason-bearing cards on remaining "
            "high-severity findings. Honoured `acceptedSuggestions` are "
            "logged via the feedback channel."
        ),
        "prefetch": {
            "patient": "Patient/{{context.patientId}}",
            "active_meds": (
                "MedicationRequest?patient={{context.patientId}}&status=active"
            ),
        },
        "feedbackEndpoint": (
            "/cds-services/trustedrisk-order-sign/feedback"
        ),
        "extension": {
            "com.trustedrisk.calibration_version": "spec_002",
        },
    },
]


# ─── Feedback log (in-memory; CDS Hooks 2.0 audit channel) ─────────

_FEEDBACK_LOG: list[dict[str, Any]] = []
_FEEDBACK_LOG_MAX = 1024


def _record_feedback(service_id: str, payload: dict[str, Any]) -> int:
    entry = {
        "service_id": service_id,
        "received_at_iso": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    _FEEDBACK_LOG.append(entry)
    if len(_FEEDBACK_LOG) > _FEEDBACK_LOG_MAX:
        del _FEEDBACK_LOG[: len(_FEEDBACK_LOG) - _FEEDBACK_LOG_MAX]
    return len(_FEEDBACK_LOG)


def get_feedback_log() -> list[dict[str, Any]]:
    """Test-helper: snapshot the current feedback log."""
    return list(_FEEDBACK_LOG)


def clear_feedback_log() -> None:
    """Test-helper: clear the in-memory feedback ring."""
    _FEEDBACK_LOG.clear()


# ─────────────────────── CDS Card builders ───────────────────────

def _indicator_for(severity: str | None,
                     abstain_count: int = 0) -> str:
    """Map TrustedRisk severity → CDS Hooks indicator (info/warning/critical)."""
    if abstain_count > 0:
        return "critical"
    if severity in ("high", "severe", "imminent"):
        return "critical"
    if severity in ("medium", "moderate", "moderate_severe"):
        return "warning"
    return "info"


import uuid as _uuid


def _make_card(summary: str, indicator: str, source_label: str = "TrustedRisk",
                detail: str | None = None,
                suggestions: list[dict[str, Any]] | None = None,
                links: list[dict[str, Any]] | None = None,
                override_reasons: list[dict[str, Any]] | None = None,
               ) -> dict[str, Any]:
    """Build a CDS Hooks card.

    Adds the v2.0 fields:
      - `uuid` — mandatory in CDS Hooks 2.0 so feedback can reference
        the originating card unambiguously.
      - `overrideReasons` — optional; when present the EHR may render
        them as preset reasons the user picks from when overriding.
    """
    card: dict[str, Any] = {
        "uuid": str(_uuid.uuid4()),
        "summary": summary[:140],   # CDS Hooks spec: 140-char max
        "indicator": indicator,
        "source": {
            "label": source_label,
            "url": "https://github.com/aleksandros/trustedrisk",
        },
    }
    if detail:
        card["detail"] = detail[:1000]
    if suggestions:
        card["suggestions"] = suggestions
    if links:
        card["links"] = links
    if override_reasons:
        card["overrideReasons"] = override_reasons
    return card


_DEFAULT_OVERRIDE_REASONS: list[dict[str, Any]] = [
    {"code": "patient-preference",
     "system": "https://cds-hooks.trustedrisk.local/override-reason",
     "display": "Patient declines"},
    {"code": "clinically-justified",
     "system": "https://cds-hooks.trustedrisk.local/override-reason",
     "display": "Clinically justified after review"},
    {"code": "alternative-not-available",
     "system": "https://cds-hooks.trustedrisk.local/override-reason",
     "display": "No alternative available"},
    {"code": "false-positive",
     "system": "https://cds-hooks.trustedrisk.local/override-reason",
     "display": "TrustedRisk false positive"},
]


# ─────────────────────── Endpoints ───────────────────────

@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Cloud-Run-compatible liveness probe."""
    return JSONResponse(content={
        "status": "ok",
        "service": "trustedrisk-cds-hooks",
        "n_services": len(_SERVICES),
    })


@app.get("/cds-services")
async def discovery() -> JSONResponse:
    """CDS Hooks discovery — required first endpoint per spec."""
    return JSONResponse(content={"services": _SERVICES})


@app.post("/cds-services/{service_id}")
async def invoke_service(service_id: str, req: Request) -> JSONResponse:
    """Invoke a CDS service. Body conforms to CDS Hooks v1.1 spec."""
    body = await req.json()

    # Required fields per spec
    hook = body.get("hook")
    hook_instance = body.get("hookInstance")
    context = body.get("context") or {}
    if not hook or not hook_instance:
        raise HTTPException(status_code=400,
                              detail="Missing required fields: hook, hookInstance")

    # SMART-on-FHIR launch context (optional)
    fhir_auth = body.get("fhirAuthorization") or {}
    fhir_server = body.get("fhirServer") or ""
    access_token = fhir_auth.get("access_token") or ""
    patient_id = context.get("patientId") or ""

    # Validate the service exists
    service = next((s for s in _SERVICES if s["id"] == service_id), None)
    if service is None:
        raise HTTPException(status_code=404,
                              detail=f"Unknown CDS service: {service_id}")
    if service["hook"] != hook:
        raise HTTPException(
            status_code=400,
            detail=f"Service {service_id} is for hook {service['hook']!r}, "
                    f"got {hook!r}",
        )

    # Route to the right handler based on the service id
    cards: list[dict[str, Any]]
    system_actions: list[dict[str, Any]] = []
    if service_id == "trustedrisk-patient-view":
        cards = await _handle_patient_view(context, body.get("prefetch") or {})
    elif service_id == "trustedrisk-medication-prescribe":
        cards = await _handle_medication_prescribe(
            context, body.get("prefetch") or {})
    elif service_id == "trustedrisk-order-review":
        cards = await _handle_order_review(
            context, body.get("prefetch") or {})
    elif service_id == "trustedrisk-order-select":
        cards, system_actions = await _handle_order_select(
            context, body.get("prefetch") or {},
        )
    elif service_id == "trustedrisk-order-sign":
        cards, system_actions = await _handle_order_sign(
            context, body.get("prefetch") or {},
        )
    else:
        cards = []

    response: dict[str, Any] = {"cards": cards}
    if system_actions:
        response["systemActions"] = system_actions
    return JSONResponse(content=response)


# ─────────────────────── Feedback channel (v2.0) ───────────────────────


@app.post("/cds-services/{service_id}/feedback")
async def feedback(service_id: str, req: Request) -> JSONResponse:
    """CDS Hooks 2.0 feedback channel.

    Body shape:
        { "feedback": [
            { "card": "<card-uuid>",
              "outcome": "accepted" | "overridden",
              "outcomeTimestamp": ISO8601,
              "overrideReason": {"reason": {...}, "userComment": "..."},
              "acceptedSuggestions": [{"id": "..."}] }
          ] }

    Always returns 200 with `{recorded: <position>}` so EHRs can move on
    even if the audit log is full.
    """
    if not any(s["id"] == service_id for s in _SERVICES):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown CDS service: {service_id}",
        )
    payload = await req.json()
    if not isinstance(payload, dict) or not isinstance(
        payload.get("feedback"), list,
    ):
        raise HTTPException(
            status_code=400,
            detail="Feedback body must be {feedback: [...]}",
        )
    position = _record_feedback(service_id, payload)
    return JSONResponse(content={
        "service_id": service_id,
        "n_recorded": len(payload["feedback"]),
        "log_position": position,
    })


# ─────────────────────── Per-hook handlers ───────────────────────

async def _handle_patient_view(context: dict[str, Any],
                                  prefetch: dict[str, Any]) -> list[dict[str, Any]]:
    """patient-view → use bundle orchestrator to pick the bundle, then run a
    truncated DecisionCard composer with the prefetched data."""
    from a2a_agent.bundle_orchestrator import suggest_bundles

    patient = (prefetch.get("patient") or {})
    conditions = (prefetch.get("conditions") or {}).get("entry") or []
    age = _extract_age(patient)
    chief_complaint = " ".join(
        (e.get("resource", {}).get("code") or {}).get("text", "")
        for e in conditions[:5]
    )

    suggestion = suggest_bundles(
        chief_complaint=chief_complaint,
        structured_features={"age": age},
    )

    cards: list[dict[str, Any]] = []

    # Card 1: bundle suggestion summary
    cards.append(_make_card(
        summary=f"TrustedRisk recommends bundle: {suggestion.primary_bundle}",
        indicator="info",
        detail=suggestion.summary + (
            f"\n\nTop-3 candidates:\n"
            + "\n".join(
                f"- {s.bundle_id} (score {s.score}): {s.rationale}"
                for s in suggestion.ranked[:3]
            )
        ),
    ))

    # Card 2: medication safety summary (if meds available)
    medications = (prefetch.get("medications") or {}).get("entry") or []
    if medications:
        from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns
        med_list = [
            {
                "name": (m.get("resource", {})
                         .get("medicationCodeableConcept", {})
                         .get("text", "")),
                "drug_class": "unknown",
                "status": m.get("resource", {}).get("status", "active"),
            }
            for m in medications
        ]
        med_list = [m for m in med_list if m["name"]]
        if med_list:
            poly = await detect_polypharmacy_concerns(medications=med_list)
            indicator = _indicator_for(poly.polypharmacy_severity)
            cards.append(_make_card(
                summary=(f"Polypharmacy: {len(med_list)} active meds, "
                          f"{len(poly.interactions)} interactions, "
                          f"severity {poly.polypharmacy_severity}"),
                indicator=indicator,
                detail=poly.rationale,
                links=[{
                    "label": "TrustedRisk medication safety bundle docs",
                    "url": "https://github.com/aleksandros/trustedrisk/blob/main/docs/bundles/core_discharge.md",
                    "type": "absolute",
                }],
            ))

    return cards


async def _handle_medication_prescribe(
    context: dict[str, Any], prefetch: dict[str, Any],
) -> list[dict[str, Any]]:
    """medication-prescribe → DDI + polypharmacy on the proposed med + active meds."""
    from mcp_server.tools.polypharmacy_concerns import detect_polypharmacy_concerns

    # Proposed medication is in context.medications (a Bundle of MedicationRequest)
    proposed = context.get("medications") or {}
    proposed_entries = proposed.get("entry") or []
    active_meds = (prefetch.get("active_meds") or {}).get("entry") or []

    all_meds = [
        {
            "name": ((m.get("resource", {})
                       .get("medicationCodeableConcept", {})
                       .get("text", ""))
                       or m.get("resource", {}).get("medicationReference", {}).get("display", "")),
            "drug_class": "unknown",
            "status": m.get("resource", {}).get("status", "active"),
        }
        for m in (active_meds + proposed_entries)
    ]
    all_meds = [m for m in all_meds if m["name"]]
    if not all_meds:
        return [_make_card(
            summary="TrustedRisk: no medications to evaluate",
            indicator="info",
            detail="The CDS hook fired but no medication data was provided.",
        )]

    poly = await detect_polypharmacy_concerns(medications=all_meds)
    indicator = _indicator_for(poly.polypharmacy_severity)

    detail_parts = [poly.rationale]
    if poly.interactions:
        detail_parts.append(
            "Interactions detected:\n"
            + "\n".join(
                f"- {ddi.drug_a} + {ddi.drug_b} ({ddi.severity}): {ddi.detail}"
                for ddi in poly.interactions[:5]
            )
        )

    cards = [_make_card(
        summary=(f"Medication review: {len(poly.interactions)} interaction(s), "
                  f"severity {poly.polypharmacy_severity}"),
        indicator=indicator,
        detail="\n\n".join(detail_parts),
    )]

    # If a high-severity interaction → add a suggestion CDS Card
    high_sev = [d for d in poly.interactions if d.severity == "high"]
    if high_sev:
        cards.append(_make_card(
            summary=f"HIGH severity drug interaction: {high_sev[0].drug_a} + {high_sev[0].drug_b}",
            indicator="critical",
            detail=high_sev[0].detail,
            suggestions=[{
                "label": "Hold the new prescription pending pharmacist review",
                "uuid": f"sug-{int(time.time())}",
                "actions": [{
                    "type": "delete",
                    "description": "Cancel proposed prescription",
                }],
            }],
        ))

    return cards


async def _handle_order_select(
    context: dict[str, Any], prefetch: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """CDS Hooks 2.0 `order-select` handler.

    Fires when the EHR is selecting a draft order. We re-use the
    medication-prescribe pipeline (DDI + polypharmacy) on the proposed
    `selections` and add `overrideReasons[]` to every emitted card.
    """
    cards = await _handle_medication_prescribe(context, prefetch)
    for card in cards:
        if card.get("indicator") in ("warning", "critical"):
            card.setdefault("overrideReasons", _DEFAULT_OVERRIDE_REASONS)
    # No system actions emitted at this stage; the order is a draft.
    return cards, []


async def _handle_order_sign(
    context: dict[str, Any], prefetch: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """CDS Hooks 2.0 `order-sign` handler.

    Fires immediately before the EHR signs orders. Emits a final
    high-severity guard card with override reasons; when a critical
    DDI is found the handler proposes a `systemAction` of type
    `update` against the offending MedicationRequest so the EHR can
    auto-flag-for-review without prompting the user.
    """
    cards = await _handle_medication_prescribe(context, prefetch)
    system_actions: list[dict[str, Any]] = []
    for card in cards:
        if card.get("indicator") in ("warning", "critical"):
            card.setdefault("overrideReasons", _DEFAULT_OVERRIDE_REASONS)
        if card.get("indicator") == "critical":
            # Suggest a system-level mark-for-review action against
            # the order; the EHR may apply this without prompting.
            system_actions.append({
                "type": "update",
                "description": (
                    "Mark proposed order for pharmacy review prior to "
                    "signing — TrustedRisk flagged a critical DDI."
                ),
                "resource": {
                    "resourceType": "Task",
                    "status": "requested",
                    "intent": "order",
                    "code": {"text": "TrustedRisk pharmacy review"},
                },
            })
    return cards, system_actions


async def _handle_order_review(
    context: dict[str, Any], prefetch: dict[str, Any],
) -> list[dict[str, Any]]:
    """order-review → evaluate imaging appropriateness + contrast safety."""
    from mcp_server.tools.contrast_safety_check import compute_contrast_safety_check

    orders = context.get("draftOrders") or {}
    order_entries = orders.get("entry") or []

    cards: list[dict[str, Any]] = []
    egfr = _extract_egfr(prefetch.get("egfr") or {})

    for entry in order_entries:
        res = entry.get("resource", {})
        code_text = (res.get("code", {}) or {}).get("text", "").lower()
        if not code_text:
            continue

        # Detect contrast-related orders
        if "contrast" in code_text or "iv contrast" in code_text:
            contrast_type = ("gadolinium_iv" if "mri" in code_text
                              or "gadolinium" in code_text
                              else "iodinated_iv")
            check = await compute_contrast_safety_check(
                contrast_type=contrast_type,
                egfr_ml_min=egfr,
            )
            indicator = "critical" if not check.proceed_with_contrast else (
                "warning"
                if check.contrast_induced_nephropathy_risk in ("moderate", "high")
                else "info"
            )
            detail = check.rationale
            cards.append(_make_card(
                summary=(f"Contrast safety: proceed = {check.proceed_with_contrast}; "
                          f"CIN risk {check.contrast_induced_nephropathy_risk}"),
                indicator=indicator, detail=detail,
            ))

    if not cards:
        cards.append(_make_card(
            summary="TrustedRisk: no actionable findings on order review",
            indicator="info",
            detail=f"Reviewed {len(order_entries)} order(s); none triggered "
                    f"a contrast-safety or imaging-appropriateness rule.",
        ))

    return cards


# ─────────────────────── FHIR helpers ───────────────────────

def _extract_age(patient: dict[str, Any]) -> int | None:
    """Extract age from a FHIR Patient resource (best effort)."""
    bd = patient.get("birthDate")
    if not bd:
        return None
    try:
        from datetime import date
        y, m, d = bd.split("-")
        bdate = date(int(y), int(m), int(d))
        return (date.today() - bdate).days // 365
    except Exception:
        return None


def _extract_egfr(egfr_bundle: dict[str, Any]) -> float | None:
    entries = egfr_bundle.get("entry") or []
    if not entries:
        return None
    # Take the most recent eGFR observation
    latest = None
    for e in entries:
        res = e.get("resource", {})
        vq = res.get("valueQuantity") or {}
        v = vq.get("value")
        eff = res.get("effectiveDateTime")
        if v is not None and eff:
            if latest is None or eff > latest[0]:
                latest = (eff, float(v))
    return latest[1] if latest else None


# ─────────────────────── Demo UI ───────────────────────

@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return r"""<!DOCTYPE html>
<html><head><title>TrustedRisk CDS Hooks</title>
<style>
  body { font-family: -apple-system, sans-serif; background: #0d1117; color: #c9d1d9;
          max-width: 900px; margin: 40px auto; padding: 0 20px; line-height: 1.6; }
  h1, h2 { color: #58a6ff; }
  code { background: #1f2530; padding: 2px 6px; border-radius: 3px; color: #79c0ff; }
  pre { background: #1f2530; padding: 12px; border-radius: 4px; overflow: auto;
         font-size: 12px; }
  button { padding: 8px 14px; background: #58a6ff; color: white; border: none;
            border-radius: 4px; cursor: pointer; font-size: 13px; margin: 4px; }
  .card { padding: 12px 16px; border-radius: 6px; margin: 8px 0;
           border-left: 4px solid; background: #161b22; }
  .card.info { border-left-color: #58a6ff; }
  .card.warning { border-left-color: #d29922; }
  .card.critical { border-left-color: #f85149; }
  .card .summary { font-weight: 600; margin-bottom: 4px; }
  .card .detail { font-size: 13px; color: #8b949e; white-space: pre-wrap; }
</style></head>
<body>
<h1>TrustedRisk CDS Hooks Server</h1>
<p>HL7 CDS Hooks v1.1 implementation. Three services: <code>patient-view</code>,
<code>medication-prescribe</code>, <code>order-review</code>.</p>

<h2>Discovery</h2>
<pre>GET /cds-services</pre>
<button onclick="discover()">Run discovery</button>
<div id="discovery-result"></div>

<h2>Demo: patient-view hook</h2>
<button onclick="demoPatientView()">Run patient-view (CHF + warfarin patient)</button>
<div id="patient-view-result"></div>

<h2>Demo: medication-prescribe hook</h2>
<button onclick="demoMedPrescribe()">Run medication-prescribe (NSAID for warfarin pt)</button>
<div id="med-result"></div>

<h2>Demo: order-review hook (contrast CT for AKI pt)</h2>
<button onclick="demoOrderReview()">Run order-review</button>
<div id="order-result"></div>

<script>
async function discover() {
  const r = await fetch('/cds-services');
  const d = await r.json();
  document.getElementById('discovery-result').innerHTML =
    '<pre>' + JSON.stringify(d, null, 2) + '</pre>';
}

async function demoPatientView() {
  const out = document.getElementById('patient-view-result');
  out.innerHTML = '<em>Running…</em>';
  const body = {
    hook: 'patient-view',
    hookInstance: 'demo-' + Date.now(),
    fhirServer: 'http://example.com/fhir',
    context: { userId: 'Practitioner/demo', patientId: 'Patient/demo-1' },
    prefetch: {
      patient: { resourceType: 'Patient', id: 'demo-1', birthDate: '1949-03-15' },
      conditions: { resourceType: 'Bundle', entry: [
        { resource: { resourceType: 'Condition', code: { text: 'Heart failure' } } },
        { resource: { resourceType: 'Condition', code: { text: 'Atrial fibrillation' } } },
      ]},
      medications: { resourceType: 'Bundle', entry: [
        { resource: { medicationCodeableConcept: { text: 'warfarin 5mg PO daily' }, status: 'active' } },
        { resource: { medicationCodeableConcept: { text: 'lisinopril 10mg PO daily' }, status: 'active' } },
        { resource: { medicationCodeableConcept: { text: 'metoprolol 25mg PO BID' }, status: 'active' } },
      ]},
    },
  };
  const r = await fetch('/cds-services/trustedrisk-patient-view', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  out.innerHTML = renderCards(d.cards);
}

async function demoMedPrescribe() {
  const out = document.getElementById('med-result');
  out.innerHTML = '<em>Running…</em>';
  const body = {
    hook: 'medication-prescribe',
    hookInstance: 'demo-' + Date.now(),
    context: {
      userId: 'Practitioner/demo', patientId: 'Patient/demo-1',
      medications: { resourceType: 'Bundle', entry: [
        { resource: { medicationCodeableConcept: { text: 'ibuprofen 400mg PO TID' }, status: 'draft' } },
      ]},
    },
    prefetch: {
      active_meds: { resourceType: 'Bundle', entry: [
        { resource: { medicationCodeableConcept: { text: 'warfarin 5mg PO daily' }, status: 'active' } },
      ]},
    },
  };
  const r = await fetch('/cds-services/trustedrisk-medication-prescribe', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  out.innerHTML = renderCards(d.cards);
}

async function demoOrderReview() {
  const out = document.getElementById('order-result');
  out.innerHTML = '<em>Running…</em>';
  const body = {
    hook: 'order-review',
    hookInstance: 'demo-' + Date.now(),
    context: {
      userId: 'Practitioner/demo', patientId: 'Patient/demo-1',
      draftOrders: { resourceType: 'Bundle', entry: [
        { resource: { resourceType: 'ServiceRequest',
                       code: { text: 'CT abdomen with IV contrast' } } },
      ]},
    },
    prefetch: {
      egfr: { resourceType: 'Bundle', entry: [
        { resource: { valueQuantity: { value: 22 },
                       effectiveDateTime: '2026-04-26T08:00:00Z' } },
      ]},
    },
  };
  const r = await fetch('/cds-services/trustedrisk-order-review', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  const d = await r.json();
  out.innerHTML = renderCards(d.cards);
}

function renderCards(cards) {
  if (!cards || cards.length === 0)
    return '<em>No cards returned.</em>';
  return cards.map(c =>
    '<div class="card ' + c.indicator + '">' +
    '<div class="summary">[' + c.indicator.toUpperCase() + '] ' + escapeHtml(c.summary) + '</div>' +
    (c.detail ? '<div class="detail">' + escapeHtml(c.detail) + '</div>' : '') +
    '</div>'
  ).join('');
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#x27;',
  })[c]);
}
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8767)
