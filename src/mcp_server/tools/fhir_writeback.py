"""healthcare.compute_write_decision_to_fhir -- Phase 13.10 F1.

EHR write-back: emit a FHIR R4 `Composition` + `Provenance` pair
representing the agent's DecisionCard for the current SHARP-context
patient. Idempotent via `Composition.identifier` (system + value).

The two POSTs go through `httpx.AsyncClient` directly (rather than the
fhirpy read-path used by the rest of the system) so the write-back
surface stays narrowly scoped: a write tool should not accidentally
inherit retry / pagination behaviour built for reads.

References:
- HL7 FHIR R4 Composition: https://hl7.org/fhir/R4/composition.html
- HL7 FHIR R4 Provenance: https://hl7.org/fhir/R4/provenance.html
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from shared.schemas import FHIRWriteBackReport

from ..fhir.client import resolve_patient_id
from ..sharp.headers import get_fhir_context


_TR_SYSTEM = "https://trustedrisk.local/decision-card-id"


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


async def _put_resource(
    client, server_url: str, resource_type: str,
    identifier_system: str, identifier_value: str,
    payload: dict[str, Any], bearer_token: str,
) -> tuple[int, str]:
    """PUT a resource by identifier (idempotent upsert per FHIR R4).

    Returns (status_code, server_assigned_id). On a 200/201 the server
    echoes back the resource id in the response body.
    """
    url = (
        f"{server_url.rstrip('/')}/{resource_type}"
        f"?identifier={identifier_system}|{identifier_value}"
    )
    headers = {
        "Content-Type": "application/fhir+json",
        "Accept": "application/fhir+json",
    }
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    resp = await client.put(url, json=payload, headers=headers, timeout=15.0)
    body: dict[str, Any] = {}
    try:
        body = resp.json()
    except Exception:
        pass
    server_id = (
        body.get("id")
        or payload.get("id", "")
        or _stable_id(resource_type.lower(), identifier_value)
    )
    return resp.status_code, str(server_id)


async def compute_write_decision_to_fhir(
    decision_summary: str,
    *,
    patient_id: str | None = None,
    decision_card_logical_id: str,
    recommended_action: str | None = None,
    rationale_text: str | None = None,
    cited_resource_ids: list[str] | None = None,
) -> FHIRWriteBackReport:
    """Write a DecisionCard back to the FHIR server as a
    `Composition` + `Provenance` pair.

    Args:
        decision_summary: short prose summary (used for Composition.title).
        patient_id: optional explicit Patient ID; defaults to the
            SHARP context's `X-Patient-ID`.
        decision_card_logical_id: caller-provided id that uniquely
            identifies the DecisionCard. The same id always yields
            the same Composition.id on PUT-by-identifier (idempotency).
        recommended_action: free-text recommendation (e.g.
            'discharge_home').
        rationale_text: longer rationale; goes into Composition.section.
        cited_resource_ids: list of FHIR resource ids the decision
            cites (Conditions, Observations, MedicationRequests).
            They land as `Composition.section.entry` references and
            as `Provenance.entity` records.

    Returns:
        FHIRWriteBackReport.
    """
    import httpx

    pid = await resolve_patient_id(patient_id)
    ctx = get_fhir_context()
    server = (ctx.server_url or "").rstrip("/")
    token = ctx.access_token or ""

    cited_resource_ids = cited_resource_ids or []

    composition_id = _stable_id("decision", decision_card_logical_id)
    provenance_id = _stable_id("prov", decision_card_logical_id)
    now_iso = datetime.now(timezone.utc).isoformat()

    composition: dict[str, Any] = {
        "resourceType": "Composition",
        "id": composition_id,
        "identifier": [{
            "system": _TR_SYSTEM, "value": decision_card_logical_id,
        }],
        "status": "final",
        "type": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "11488-4",
                "display": "Consult note",
            }],
            "text": "TrustedRisk DecisionCard",
        },
        "subject": {"reference": f"Patient/{pid}"},
        "date": now_iso,
        "author": [{
            "reference": "Device/trustedrisk-agent",
            "display": "TrustedRisk agent",
        }],
        "title": decision_summary[:140],
        "section": [
            {
                "title": "Recommendation",
                "text": {
                    "status": "additional",
                    "div": (
                        f"<div xmlns=\"http://www.w3.org/1999/xhtml\">"
                        f"{recommended_action or 'unspecified'}</div>"
                    ),
                },
                "entry": [
                    {"reference": ref} for ref in cited_resource_ids[:20]
                ],
            },
            {
                "title": "Rationale",
                "text": {
                    "status": "additional",
                    "div": (
                        f"<div xmlns=\"http://www.w3.org/1999/xhtml\">"
                        f"{(rationale_text or '')[:1000]}</div>"
                    ),
                },
            },
        ],
    }

    provenance: dict[str, Any] = {
        "resourceType": "Provenance",
        "id": provenance_id,
        "identifier": [{
            "system": _TR_SYSTEM, "value": f"prov-{decision_card_logical_id}",
        }],
        "target": [{"reference": f"Composition/{composition_id}"}],
        "recorded": now_iso,
        "agent": [{
            "type": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/v3-RoleClass",
                    "code": "ASSIGNED", "display": "TrustedRisk agent",
                }],
            },
            "who": {
                "reference": "Device/trustedrisk-agent",
                "display": "TrustedRisk agent",
            },
        }],
        "entity": [
            {"role": "source", "what": {"reference": ref}}
            for ref in cited_resource_ids[:20]
        ],
    }

    async with httpx.AsyncClient() as client:
        comp_status, comp_returned_id = await _put_resource(
            client, server, "Composition",
            _TR_SYSTEM, decision_card_logical_id,
            composition, token,
        )
        prov_status, _prov_returned_id = await _put_resource(
            client, server, "Provenance",
            _TR_SYSTEM, f"prov-{decision_card_logical_id}",
            provenance, token,
        )

    succeeded = (
        200 <= comp_status < 300 and 200 <= prov_status < 300
    )
    rationale = (
        f"Wrote Composition {composition_id} + Provenance "
        f"{provenance_id} to {server} "
        f"(comp HTTP {comp_status}, prov HTTP {prov_status}); "
        f"succeeded = {succeeded}; idempotent via identifier."
    )
    return FHIRWriteBackReport(
        fhir_server_url=server, patient_id=pid,
        composition_id=comp_returned_id or composition_id,
        composition_identifier=decision_card_logical_id,
        provenance_id=provenance_id,
        composition_status_code=comp_status,
        provenance_status_code=prov_status,
        write_succeeded=succeeded,
        written_at_iso=now_iso,
        rationale=rationale,
        references=[
            "HL7 FHIR R4 Composition resource.",
            "HL7 FHIR R4 Provenance resource.",
        ],
    )


def register(mcp) -> None:
    mcp.tool()(compute_write_decision_to_fhir)
