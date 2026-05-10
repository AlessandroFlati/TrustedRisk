"""FHIR client helpers built from the SHARP-on-MCP context.

All tools should go through `get_fhir_client()` to talk to the FHIR server;
the bearer token is propagated from X-FHIR-Access-Token via the SHARP context.
"""

from __future__ import annotations

from typing import Any

from ..sharp.headers import get_fhir_context


async def get_fhir_client():
    """Return an async fhirpy client bound to the current request's FHIR context."""
    try:
        from fhirpy import AsyncFHIRClient  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "fhirpy not installed. Install with `pip install fhirpy`."
        ) from e

    ctx = get_fhir_context()
    return AsyncFHIRClient(
        url=ctx.server_url,
        authorization=f"Bearer {ctx.access_token}",
    )


async def resolve_patient_id(explicit: str | None = None) -> str:
    """Return the patient_id from explicit argument or X-Patient-ID header.

    Raises ValueError if neither is provided -- used by tools that require patient context.
    """
    ctx = get_fhir_context()
    pid = (explicit or ctx.patient_id or "").strip()
    if not pid:
        raise ValueError(
            "patient_id is required: provide as tool argument or X-Patient-ID header."
        )
    return pid


async def fetch_patient_bundle(patient_id: str) -> dict[str, Any]:
    """Fetch all resources for a patient + return as a FHIR Bundle-like dict.

    Structure matches what `healthcare.io.fhir_bundle_inspect` expects:
      {"resourceType": "Bundle", "type": "collection", "entry": [...]}
    """
    client = await get_fhir_client()
    entries: list[dict[str, Any]] = []

    # Fetch Patient. FHIR R4 `_id` matches the bare resource id (no
    # `Patient/` prefix). Callers routinely pass the canonical reference
    # form `Patient/<uuid>` via `resolve_patient_id`, so strip it here
    # before the search -- otherwise the Patient resource is silently
    # missing from the bundle while related resources (Encounter,
    # Condition, ...) still resolve via `patient=` (which accepts both
    # bare and prefixed shapes).
    bare_id = patient_id.split("/", 1)[-1] if "/" in patient_id else patient_id
    patients = await client.resources("Patient").search(_id=bare_id).fetch()
    if patients:
        entries.append({"resource": _to_dict(patients[0])})

    # Fetch related resources. The set covers both real FHIR servers
    # (which typically expose MedicationRequest) and Prompt Opinion's
    # synthetic-patient store (which scopes patient/MedicationAdministration
    # + patient/DocumentReference instead, with meds living inside the
    # markdown body of clinical notes).
    #
    # For each resource type we try two patient-search shapes in order:
    #   1. patient=<bare-id>  (FHIR R4 spec, accepted by every server)
    #   2. patient=Patient/<id>  (legacy-friendly, kept as fallback)
    # Some servers (PO included) silently return zero entries for the
    # second shape via fhirpy's URL encoding even though both shapes
    # work in plain HTTP -- the bare-id form is the authoritative one.
    for resource_type in (
        "Encounter",
        "Condition",
        "MedicationRequest",
        "MedicationAdministration",
        "Observation",
        "Procedure",
        "DocumentReference",
        "AllergyIntolerance",
        "Coverage",
        "CarePlan",
        "Goal",
        "ServiceRequest",
        "Immunization",
        "ImmunizationRecommendation",
        "DiagnosticReport",
    ):
        resources = []
        for search_value in (bare_id, patient_id):
            try:
                resources = await (
                    client.resources(resource_type)
                    .search(patient=search_value).fetch()
                )
            except Exception:
                # OAuth scope may not cover this resource type. Try the
                # next search shape; if both fail, skip silently.
                continue
            if resources:
                break
        for r in resources:
            entries.append({"resource": _to_dict(r)})

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
    }


def _to_dict(resource: Any) -> dict[str, Any]:
    """Convert a fhirpy resource object to a plain dict."""
    if hasattr(resource, "serialize"):
        return resource.serialize()
    if hasattr(resource, "to_resource"):
        return resource.to_resource()
    # Fallback: assume it's already a dict
    return dict(resource) if hasattr(resource, "items") else {}
