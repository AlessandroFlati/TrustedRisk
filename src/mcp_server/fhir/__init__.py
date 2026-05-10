"""FHIR client helpers -- build from SHARP context."""

from .client import fetch_patient_bundle, get_fhir_client, resolve_patient_id

__all__ = ["fetch_patient_bundle", "get_fhir_client", "resolve_patient_id"]
