"""Per-bundle SMART-on-FHIR scope declarations.

Used by the MCP `initialize` capability advertisement and by the A2A
agent-card extension declaration. Each bundle lists the FHIR resource
read/search scopes its tools need at the upper bound -- a workspace
admin can pull from this when granting tokens.

Scope syntax follows SMART v2 (`<context>/<Resource>.<rights>` where
rights ⊂ {c,r,u,d,s}). All TrustedRisk tools are read-only on the FHIR
side, so we use `.rs` (read + search) uniformly.

`patient/Patient.rs` is implicit on every bundle -- TrustedRisk needs a
Patient resource for cohort-level / patient-level scopes; the
declaration adds it explicitly so workspace UIs can reason about
resource needs without inferring from tool semantics.
"""

from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Per-bundle scope requirements
# ─────────────────────────────────────────────────────────────────────
#
# Conventions:
# - `patient/Patient.rs` -> required (always, every bundle except external_knowledge)
# - other resources -> required when the bundle's primary tools depend on them
# - `system/...` -> reserved for cohort / population analytics tools
#
# When a bundle declares no FHIR scopes (e.g. external_knowledge), the
# bundle still advertises an empty list rather than absent so consumers
# can detect "no FHIR access needed".

BUNDLE_SCOPES: dict[str, list[str]] = {
    "core_discharge": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Condition.rs",
        "patient/MedicationRequest.rs",
        "patient/Observation.rs",
        "patient/Procedure.rs",
        "patient/AllergyIntolerance.rs",
    ],
    "ed_acute": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "pediatric": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/AllergyIntolerance.rs",
    ],
    "mental_health": [
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/Encounter.rs",
    ],
    "antimicrobial": [
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
        "patient/AllergyIntolerance.rs",
    ],
    "oncology": [
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
    ],
    "stroke_acs": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "obstetric_geriatric": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/MedicationRequest.rs",
    ],
    "trauma_critical": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "endocrine_acute": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
    ],
    "imaging": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Procedure.rs",
        "patient/ServiceRequest.rs",
    ],
    "nephrology": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/Procedure.rs",
    ],
    "economics": [
        "patient/Patient.rs",
    ],
    "context_resolution": [
        "patient/Patient.rs",
    ],
    "diagnosis": [
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
    ],
    "patient_facing": [
        "patient/Patient.rs",
    ],
    "data_normalization": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
    ],
    "clinical_workflow": [
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
    ],
    "external_knowledge": [
        # No FHIR resources -- pure external retrieval (PubMed / CT.gov /
        # NIH RePORTER / drug pricing).
    ],
    "chart_intelligence": [
        "patient/Patient.rs",
        "patient/DocumentReference.rs",
        "patient/Composition.rs",
    ],
    "prior_authorization": [
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
        "patient/Procedure.rs",
        "patient/AllergyIntolerance.rs",
        "patient/DocumentReference.rs",
    ],
    "clinical_documentation": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
        "patient/AllergyIntolerance.rs",
        "patient/Procedure.rs",
        "patient/DocumentReference.rs",
    ],
    "patient_qa": [
        "patient/Patient.rs",
    ],
    "auto_coding": [
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Condition.rs",
        "patient/Procedure.rs",
        "patient/MedicationRequest.rs",
        "patient/DocumentReference.rs",
        "patient/Composition.rs",
    ],
    "pharmacogenomics": [
        "patient/Patient.rs",
        "patient/MedicationRequest.rs",
        "patient/Observation.rs",
        "patient/MolecularSequence.rs",
        "patient/Specimen.rs",
    ],
    "preadmit_triage": [
        # Pre-arrival -- patient typed input only; the patient/Patient.rs
        # scope is documented for workspace UIs that want to attach the
        # user's identity, but the tools themselves require no FHIR.
        "patient/Patient.rs",
    ],
    "research_design": [
        "patient/Patient.rs",
    ],
    "quality_stars": [
        # HEDIS measures pull from Conditions/Observations/Meds/Encounters
        # at the cohort level -- workspace admins typically grant
        # `system/...` for population sweeps; we publish the
        # patient-scoped union for the per-patient close-the-loop UI.
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
        "patient/Encounter.rs",
        "patient/Procedure.rs",
    ],
    "population_health": [
        # Cohort-level surveillance + cohort-level immunisation gaps:
        # syndromic surveillance reads chief-complaint Encounter codes;
        # vaccine cohorts read Immunization resources; the DP heatmap
        # publishes aggregate counts only.
        "patient/Patient.rs",
        "patient/Encounter.rs",
        "patient/Condition.rs",
        "patient/Immunization.rs",
        "patient/Observation.rs",
    ],
    "insurance_appeals": [
        # Appeals letter draft pulls Conditions/Observations/Procedures/
        # MedicationRequests + DocumentReference to cite-back the
        # medical-necessity argument; the patient floor is required.
        "patient/Patient.rs",
        "patient/Condition.rs",
        "patient/Observation.rs",
        "patient/MedicationRequest.rs",
        "patient/Procedure.rs",
        "patient/DocumentReference.rs",
        "patient/Coverage.rs",
    ],
    "multimodal": [
        # Multi-modal: ECG signal + DICOM SR / DiagnosticReport. The
        # tools accept dict-shaped inputs so the SMART scope set is
        # the consuming UI's responsibility, but we publish the
        # canonical R4 surfaces a workspace UI would attach.
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/DiagnosticReport.rs",
        "patient/ImagingStudy.rs",
    ],
    "critical_care": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Encounter.rs",
        "patient/Condition.rs",
        "patient/MedicationAdministration.rs",
    ],
    "specialty_clinics": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/DiagnosticReport.rs",
    ],
    "cardiology_depth": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/MedicationRequest.rs",
    ],
    "heme_onc_depth": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/DiagnosticReport.rs",
    ],
    "endocrinology_advanced": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/MedicationRequest.rs",
    ],
    "sleep_pain": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "transplant": [
        "patient/Patient.rs",
        "patient/Observation.rs",
        "patient/Condition.rs",
        "patient/MolecularSequence.rs",
        "patient/Procedure.rs",
    ],
    "fhir_writeback": [
        # WRITE scopes -- required for the Composition + Provenance POST.
        # The READ scope is the patient-floor we already publish; the
        # `cu` (create + update) rights unlock the write-back.
        "patient/Patient.rs",
        "patient/Composition.cu",
        "patient/Provenance.cu",
    ],
    "rheumatology": [
        "patient/Patient.rs", "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "peri_op_risk": [
        "patient/Patient.rs", "patient/Observation.rs",
        "patient/Condition.rs", "patient/Procedure.rs",
    ],
    "infectious_disease": [
        "patient/Patient.rs", "patient/Observation.rs",
        "patient/Condition.rs", "patient/MedicationRequest.rs",
    ],
    "gi_hepatology_depth": [
        "patient/Patient.rs", "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "neurology_depth": [
        "patient/Patient.rs", "patient/Observation.rs",
        "patient/Condition.rs",
    ],
    "ob_peds_advanced": [
        "patient/Patient.rs", "patient/Observation.rs",
        "patient/Condition.rs", "patient/Encounter.rs",
    ],
    "model_research": [
        # Cohort-level analytics -- tools accept dict-shaped inputs;
        # the SMART scopes published here are advisory.
        "patient/Patient.rs",
    ],
    "legacy_ehr_parsers": [
        # HL7 v2 / C-CDA -- input is a raw payload string, not FHIR.
        # SMART scopes published here are advisory.
        "patient/Patient.rs",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Marketplace scope set -- the union (so the agent-card declares the
# upper bound; workspace admins narrow as needed).
# ─────────────────────────────────────────────────────────────────────

def union_scopes(bundle_ids: list[str] | None = None) -> list[str]:
    """Return the deduplicated union of scopes across the named bundles.

    If `bundle_ids` is None -> union over all bundles. Order is
    Patient-first then alphabetical so the declaration is stable across
    runs (relevant for marketplace-listing diffs).
    """
    if bundle_ids is None:
        bundle_ids = list(BUNDLE_SCOPES.keys())
    seen: set[str] = set()
    for bid in bundle_ids:
        seen.update(BUNDLE_SCOPES.get(bid, []))
    # Patient.rs first (most fundamental), then sorted
    out = []
    if "patient/Patient.rs" in seen:
        out.append("patient/Patient.rs")
        seen.discard("patient/Patient.rs")
    out.extend(sorted(seen))
    return out


def required_scopes(bundle_ids: list[str] | None = None) -> list[str]:
    """Return only the scopes considered required (Patient.rs is the
    canonical universal floor; everything else is conditionally required
    per bundle, but we publish all of them as `required: false` and let
    workspace admins toggle).
    """
    return ["patient/Patient.rs"] if any(
        "patient/Patient.rs" in BUNDLE_SCOPES.get(b, [])
        for b in (bundle_ids or BUNDLE_SCOPES.keys())
    ) else []


def scope_objects(bundle_ids: list[str] | None = None) -> list[dict]:
    """Return scope declarations in the Prompt Opinion capability shape:

        [{"name": "patient/Patient.rs", "required": true},
         {"name": "patient/Condition.rs"}, ...]

    Per the PO spec, only the universally-required scopes carry
    `required: true` -- everything else is optional at the agent level
    and gates per-bundle at runtime.
    """
    union = union_scopes(bundle_ids)
    required = set(required_scopes(bundle_ids))
    return [
        {"name": s, "required": True} if s in required else {"name": s}
        for s in union
    ]


def list_bundle_scopes() -> dict[str, list[str]]:
    """Return per-bundle scope map (read-only convenience)."""
    return {k: list(v) for k, v in BUNDLE_SCOPES.items()}
