"""Generate the 5 specialist agent_card.json files from the master card.

Re-run after changing `src/a2a_agent/agent-card.json` (master), the
specialist roster, or `src/mcp_server/scopes.py` (per-bundle SMART
scopes).

The output cards stay in sync with the master across:
  - `name` / `description` / `version` / `schemaVersion`
  - `capabilities` (PO FHIR-context extension URI is identical; only
    the embedded `params.scopes` list narrows to the specialist's
    bundle subset)
  - `securitySchemes` (identical -- same OAuth + API-key surface)
  - skill descriptions / examples / tags (subset filtered by skill ID)
  - `_bundles` (subset filtered by bundle ID)

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_specialist_cards.py
"""

from __future__ import annotations

import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Imported here so the script fails loudly if the scopes module is
# missing -- the specialist cards' scope decls depend on it.
from mcp_server.scopes import scope_objects


# Resolve the public deploy host from the environment. When unset we
# leave the literal placeholder string so a generated card flags itself
# clearly as "not yet wired to a public URL".
_DEPLOY_HOST = os.environ.get("TRUSTEDRISK_DEPLOY_HOST", "<deploy-url>")


# User-friendly display names. The card's `name` field becomes the
# label shown in the A2A v1 chat client's dropdown, so it should let a
# clinician spot "what does this agent do?" at a glance. The URL slug
# (last path segment of `base_url_placeholder`) stays stable as the
# routing identifier, so renaming `name` does not break registrations.
_DISPLAY_NAMES = {
    "specialist_discharge":     "Discharge Planner",
    "specialist_acute":         "Acute Care Specialist",
    "specialist_evidence":      "Evidence & Diagnosis",
    "specialist_population":    "Health Economics (Cost / QALY)",
    "specialist_pediatric":     "Pediatric Care",
    "specialist_mental_health": "Mental Health Crisis",
    "specialist_pa":            "Prior Authorization",
    "specialist_scribe":        "Clinical Scribe",
    "specialist_patient":       "Patient Advocate",
    "specialist_coder":         "Medical Coder (ICD/CPT)",
    "specialist_pgx":           "Genetic Drug Safety (PGx)",
    "specialist_preadmit":      "Pre-Op Risk Triage",
    "specialist_quality":       "HEDIS Quality Stars",
    "specialist_pophealth":     "Population Surveillance",
    "specialist_appeals":       "Insurance Appeals",
    "specialist_multimodal":    "ECG & DICOM",
}


def _resolve_base_url(template: str) -> str:
    return template.replace("<deploy-url>", _DEPLOY_HOST)


def _substitute_deploy_host(obj: Any) -> Any:
    """Recursively replace `<deploy-url>` inside any string in `obj`.

    Used to substitute the placeholder anywhere it appears in the
    copied master schema -- in particular the OAuth2 `tokenUrl` under
    `securitySchemes.oauth_client_credentials.oauth2SecurityScheme.flows.
    clientCredentials.tokenUrl`.
    """
    if isinstance(obj, str):
        return obj.replace("<deploy-url>", _DEPLOY_HOST)
    if isinstance(obj, dict):
        return {k: _substitute_deploy_host(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_deploy_host(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────
# Specialist roster
# ─────────────────────────────────────────────────────────────────────
#
# Each entry maps a specialist id to its public name, port, base URL,
# the skills (by skill `id`) it advertises, and the bundles (by bundle
# id) whose scopes it requires.

SPECIALISTS: list[dict[str, Any]] = [
    {
        "id": "specialist_discharge",
        "name": "trustedrisk-discharge",
        "tagline": "Adult inpatient discharge planning + outpatient medication review",
        "port": 8770,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-discharge",
        "skills": [
            "safe_discharge_review",
            "medication_safety_review",
            "phi_check",
            "discharge_counseling",
            "antimicrobial_stewardship",
            "oncology_decisions",
            "patient_facing_translation",
            "data_normalization",
            "clinical_workflow_orders",
        ],
        "bundles": [
            "core_discharge",
            "clinical_workflow",
            "patient_facing",
            "data_normalization",
            "antimicrobial",
            "oncology",
        ],
    },
    {
        "id": "specialist_acute",
        "name": "trustedrisk-acute",
        "tagline": "Emergency department + acute / critical-care decision support",
        "port": 8771,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-acute",
        "skills": [
            "ed_admission_triage",
            "deterioration_score",
            "acute_stroke",
            "chest_pain_acs",
            "trauma_critical_care",
            "endocrine_acute",
            "imaging_appropriateness_safety",
            "nephrology",
            "maternal_obstetric",
            "geriatric_assessment",
            "phi_check",
        ],
        "bundles": [
            "ed_acute",
            "stroke_acs",
            "trauma_critical",
            "endocrine_acute",
            "imaging",
            "nephrology",
            "obstetric_geriatric",
        ],
    },
    {
        "id": "specialist_evidence",
        "name": "trustedrisk-evidence",
        "tagline": "Clinician evidence retrieval + grounded differential diagnosis",
        "port": 8772,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-evidence",
        "skills": [
            "differential_diagnosis",
            "treatment_selection",
            "external_knowledge_retrieval",
            "chart_intelligence",
            "context_resolution",
            "phi_check",
        ],
        "bundles": [
            "diagnosis",
            "external_knowledge",
            "chart_intelligence",
            "context_resolution",
        ],
    },
    {
        "id": "specialist_population",
        "name": "trustedrisk-population",
        "tagline": "Cost-effectiveness + subgroup fairness + DP equity dashboard",
        "port": 8773,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-population",
        "skills": [
            "cost_effectiveness",
            "phi_check",
        ],
        "bundles": [
            "economics",
        ],
    },
    {
        "id": "specialist_pediatric",
        "name": "trustedrisk-pediatric",
        "tagline": "Pediatric early warning + weight-based dosing",
        "port": 8774,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-pediatric",
        "skills": [
            "pediatric_assessment",
            "phi_check",
        ],
        "bundles": [
            "pediatric",
        ],
    },
    {
        "id": "specialist_mental_health",
        "name": "trustedrisk-mental-health",
        "tagline": "Suicide risk screening + psychiatric admission decision",
        "port": 8785,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-mental-health",
        "skills": [
            "mental_health_crisis",
            "phi_check",
        ],
        "bundles": [
            "mental_health",
        ],
    },
    {
        "id": "specialist_pa",
        "name": "trustedrisk-pa",
        "tagline": "Prior Authorization evidence + letter + appeal-likelihood (Phase 2.1)",
        "port": 8775,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-pa",
        "skills": [
            "prior_authorization",
            "phi_check",
        ],
        "bundles": [
            "prior_authorization",
        ],
    },
    {
        "id": "specialist_scribe",
        "name": "trustedrisk-scribe",
        "tagline": "Clinical documentation drafting (Phase 2.2 SCRIBE-1/2/3/4)",
        "port": 8776,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-scribe",
        "skills": [
            "clinical_documentation",
            "phi_check",
        ],
        "bundles": [
            "clinical_documentation",
        ],
    },
    {
        "id": "specialist_patient",
        "name": "trustedrisk-patient",
        "tagline": "Post-discharge Q&A + caregiver hand-off (Phase 2.3 PATIENT-Q1/2/3)",
        "port": 8777,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-patient",
        "skills": [
            "patient_qa",
            "patient_facing_translation",
        ],
        "bundles": [
            "patient_qa",
            "patient_facing",
        ],
    },
    {
        "id": "specialist_coder",
        "name": "trustedrisk-coder",
        "tagline": "Auto-coding agent (Phase 7.1 CODE-1/2/3/4)",
        "port": 8778,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-coder",
        "skills": [
            "auto_coding",
            "phi_check",
        ],
        "bundles": [
            "auto_coding",
        ],
    },
    {
        "id": "specialist_pgx",
        "name": "trustedrisk-pgx",
        "tagline": "Pharmacogenomic DSS (Phase 7.2 PGX-1/2/3)",
        "port": 8779,
        "base_url_placeholder": "https://<deploy-url>/a2a/trustedrisk-pgx",
        "skills": [
            "pharmacogenomics",
            "phi_check",
        ],
        "bundles": [
            "pharmacogenomics",
        ],
    },
    {
        "id": "specialist_preadmit",
        "name": "trustedrisk-preadmit",
        "tagline": "Pre-arrival triage agent (Phase 7.5 PREADMIT-1/2/3)",
        "port": 8781,
        "base_url_placeholder":
            "https://<deploy-url>/a2a/trustedrisk-preadmit",
        "skills": [
            "preadmit_triage",
            "phi_check",
        ],
        "bundles": [
            "preadmit_triage",
        ],
    },
    {
        "id": "specialist_quality",
        "name": "trustedrisk-quality",
        "tagline": "HEDIS / CMS Stars Rating + care-gap prioritisation (Phase 10.1 STARS-1/2/3)",
        "port": 8782,
        "base_url_placeholder":
            "https://<deploy-url>/a2a/trustedrisk-quality",
        "skills": [
            "quality_stars",
            "phi_check",
        ],
        "bundles": [
            "quality_stars",
        ],
    },
    {
        "id": "specialist_pophealth",
        "name": "trustedrisk-pophealth",
        "tagline": "Population health + outbreak detection (Phase 10.2 POPHEALTH-1/2/3)",
        "port": 8783,
        "base_url_placeholder":
            "https://<deploy-url>/a2a/trustedrisk-pophealth",
        "skills": [
            "population_health",
            "phi_check",
        ],
        "bundles": [
            "population_health",
        ],
    },
    {
        "id": "specialist_appeals",
        "name": "trustedrisk-appeals",
        "tagline": "Insurance appeals: denial parse + letter draft + escalation (Phase 10.3 APPEALS-1/2/3)",
        "port": 8784,
        "base_url_placeholder":
            "https://<deploy-url>/a2a/trustedrisk-appeals",
        "skills": [
            "insurance_appeals",
            "phi_check",
        ],
        "bundles": [
            "insurance_appeals",
        ],
    },
    {
        "id": "specialist_multimodal",
        "name": "trustedrisk-multimodal",
        "tagline": "Multi-modal: ECG QT analyzer + DICOM SR ingest (Phase 11.4 MULTIMODAL-1/2)",
        "port": 8786,
        "base_url_placeholder":
            "https://<deploy-url>/a2a/trustedrisk-multimodal",
        "skills": [
            "multimodal",
            "phi_check",
        ],
        "bundles": [
            "multimodal",
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────
# Card generator
# ─────────────────────────────────────────────────────────────────────

def _master_card_path() -> Path:
    return ROOT / "src" / "a2a_agent" / "agent-card.json"


def _output_card_path(specialist_id: str) -> Path:
    return ROOT / "apps" / specialist_id / "agent_card.json"


def _description_for(spec: dict[str, Any], master_skills: list[dict]) -> str:
    """Build a focused description from the specialist tagline + the
    list of advertised skill names."""
    skill_names = [s["name"] for s in master_skills]
    return (
        f"{spec['tagline']}. Marketplace-publishable A2A specialist "
        f"sharing the TrustedRisk MCP backend. Advertises "
        f"{len(spec['skills'])} skills: {', '.join(skill_names)}. "
        f"FHIR context propagated via the A2A v1 FHIR-context "
        f"extension; tools see the FHIRContext via the shared "
        f"ContextVar without ever surfacing the bearer token to the LLM "
        f"prompt. Every recommendation surfaces a calibrated CI when "
        f"applicable, a deterministic safety gate, an abstain trigger "
        f"when uncertainty or demographic bias is high, a 4-critic "
        f"ensemble vote (clinical_safety + fairness + evidence + "
        f"llm_judge) with bounded plan revision, a fairness audit by "
        f"demographic subgroup, and a longitudinal patient timeline "
        f"that surfaces drift across encounters."
    )


def _filter_extensions_for_specialist(
    master_extensions: list[dict],
    bundle_ids: list[str],
) -> list[dict]:
    """Filter the master extensions list, replacing the FHIR-context
    extension's `params.scopes` with the union of scopes required by
    the specialist's bundles. Other extensions pass through unchanged."""
    PO_URI = "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context"
    out: list[dict] = []
    for ext in master_extensions:
        ext = deepcopy(ext)
        if ext.get("uri") == PO_URI:
            ext.setdefault("params", {})["scopes"] = scope_objects(bundle_ids)
        out.append(ext)
    return out


def _build_specialist_card(spec: dict[str, Any], master: dict) -> dict:
    """Produce the agent-card for one specialist."""
    # Filter skills
    master_skills = master.get("skills", [])
    skill_index = {s["id"]: s for s in master_skills}
    missing = [s for s in spec["skills"] if s not in skill_index]
    if missing:
        raise ValueError(
            f"Specialist {spec['id']!r}: skills not in master card: {missing}"
        )
    skills = [skill_index[s] for s in spec["skills"]]

    # Filter bundles (preserve master's _doc annotation)
    master_bundles = master.get("_bundles", {})
    filtered_bundles: dict[str, Any] = {}
    if "_doc" in master_bundles:
        filtered_bundles["_doc"] = master_bundles["_doc"]
    for bid in spec["bundles"]:
        if bid in master_bundles:
            filtered_bundles[bid] = master_bundles[bid]

    # Capabilities: copy master, replace extensions
    capabilities = deepcopy(master.get("capabilities", {}))
    capabilities["extensions"] = _filter_extensions_for_specialist(
        capabilities.get("extensions", []),
        spec["bundles"],
    )

    base_url = _resolve_base_url(spec["base_url_placeholder"])
    display_name = _DISPLAY_NAMES.get(spec["id"], spec["name"])
    return {
        "schemaVersion": master.get("schemaVersion", "1.0.0"),
        "name": display_name,
        "_legacy_name": spec["name"],
        "description": _description_for(spec, skills),
        "version": master.get("version", "0.7.0"),
        "_bundles": filtered_bundles,
        "_specialist": spec["id"],
        "url": base_url,
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "A2A",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": master.get(
            "defaultInputModes", ["text/plain", "application/json"]
        ),
        "defaultOutputModes": master.get(
            "defaultOutputModes", ["application/json"]
        ),
        "skills": skills,
        "capabilities": capabilities,
        "securitySchemes": _substitute_deploy_host(
            deepcopy(master.get("securitySchemes", {}))
        ),
    }


def main() -> None:
    master = json.loads(_master_card_path().read_text(encoding="utf-8"))
    for spec in SPECIALISTS:
        out_path = _output_card_path(spec["id"])
        if not out_path.parent.exists():
            raise FileNotFoundError(
                f"Specialist directory missing: {out_path.parent!s}"
            )
        card = _build_specialist_card(spec, master)
        out_path.write_text(
            json.dumps(card, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Wrote {out_path.relative_to(ROOT)!s} -- "
            f"{len(card['skills'])} skills, "
            f"{len([k for k in card['_bundles'] if not k.startswith('_')])} bundles, "
            f"{len(card['capabilities']['extensions'][0]['params']['scopes'])} scopes."
        )


if __name__ == "__main__":
    main()
