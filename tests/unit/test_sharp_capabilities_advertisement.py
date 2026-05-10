"""Unit tests for the MCP `initialize` capability advertisement.

Verifies that SHARP-on-MCP capabilities surface BOTH the vendor-neutral
SHARP draft path (`experimental.fhir_context_required`) AND the Prompt
Opinion namespaced path (`ai.promptopinion/fhir-context`) with the
expected SMART scope shape.
"""

from __future__ import annotations

from mcp_server.scopes import (
    BUNDLE_SCOPES,
    list_bundle_scopes,
    required_scopes,
    scope_objects,
    union_scopes,
)
from mcp_server.sharp.headers import initialize_capabilities


# ─────────────────────── initialize_capabilities ───────────────────────

def test_initialize_advertises_vendor_neutral_capability():
    caps = initialize_capabilities()
    assert "experimental" in caps
    assert "fhir_context_required" in caps["experimental"]
    decl = caps["experimental"]["fhir_context_required"]
    assert decl["value"] is True
    for h in ("X-FHIR-Server-URL", "X-FHIR-Access-Token", "X-Patient-ID"):
        assert h in decl["headers"]


def test_initialize_advertises_offline_access_headers():
    decl = initialize_capabilities()["experimental"]["fhir_context_required"]
    offline = decl.get("headers_offline_access", [])
    assert "X-FHIR-Refresh-Token" in offline
    assert "X-FHIR-Refresh-Url" in offline


def test_initialize_advertises_prompt_opinion_capability():
    caps = initialize_capabilities()
    assert "ai.promptopinion/fhir-context" in caps
    decl = caps["ai.promptopinion/fhir-context"]
    assert "scopes" in decl
    assert isinstance(decl["scopes"], list)
    assert decl["scopes"], "Scopes list must not be empty"


def test_prompt_opinion_scopes_shape_matches_spec():
    """Scope objects must be `{name: str, required?: bool}`, exactly as
    declared at https://docs.promptopinion.ai/fhir-context/mcp-fhir-context.html."""
    decl = initialize_capabilities()["ai.promptopinion/fhir-context"]
    for s in decl["scopes"]:
        assert isinstance(s, dict)
        assert isinstance(s.get("name"), str) and s["name"]
        # `required` is optional but must be bool when present
        if "required" in s:
            assert isinstance(s["required"], bool)


def test_prompt_opinion_scopes_include_patient_required():
    decl = initialize_capabilities()["ai.promptopinion/fhir-context"]
    patient = next((s for s in decl["scopes"]
                       if s["name"] == "patient/Patient.rs"), None)
    assert patient is not None, "patient/Patient.rs must be advertised"
    assert patient.get("required") is True


# ─────────────────────── scopes module ───────────────────────

def test_bundle_scopes_covers_all_bundles():
    """Single source of truth: scopes.BUNDLE_SCOPES must match
    mcp_server.tools.BUNDLES exactly so the agent-card scope
    advertisement never diverges from the runtime tool registration."""
    from mcp_server.tools import BUNDLES
    assert set(BUNDLE_SCOPES.keys()) == set(BUNDLES.keys()), (
        f"BUNDLE_SCOPES drift vs BUNDLES: scopes={set(BUNDLE_SCOPES.keys())}, "
        f"bundles={set(BUNDLES.keys())}"
    )


def test_external_knowledge_has_no_fhir_scopes():
    """Pure external-retrieval bundle -- no FHIR access required."""
    assert BUNDLE_SCOPES["external_knowledge"] == []


def test_chart_intelligence_includes_document_resources():
    scopes = BUNDLE_SCOPES["chart_intelligence"]
    assert "patient/DocumentReference.rs" in scopes
    assert "patient/Composition.rs" in scopes


def test_union_scopes_default_returns_full_set():
    union = union_scopes()
    assert "patient/Patient.rs" in union
    # Patient is the first element by convention (most fundamental)
    assert union[0] == "patient/Patient.rs"
    # Rest is sorted alphabetically
    assert union[1:] == sorted(union[1:])


def test_union_scopes_subset_filters_correctly():
    union = union_scopes(["external_knowledge"])
    assert union == []   # external_knowledge has no scopes


def test_required_scopes_includes_patient():
    req = required_scopes()
    assert req == ["patient/Patient.rs"]


def test_scope_objects_marks_patient_required():
    objs = scope_objects()
    patient = next((o for o in objs if o["name"] == "patient/Patient.rs"), None)
    assert patient is not None
    assert patient.get("required") is True
    # Other scopes should NOT be marked required
    for o in objs:
        if o["name"] != "patient/Patient.rs":
            assert "required" not in o or o.get("required") is False


def test_list_bundle_scopes_returns_independent_copy():
    """Mutating the returned dict must not affect the source."""
    snapshot = list_bundle_scopes()
    snapshot["core_discharge"].append("EVIL_SCOPE")
    assert "EVIL_SCOPE" not in BUNDLE_SCOPES["core_discharge"]
