"""DATA-2 unit tests for RxNorm DDI lookup."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools import rxnorm_ddi as mod
from mcp_server.tools.rxnorm_ddi import compute_rxnorm_ddi_lookup


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Validation ───────────────────────

def test_non_list_raises():
    with pytest.raises(ValueError, match="must be a list"):
        _run(compute_rxnorm_ddi_lookup("warfarin"))  # type: ignore[arg-type]


def test_empty_list_returns_abstain():
    r = _run(compute_rxnorm_ddi_lookup([]))
    assert r.abstain_recommended is True
    assert r.abstain_reason == "empty_drug_list"


def test_non_string_entries_filtered():
    r = _run(compute_rxnorm_ddi_lookup([None, 42, "warfarin"]))
    assert "warfarin" in r.drugs_requested
    assert len(r.drugs_requested) == 1


# ─────────────────────── Offline cache resolution ───────────────────────

def test_offline_cache_resolves_warfarin():
    r = _run(compute_rxnorm_ddi_lookup(["warfarin"]))
    assert r.rxnorm_ids_resolved.get("warfarin") == "11289"


def test_offline_cache_case_insensitive():
    r = _run(compute_rxnorm_ddi_lookup(["WARFARIN", "Warfarin"]))
    for d in r.drugs_requested:
        assert r.rxnorm_ids_resolved[d] == "11289"


def test_unknown_drug_falls_back_to_rxnav(monkeypatch):
    """Unknown drug name -> tries RxNav. Mock the network."""
    async def _fake_resolve(name, *, timeout_s=10.0):
        return "999999" if name.lower() == "unobtainium" else None
    monkeypatch.setattr(mod, "resolve_rxcui", _fake_resolve)
    r = _run(compute_rxnorm_ddi_lookup(["unobtainium"]))
    assert r.rxnorm_ids_resolved["unobtainium"] == "999999"
    assert r.method == "rxnav_live"


def test_unknown_drug_with_no_network_returns_none(monkeypatch):
    async def _fail(_name, *, timeout_s=10.0):
        return None
    monkeypatch.setattr(mod, "resolve_rxcui", _fail)
    r = _run(compute_rxnorm_ddi_lookup(["xyzzy"]))
    assert r.rxnorm_ids_resolved["xyzzy"] is None
    assert r.method == "deterministic_fallback"


# ─────────────────────── DDI cross-reference ───────────────────────

def test_warfarin_aspirin_high_severity_flagged():
    r = _run(compute_rxnorm_ddi_lookup(["warfarin", "aspirin"]))
    assert r.n_interactions == 1
    interaction = r.interactions[0]
    assert interaction.severity == "high"
    assert "bleeding" in interaction.description.lower()


def test_warfarin_ibuprofen_high_severity_flagged():
    r = _run(compute_rxnorm_ddi_lookup(["warfarin", "ibuprofen"]))
    assert r.n_interactions == 1
    assert r.interactions[0].severity == "high"


def test_lisinopril_spironolactone_moderate():
    r = _run(compute_rxnorm_ddi_lookup(["lisinopril", "spironolactone"]))
    assert r.n_interactions == 1
    assert r.interactions[0].severity == "moderate"
    assert "hyperkalemia" in r.interactions[0].description.lower()


def test_clopidogrel_omeprazole_moderate():
    r = _run(compute_rxnorm_ddi_lookup(["clopidogrel", "omeprazole"]))
    assert r.n_interactions == 1
    assert r.interactions[0].severity == "moderate"


def test_lithium_thiazide_high():
    r = _run(compute_rxnorm_ddi_lookup(["lithium", "hydrochlorothiazide"]))
    assert r.n_interactions == 1
    assert r.interactions[0].severity == "high"


def test_methotrexate_tmp_high():
    r = _run(compute_rxnorm_ddi_lookup(["methotrexate", "trimethoprim"]))
    assert r.n_interactions == 1
    assert r.interactions[0].severity == "high"


def test_no_interaction_for_clean_pair():
    """Acetaminophen + simvastatin have no curated interaction."""
    r = _run(compute_rxnorm_ddi_lookup(["acetaminophen", "simvastatin"]))
    assert r.n_interactions == 0


# ─────────────────────── Multi-drug cohorts ───────────────────────

def test_three_drug_cohort_finds_all_pairs():
    """warfarin + aspirin + ibuprofen -> 2 interactions (warf-asa, warf-ibu)."""
    r = _run(compute_rxnorm_ddi_lookup([
        "warfarin", "aspirin", "ibuprofen"]))
    assert r.n_interactions == 2
    pairs = {(i.drug_a, i.drug_b) for i in r.interactions}
    drug_names = set()
    for a, b in pairs:
        drug_names.add(a)
        drug_names.add(b)
    assert "warfarin" in drug_names


def test_full_cardiac_cohort():
    """Realistic CHF discharge: warfarin + aspirin + lisinopril +
    spironolactone + amiodarone + simvastatin -> multiple interactions."""
    r = _run(compute_rxnorm_ddi_lookup([
        "warfarin", "aspirin", "lisinopril", "spironolactone",
        "amiodarone", "simvastatin",
    ]))
    pairs = {(i.drug_a, i.drug_b) for i in r.interactions}
    expected_pairs = {
        ("warfarin", "aspirin"),
        ("warfarin", "amiodarone"),
        ("lisinopril", "spironolactone"),
        ("amiodarone", "simvastatin"),
    }
    assert expected_pairs <= pairs


def test_n_interactions_scales_with_pair_count():
    """5 mutually-non-interacting drugs -> 0 interactions."""
    r = _run(compute_rxnorm_ddi_lookup([
        "acetaminophen", "metformin", "pantoprazole", "sertraline",
        "hydrochlorothiazide",
    ]))
    assert r.n_interactions == 0


# ─────────────────────── Resolution sources ───────────────────────

def test_method_offline_when_only_cached():
    r = _run(compute_rxnorm_ddi_lookup(["warfarin", "aspirin"]))
    assert r.method == "offline_cache"


def test_method_rxnav_when_at_least_one_call(monkeypatch):
    async def _resolve(name, *, timeout_s=10.0):
        return "70000" if name.lower() == "newdrug" else None
    monkeypatch.setattr(mod, "resolve_rxcui", _resolve)
    r = _run(compute_rxnorm_ddi_lookup(["newdrug"]))
    assert r.method == "rxnav_live"


def test_method_fallback_when_nothing_resolved(monkeypatch):
    async def _fail(_name, *, timeout_s=10.0):
        return None
    monkeypatch.setattr(mod, "resolve_rxcui", _fail)
    r = _run(compute_rxnorm_ddi_lookup(["xyzzy", "zoofar"]))
    assert r.method == "deterministic_fallback"
    assert all(v is None for v in r.rxnorm_ids_resolved.values())


# ─────────────────────── References + provenance ───────────────────────

def test_references_include_rxnav():
    r = _run(compute_rxnorm_ddi_lookup(["warfarin"]))
    refs = " ".join(r.references)
    assert "RxNav" in refs or "RxNorm" in refs


def test_interaction_carries_source_attribution():
    r = _run(compute_rxnorm_ddi_lookup(["warfarin", "amiodarone"]))
    assert r.interactions[0].source != ""
    assert "FDA" in r.interactions[0].source or \
           "Lexicomp" in r.interactions[0].source


# ─────────────────────── Bundle registration ───────────────────────

def test_rxnorm_in_data_normalization_bundle():
    from mcp_server.tools import BUNDLES
    assert "compute_rxnorm_ddi_lookup" in BUNDLES["data_normalization"]


# ─────────────────────── Live RxNav (opt-in) ───────────────────────

@pytest.mark.skipif(
    "TRUSTEDRISK_LIVE_RXNAV" not in __import__("os").environ,
    reason="Set TRUSTEDRISK_LIVE_RXNAV=1 to hit the live NLM API.",
)
def test_live_rxnav_resolves_real_drug():
    from mcp_server.tools.rxnorm_ddi import resolve_rxcui
    cui = _run(resolve_rxcui("warfarin"))
    assert cui == "11289"
