"""Unit tests for KNOWLEDGE-1/2/3/4 external-knowledge tools."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools import knowledge_integration as mod
from mcp_server.tools.knowledge_integration import (
    compute_clinical_trials_matcher,
    compute_drug_pricing,
    compute_nih_reporter_search,
    compute_pubmed_search,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── KNOWLEDGE-1: PubMed ───────────────────────

def test_pubmed_empty_query_raises():
    with pytest.raises(ValueError, match="non-empty"):
        _run(compute_pubmed_search(""))


def test_pubmed_invalid_max_raises():
    with pytest.raises(ValueError, match="max_results"):
        _run(compute_pubmed_search("readmission", max_results=100))


def test_pubmed_abstains_on_network_failure():
    """Network-down path -> abstain explicit, no offline fabrication."""
    async def _fail(*args, **kwargs):
        raise httpx_error()
    import httpx
    def httpx_error():
        return httpx.RequestError("offline")
    import mcp_server.tools.knowledge_integration as m
    orig_search = m._entrez_search
    orig_fetch = m._entrez_fetch
    m._entrez_search = _fail  # type: ignore[assignment]
    m._entrez_fetch = _fail  # type: ignore[assignment]
    try:
        r = _run(compute_pubmed_search("hospital readmission interventions"))
    finally:
        m._entrez_search = orig_search
        m._entrez_fetch = orig_fetch
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert r.n_results == 0
    assert r.results == []
    assert "external_api_unavailable" in (r.abstain_reason or "")


def test_pubmed_live_path_with_mocked_network(monkeypatch):
    async def _fake_search(_query, _retmax):
        return ["12345678", "87654321"]

    async def _fake_fetch(pmids):
        return [
            {"pmid": "12345678", "title": "Live mock -- readmission",
             "journal": "NEJM", "pub_year": 2024,
             "authors": ["Foo A", "Bar B"],
             "abstract": "", "doi": "10.1056/foo"},
            {"pmid": "87654321", "title": "Live mock -- readmission 2",
             "journal": "JAMA", "pub_year": 2023,
             "authors": [], "abstract": "", "doi": None},
        ]

    monkeypatch.setattr(mod, "_entrez_search", _fake_search)
    monkeypatch.setattr(mod, "_entrez_fetch", _fake_fetch)
    r = _run(compute_pubmed_search("readmission", max_results=2))
    assert r.method == "entrez_live"
    assert r.n_results == 2
    assert r.results[0].pmid == "12345678"


def test_pubmed_no_match_returns_empty_live(monkeypatch):
    """Live API returned 0 hits -> entrez_live with n_results=0.

    A genuine empty live result is a fact, not synthesis: we pass it
    through as method='entrez_live' so the caller sees that the search
    actually executed and produced no matches."""
    async def _empty_search(*args, **kwargs):
        return []

    async def _empty_fetch(*args, **kwargs):
        return []

    monkeypatch.setattr(mod, "_entrez_search", _empty_search)
    monkeypatch.setattr(mod, "_entrez_fetch", _empty_fetch)
    r = _run(compute_pubmed_search("xyzzy_unknown"))
    assert r.method == "entrez_live"
    assert r.n_results == 0
    assert r.abstain_recommended is False


# ─────────────────────── KNOWLEDGE-2: ClinicalTrials.gov ───────────────────────

def test_ct_invalid_max_per_condition_raises():
    with pytest.raises(ValueError, match="max_per_condition"):
        _run(compute_clinical_trials_matcher(["heart failure"],
                                                    max_per_condition=50))


def test_ct_empty_conditions_abstains():
    r = _run(compute_clinical_trials_matcher([]))
    assert r.n_matches == 0
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert "missing_conditions" in (r.abstain_reason or "")


def test_ct_live_path_with_mock(monkeypatch):
    async def _fake_search(condition, max_results):
        return [{
            "protocolSection": {
                "identificationModule": {"nctId": "NCT12345678",
                                              "briefTitle": "Mock CHF Trial"},
                "statusModule": {"overallStatus": "RECRUITING"},
                "conditionsModule": {"conditions": ["Heart Failure"]},
                "eligibilityModule": {"minimumAge": "18 Years",
                                          "maximumAge": "85 Years",
                                          "sex": "ALL"},
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Mock Univ."}},
                "descriptionModule": {"briefSummary": "Mock summary."},
                "designModule": {"phases": ["PHASE3"]},
                "contactsLocationsModule": {
                    "locations": [{"city": "Boston", "country": "US"}]},
            }
        }]
    monkeypatch.setattr(mod, "_clinicaltrials_search", _fake_search)
    r = _run(compute_clinical_trials_matcher(["heart failure"]))
    assert r.method == "clinicaltrials_live"
    assert r.n_matches == 1
    assert r.matches[0].nct_id == "NCT12345678"
    assert r.matches[0].phase == "PHASE3"


def test_ct_abstains_on_network_failure(monkeypatch):
    async def _fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(mod, "_clinicaltrials_search", _fail)
    r = _run(compute_clinical_trials_matcher(["heart failure"]))
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert r.n_matches == 0
    assert "external_api_unavailable" in (r.abstain_reason or "")


def test_ct_filters_to_conditions_provided(monkeypatch):
    async def _fake_search(condition, max_results):
        return []
    monkeypatch.setattr(mod, "_clinicaltrials_search", _fake_search)
    r = _run(compute_clinical_trials_matcher(["heart failure", "diabetes"]))
    assert sorted(r.conditions_searched) == ["diabetes", "heart failure"]


# ─────────────────────── KNOWLEDGE-3: Drug pricing ───────────────────────

def test_pricing_invalid_input_raises():
    with pytest.raises(ValueError, match="must be a list"):
        _run(compute_drug_pricing("warfarin"))  # type: ignore[arg-type]


def test_pricing_empty_list_returns_zero():
    r = _run(compute_drug_pricing([]))
    assert r.n_drugs == 0
    assert r.total_monthly_cost_usd == 0.0


def test_pricing_warfarin_very_low_tier():
    r = _run(compute_drug_pricing(["warfarin 5 mg"]))
    assert r.drugs[0].cost_tier == "very_low"
    assert r.drugs[0].is_generic_available is True


def test_pricing_apixaban_high_tier():
    r = _run(compute_drug_pricing(["apixaban 5 mg"]))
    assert r.drugs[0].cost_tier == "high"
    assert r.drugs[0].has_patient_assistance is True


def test_pricing_unknown_drug_returns_unknown_tier():
    r = _run(compute_drug_pricing(["zorbaxin 1000 mg"]))
    assert r.drugs[0].therapeutic_class == "unknown"
    assert r.drugs[0].estimated_monthly_cost_usd == 0.0


def test_pricing_total_correct():
    r = _run(compute_drug_pricing([
        "warfarin", "lisinopril", "atorvastatin"]))
    assert r.total_monthly_cost_usd == pytest.approx(
        12.0 + 4.0 + 6.0, rel=0.01)


def test_pricing_high_cost_drugs_flagged():
    r = _run(compute_drug_pricing(["apixaban", "warfarin", "empagliflozin"]))
    assert "apixaban" in r.high_cost_drugs
    assert "empagliflozin" in r.high_cost_drugs
    assert "warfarin" not in r.high_cost_drugs


def test_pricing_normalizes_dose_strings():
    r1 = _run(compute_drug_pricing(["warfarin 5 mg"]))
    r2 = _run(compute_drug_pricing(["warfarin"]))
    assert r1.drugs[0].cost_tier == r2.drugs[0].cost_tier


# ─────────────────────── KNOWLEDGE-4: NIH RePORTER ───────────────────────

def test_reporter_invalid_max_raises():
    with pytest.raises(ValueError, match="max_results"):
        _run(compute_nih_reporter_search(["heart"], max_results=100))


def test_reporter_empty_terms_abstains():
    r = _run(compute_nih_reporter_search([]))
    assert r.n_grants == 0
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert "missing_query_terms" in (r.abstain_reason or "")


def test_reporter_live_path_with_mock(monkeypatch):
    async def _fake_search(terms, max_results):
        return [{
            "project_num": "R01HL999999",
            "project_title": "Mock HF readmission research",
            "abstract_text": "...",
            "organization": {"org_name": "Mock University"},
            "fiscal_year": 2024,
            "award_amount": 1_500_000,
            "principal_investigators": [{"first_name": "Jane",
                                              "last_name": "Doe"}],
        }]
    monkeypatch.setattr(mod, "_reporter_search", _fake_search)
    r = _run(compute_nih_reporter_search(["heart failure", "readmission"]))
    assert r.method == "reporter_live"
    assert r.n_grants == 1
    grant = r.grants[0]
    assert grant.project_number == "R01HL999999"
    assert grant.organization_name == "Mock University"
    assert "Jane Doe" in grant.pi_names


def test_reporter_abstains_on_network_failure(monkeypatch):
    async def _fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(mod, "_reporter_search", _fail)
    r = _run(compute_nih_reporter_search(["xyz"]))
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert r.n_grants == 0
    assert "external_api_unavailable" in (r.abstain_reason or "")


# ─────────────────────── Bundle registration ───────────────────────

def test_external_knowledge_bundle_registered():
    from mcp_server.tools import BUNDLES
    assert "external_knowledge" in BUNDLES
    expected = {
        "compute_pubmed_search",
        "compute_clinical_trials_matcher",
        "compute_drug_pricing",
        "compute_nih_reporter_search",
    }
    assert expected <= set(BUNDLES["external_knowledge"])
