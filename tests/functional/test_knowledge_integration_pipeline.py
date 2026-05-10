"""Functional tests for the external-knowledge pipeline.

Chains: PubMed RAG + ClinicalTrials.gov + drug pricing + NIH RePORTER.
All four use mockable HTTP layers so the suite runs offline.
"""
from __future__ import annotations

import asyncio


# ─────────────────────── PubMed ───────────────────────

def test_pubmed_abstains_when_entrez_offline(monkeypatch):
    from mcp_server.tools import knowledge_integration as ki

    async def _fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(ki, "_entrez_search", _fail)
    monkeypatch.setattr(ki, "_entrez_fetch", _fail)
    r = asyncio.run(ki.compute_pubmed_search(
        "hospital readmission interventions"))
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert r.n_results == 0


def test_pubmed_live_path_with_mocked_search(monkeypatch):
    from mcp_server.tools import knowledge_integration as ki

    async def _search(_query, _max):
        return ["12345", "23456"]

    async def _fetch(pmids):
        return [{"pmid": p, "title": f"Article {p}",
                    "journal": "NEJM", "pub_year": 2024,
                    "authors": ["Foo A"], "abstract": "",
                    "doi": "10.1056/x"} for p in pmids]

    monkeypatch.setattr(ki, "_entrez_search", _search)
    monkeypatch.setattr(ki, "_entrez_fetch", _fetch)
    r = asyncio.run(ki.compute_pubmed_search("readmission", max_results=2))
    assert r.method == "entrez_live"
    assert r.n_results == 2


# ─────────────────────── ClinicalTrials.gov ───────────────────────

def test_clinical_trials_abstains_on_network_failure(monkeypatch):
    from mcp_server.tools import knowledge_integration as ki

    async def _fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(ki, "_clinicaltrials_search", _fail)
    r = asyncio.run(ki.compute_clinical_trials_matcher(
        ["heart failure"]))
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert r.n_matches == 0


def test_clinical_trials_live_path_with_mock(monkeypatch):
    from mcp_server.tools import knowledge_integration as ki

    async def _search(condition, max_results):
        return [{"protocolSection": {
            "identificationModule": {"nctId": "NCT-X-1",
                                          "briefTitle": "Mock Trial"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "conditionsModule": {"conditions": [condition]},
            "eligibilityModule": {"minimumAge": "18 Years",
                                       "maximumAge": "85 Years",
                                       "sex": "ALL"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Mock University"}},
            "descriptionModule": {"briefSummary": "Mock summary."},
            "designModule": {"phases": ["PHASE3"]},
            "contactsLocationsModule": {"locations": [
                {"city": "Boston", "country": "US"}]},
        }}]

    monkeypatch.setattr(ki, "_clinicaltrials_search", _search)
    r = asyncio.run(ki.compute_clinical_trials_matcher(
        ["heart failure", "diabetes"]))
    assert r.method == "clinicaltrials_live"
    assert r.n_matches == 2


# ─────────────────────── Drug pricing ───────────────────────

def test_drug_pricing_chf_cohort_total():
    from mcp_server.tools.knowledge_integration import compute_drug_pricing
    r = asyncio.run(compute_drug_pricing([
        "warfarin 5 mg", "lisinopril 10 mg",
        "atorvastatin 40 mg", "apixaban 5 mg",
    ]))
    assert r.n_drugs == 4
    assert r.total_monthly_cost_usd > 0
    # Apixaban is high-cost
    assert "apixaban 5 mg" in r.high_cost_drugs


def test_drug_pricing_unknown_drug_marked():
    from mcp_server.tools.knowledge_integration import compute_drug_pricing
    r = asyncio.run(compute_drug_pricing(["zorbaxin"]))
    assert r.drugs[0].therapeutic_class == "unknown"


# ─────────────────────── NIH RePORTER ───────────────────────

def test_nih_reporter_live_path_with_mock(monkeypatch):
    from mcp_server.tools import knowledge_integration as ki

    async def _search(terms, max_results):
        return [{
            "project_num": "R01HL999999",
            "project_title": "Mock CHF research",
            "abstract_text": "...",
            "organization": {"org_name": "Mock Univ."},
            "fiscal_year": 2024, "award_amount": 1_500_000,
            "principal_investigators": [
                {"first_name": "Jane", "last_name": "Doe"}],
        }]

    monkeypatch.setattr(ki, "_reporter_search", _search)
    r = asyncio.run(ki.compute_nih_reporter_search(
        ["heart failure", "readmission"]))
    assert r.method == "reporter_live"
    assert r.n_grants == 1


def test_nih_reporter_abstains_offline(monkeypatch):
    from mcp_server.tools import knowledge_integration as ki

    async def _fail(*args, **kwargs):
        raise RuntimeError("offline")
    monkeypatch.setattr(ki, "_reporter_search", _fail)
    r = asyncio.run(ki.compute_nih_reporter_search(["heart failure"]))
    assert r.method == "abstain"
    assert r.abstain_recommended is True
    assert r.n_grants == 0


# ─────────────────────── End-to-end knowledge chain ───────────────────────

def test_full_knowledge_chain_for_chf_patient(monkeypatch):
    """For a CHF patient with all live APIs reachable: PubMed (live mock)
    + drug pricing (offline curated table) + ClinicalTrials (live mock)
    + NIH RePORTER (live mock). All four surfaces produce well-formed
    reports."""
    from mcp_server.tools import knowledge_integration as ki

    # PubMed -> live mock (abstain when entrez offline is the abstain
    # contract; here we exercise the success path).
    async def _pm_search(_q, _max):
        return ["12345"]

    async def _pm_fetch(_pmids):
        return [{"pmid": "12345", "title": "Live CHF article",
                    "journal": "NEJM", "pub_year": 2024,
                    "authors": ["Foo A"], "abstract": "",
                    "doi": "10.1056/x"}]
    monkeypatch.setattr(ki, "_entrez_search", _pm_search)
    monkeypatch.setattr(ki, "_entrez_fetch", _pm_fetch)

    # CT.gov -> mocked live
    async def _ct_search(condition, max_results):
        return [{"protocolSection": {
            "identificationModule": {"nctId": "NCT-X-CHF",
                                          "briefTitle": "CHF Trial"},
            "statusModule": {"overallStatus": "RECRUITING"},
            "conditionsModule": {"conditions": [condition]},
            "eligibilityModule": {"minimumAge": "18 Years",
                                       "maximumAge": "85 Years",
                                       "sex": "ALL"},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": "Mock Hosp"}},
            "descriptionModule": {"briefSummary": "."},
        }}]
    monkeypatch.setattr(ki, "_clinicaltrials_search", _ct_search)

    # NIH RePORTER -> mocked live
    async def _rep_search(terms, max_results):
        return [{"project_num": "R01HL", "project_title": "CHF research",
                   "abstract_text": "...",
                   "organization": {"org_name": "Mock Univ"},
                   "fiscal_year": 2024, "award_amount": 1_000_000,
                   "principal_investigators": []}]
    monkeypatch.setattr(ki, "_reporter_search", _rep_search)

    pubmed = asyncio.run(ki.compute_pubmed_search("heart failure"))
    trials = asyncio.run(ki.compute_clinical_trials_matcher(
        ["heart failure"]))
    pricing = asyncio.run(ki.compute_drug_pricing([
        "warfarin", "lisinopril", "apixaban"]))
    grants = asyncio.run(ki.compute_nih_reporter_search(
        ["heart failure"]))

    assert pubmed.method == "entrez_live"
    assert pubmed.n_results == 1
    assert trials.method == "clinicaltrials_live"
    assert trials.n_matches == 1
    assert pricing.n_drugs == 3
    assert grants.method == "reporter_live"
    assert grants.n_grants == 1
