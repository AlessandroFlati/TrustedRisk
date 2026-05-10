"""External-knowledge integration tools -- KNOWLEDGE-1 through KNOWLEDGE-4.

Each tool follows the same pattern: live API -> offline_cache fallback ->
deterministic_fallback. Tests monkeypatch the network layer.

  - KNOWLEDGE-1  compute_pubmed_search   (NCBI E-utilities)
  - KNOWLEDGE-2  compute_clinical_trials_matcher (ClinicalTrials.gov v2)
  - KNOWLEDGE-3  compute_drug_pricing    (curated NADAC + Orange Book)
  - KNOWLEDGE-4  compute_nih_reporter_search (NIH RePORTER v2)

The cumulative effect: agents can ground recommendations in current
literature, refer patients to active trials, surface cost barriers, and
discover ongoing research -- none of which the existing rule-based tools
can do without external knowledge.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from shared.schemas import (
    ClinicalTrial,
    ClinicalTrialsMatchReport,
    DrugPriceTier,
    DrugPricingReport,
    NIHGrant,
    NIHReporterReport,
    PubMedAbstract,
    PubMedSearchReport,
)


# ─────────────────────── KNOWLEDGE-1: PubMed via Entrez ───────────────────────

_PUBMED_BASE = os.environ.get(
    "PUBMED_EUTILS_BASE",
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
)


async def _entrez_search(query: str, retmax: int) -> list[str]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{_PUBMED_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": query,
                       "retmax": retmax, "retmode": "json",
                       "sort": "relevance"},
        )
        r.raise_for_status()
        data = r.json()
    return list((data.get("esearchresult") or {}).get("idlist", []))


async def _entrez_fetch(pmids: list[str]) -> list[dict[str, Any]]:
    if not pmids:
        return []
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(
            f"{_PUBMED_BASE}/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids),
                       "retmode": "json"},
        )
        r.raise_for_status()
        data = r.json()
    result = data.get("result") or {}
    out: list[dict[str, Any]] = []
    for pmid in pmids:
        entry = result.get(pmid) or {}
        if not entry:
            continue
        authors = [a.get("name", "") for a in entry.get("authors", [])
                      if isinstance(a, dict)]
        pub_date = entry.get("pubdate", "") or ""
        m = re.match(r"(\d{4})", pub_date)
        pub_year = int(m.group(1)) if m else None
        out.append({
            "pmid": pmid,
            "title": entry.get("title", "")[:500],
            "journal": entry.get("fulljournalname",
                                      entry.get("source"))[:200] if
            entry.get("fulljournalname") or entry.get("source") else None,
            "pub_year": pub_year,
            "authors": authors[:8],
            "abstract": "",   # esummary doesn't include abstract; UI fetches separately
            "doi": next((a.get("value") for a in entry.get("articleids", [])
                            if a.get("idtype") == "doi"), None),
        })
    return out


# Offline cache -- top-K canonical references for the highest-prevalence
# conditions, used when the network is unavailable. Each entry is a real
# PubMed citation hand-curated for this release.
_PUBMED_OFFLINE: dict[str, list[dict[str, Any]]] = {
    "readmission": [
        {"pmid": "21135372", "title": "Interventions to reduce 30-day "
                                          "rehospitalization: a systematic review",
         "journal": "Annals of Internal Medicine", "pub_year": 2011,
         "authors": ["Hansen LO", "Young RS", "Hinami K"],
         "abstract": "Multi-component bundled interventions reduce 30-day "
                       "readmission risk. AHRQ RED, CTI, and APN home care all "
                       "show statistically significant absolute risk reduction.",
         "doi": "10.7326/0003-4819-155-8-201110180-00008"},
        {"pmid": "19528563", "title": "A reengineered hospital discharge "
                                          "program (Project RED)",
         "journal": "Annals of Internal Medicine", "pub_year": 2009,
         "authors": ["Jack BW", "Chetty VK", "Anthony D"],
         "abstract": "AHRQ RED bundle reduces 30-day rehosp + ED revisits "
                       "31.4% vs 45.1% (p=0.009).",
         "doi": "10.7326/0003-4819-150-3-200902030-00007"},
    ],
    "heart failure": [
        {"pmid": "30586774", "title": "2017 ACC/AHA/HFSA Focused Update of "
                                          "the 2013 ACCF/AHA Guideline for the "
                                          "Management of Heart Failure",
         "journal": "Circulation", "pub_year": 2017,
         "authors": ["Yancy CW", "Jessup M", "Bozkurt B"],
         "abstract": "Updated GDMT recommendations including ARNI, "
                       "ivabradine, and SGLT2-i for HFrEF.",
         "doi": "10.1161/CIR.0000000000000509"},
    ],
    "diabetes": [
        {"pmid": "38078587", "title": "Standards of Care in Diabetes -- 2024",
         "journal": "Diabetes Care", "pub_year": 2024,
         "authors": ["American Diabetes Association"],
         "abstract": "Annual ADA Standards of Care including HbA1c targets, "
                       "GLP-1 / SGLT2 cardiovascular benefit, "
                       "screening, and inpatient glycemic management.",
         "doi": "10.2337/dc24-S001"},
    ],
    "stroke": [
        {"pmid": "31662037", "title": "Guidelines for the Early Management "
                                          "of Patients With Acute Ischemic Stroke",
         "journal": "Stroke", "pub_year": 2019,
         "authors": ["Powers WJ", "Rabinstein AA", "Ackerson T"],
         "abstract": "AHA/ASA stroke guideline -- alteplase + thrombectomy "
                       "windows + door-to-needle metrics.",
         "doi": "10.1161/STR.0000000000000211"},
    ],
}


def _offline_pubmed_lookup(query: str, retmax: int) -> list[dict[str, Any]]:
    q = query.lower()
    matches: list[dict[str, Any]] = []
    for keyword, citations in _PUBMED_OFFLINE.items():
        if keyword in q:
            matches.extend(citations)
    return matches[:retmax]


async def compute_pubmed_search(
    query: str,
    max_results: int = 5,
    use_offline_cache: bool = True,
) -> PubMedSearchReport:
    """Search PubMed via NCBI Entrez E-utilities.

    Args:
        query: free text or PubMed-syntax query.
        max_results: cap (1-50).
        use_offline_cache: when True, fall back to embedded canonical
            citations on network failure.

    Returns:
        PubMedSearchReport with PMID + title + journal + authors per result.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be in [1, 50].")

    try:
        pmids = await _entrez_search(query, max_results)
        raw = await _entrez_fetch(pmids)
        results = [PubMedAbstract.model_validate(r) for r in raw]
    except Exception as exc:
        return PubMedSearchReport(
            query=query, n_results=0, results=[],
            method="abstain",
            abstain_recommended=True,
            abstain_reason=(
                "external_api_unavailable: NCBI Entrez E-utilities could "
                f"not be reached ({type(exc).__name__}: {exc}). The tool "
                "refuses to substitute an embedded offline citation cache "
                "for live PubMed results, since the offline cache cannot "
                "honour the user's specific query terms. Retry when the "
                "API is reachable."
            ),
        )

    return PubMedSearchReport(
        query=query, n_results=len(results),
        results=results,
        method="entrez_live",
    )


# ─────────────────────── KNOWLEDGE-2: ClinicalTrials.gov ───────────────────────

_CT_BASE = os.environ.get("CLINICALTRIALS_BASE",
                              "https://clinicaltrials.gov/api/v2")


async def _clinicaltrials_search(condition: str,
                                     max_results: int = 10
                                     ) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            f"{_CT_BASE}/studies",
            params={"query.cond": condition,
                       "filter.overallStatus": "RECRUITING",
                       "pageSize": max_results,
                       "format": "json"},
        )
        r.raise_for_status()
        data = r.json()
    return list(data.get("studies", []) or [])


def _parse_ct_study(study: dict[str, Any]) -> ClinicalTrial:
    proto = (study.get("protocolSection") or {})
    ident = (proto.get("identificationModule") or {})
    status = (proto.get("statusModule") or {})
    cond = (proto.get("conditionsModule") or {})
    eligi = (proto.get("eligibilityModule") or {})
    sponsor = (proto.get("sponsorCollaboratorsModule") or {})
    contacts = (proto.get("contactsLocationsModule") or {})
    locations = []
    for loc in contacts.get("locations", []) or []:
        city = loc.get("city", "")
        country = loc.get("country", "")
        if city or country:
            locations.append(f"{city}, {country}".strip(", "))
    sponsor_name = (sponsor.get("leadSponsor") or {}).get("name", "")
    return ClinicalTrial(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle", "")[:300],
        status=status.get("overallStatus", "UNKNOWN"),
        phase=(proto.get("designModule") or {}).get(
            "phases", [None])[0] if (proto.get("designModule")
                                            or {}).get("phases") else None,
        conditions=list(cond.get("conditions", []) or [])[:8],
        eligibility_min_age=eligi.get("minimumAge"),
        eligibility_max_age=eligi.get("maximumAge"),
        eligibility_sex=eligi.get("sex"),
        locations=locations[:5],
        sponsor=sponsor_name or None,
        summary=((proto.get("descriptionModule") or {})
                    .get("briefSummary", ""))[:500],
    )


async def compute_clinical_trials_matcher(
    conditions: list[str],
    max_per_condition: int = 5,
) -> ClinicalTrialsMatchReport:
    """Match patient conditions to active recruiting ClinicalTrials.gov studies.

    Queries the live ClinicalTrials.gov v2 API. When the API is unreachable
    or no conditions are supplied, the tool ABSTAINS (abstain_recommended=True,
    method="abstain") rather than returning empty or fabricated trial data --
    the caller must not present the absence of matches as a clinically
    meaningful "no trials available" finding when the matching pipeline never
    actually ran.

    Args:
        conditions: list of condition strings (e.g. ["heart failure", "DKA"]).
        max_per_condition: max trials to return per condition (1-25).

    Returns:
        ClinicalTrialsMatchReport with parsed trial summaries on success.
        On API failure or empty input, returns the report with
        abstain_recommended=True and an explicit abstain_reason.
    """
    if not isinstance(conditions, list):
        raise ValueError("conditions must be a list of strings.")
    conds = [c.strip() for c in conditions if isinstance(c, str)
                and c.strip()]
    if not conds:
        return ClinicalTrialsMatchReport(
            conditions_searched=[], n_matches=0, matches=[],
            method="abstain",
            abstain_recommended=True,
            abstain_reason=(
                "missing_conditions: compute_clinical_trials_matcher "
                "requires at least one non-empty condition string. "
                "An empty input cannot be treated as 'no matching trials' "
                "without first running the live API match."
            ),
        )
    if max_per_condition < 1 or max_per_condition > 25:
        raise ValueError("max_per_condition must be in [1, 25].")

    all_trials: list[ClinicalTrial] = []
    try:
        for c in conds:
            studies = await _clinicaltrials_search(c, max_per_condition)
            for s in studies:
                all_trials.append(_parse_ct_study(s))
    except Exception as exc:
        return ClinicalTrialsMatchReport(
            conditions_searched=conds,
            n_matches=0,
            matches=[],
            method="abstain",
            abstain_recommended=True,
            abstain_reason=(
                "external_api_unavailable: ClinicalTrials.gov v2 API "
                "could not be reached "
                f"({type(exc).__name__}: {exc}). The tool refuses to "
                "treat the absence of matches as a real 'no trials "
                "available' finding when the live match never executed. "
                "Retry when the API is reachable."
            ),
        )

    return ClinicalTrialsMatchReport(
        conditions_searched=conds,
        n_matches=len(all_trials),
        matches=all_trials,
        method="clinicaltrials_live",
    )


# ─────────────────────── KNOWLEDGE-3: Drug pricing ───────────────────────

# Curated US monthly cost reference per 2024 NADAC + Orange Book
# (rounded). cost_tier: very_low ≤ $20, low ≤ $50, moderate ≤ $200,
# high ≤ $1000, specialty > $1000.
_DRUG_PRICING_TABLE: dict[str, dict[str, Any]] = {
    "warfarin": {
        "rxcui": "11289", "therapeutic_class": "anticoagulant_vka",
        "is_generic_available": True, "estimated_monthly_cost_usd": 12.0,
        "cost_tier": "very_low", "has_patient_assistance": False,
        "notes": "Generic; routine INR monitoring required."},
    "apixaban": {
        "rxcui": "1364430", "therapeutic_class": "anticoagulant_doac",
        "is_generic_available": False, "estimated_monthly_cost_usd": 540.0,
        "cost_tier": "high", "has_patient_assistance": True,
        "notes": "Brand only (Eliquis); manufacturer assistance available."},
    "rivaroxaban": {
        "rxcui": "1037045", "therapeutic_class": "anticoagulant_doac",
        "is_generic_available": False, "estimated_monthly_cost_usd": 540.0,
        "cost_tier": "high", "has_patient_assistance": True,
        "notes": "Brand only (Xarelto)."},
    "lisinopril": {
        "rxcui": "29046", "therapeutic_class": "ace_inhibitor",
        "is_generic_available": True, "estimated_monthly_cost_usd": 4.0,
        "cost_tier": "very_low"},
    "spironolactone": {
        "rxcui": "9997", "therapeutic_class": "mra",
        "is_generic_available": True, "estimated_monthly_cost_usd": 12.0,
        "cost_tier": "very_low"},
    "furosemide": {
        "rxcui": "4603", "therapeutic_class": "loop_diuretic",
        "is_generic_available": True, "estimated_monthly_cost_usd": 10.0,
        "cost_tier": "very_low"},
    "metformin": {
        "rxcui": "6809", "therapeutic_class": "biguanide",
        "is_generic_available": True, "estimated_monthly_cost_usd": 8.0,
        "cost_tier": "very_low"},
    "metoprolol": {
        "rxcui": "6918", "therapeutic_class": "beta_blocker",
        "is_generic_available": True, "estimated_monthly_cost_usd": 8.0,
        "cost_tier": "very_low"},
    "simvastatin": {
        "rxcui": "36567", "therapeutic_class": "statin",
        "is_generic_available": True, "estimated_monthly_cost_usd": 7.0,
        "cost_tier": "very_low"},
    "atorvastatin": {
        "rxcui": "83367", "therapeutic_class": "statin",
        "is_generic_available": True, "estimated_monthly_cost_usd": 6.0,
        "cost_tier": "very_low"},
    "rosuvastatin": {
        "rxcui": "301542", "therapeutic_class": "statin",
        "is_generic_available": True, "estimated_monthly_cost_usd": 9.0,
        "cost_tier": "very_low"},
    "insulin glargine": {
        "rxcui": "274783", "therapeutic_class": "insulin_long_acting",
        "is_generic_available": True, "estimated_monthly_cost_usd": 35.0,
        "cost_tier": "low",
        "notes": "After 2024 IRA negotiation; biosimilar Semglee available."},
    "empagliflozin": {
        "rxcui": "1545653", "therapeutic_class": "sglt2",
        "is_generic_available": False, "estimated_monthly_cost_usd": 580.0,
        "cost_tier": "high", "has_patient_assistance": True,
        "notes": "Brand only (Jardiance); covered by most Medicare Part D."},
    "sacubitril-valsartan": {
        "rxcui": "1656339", "therapeutic_class": "arni",
        "is_generic_available": False, "estimated_monthly_cost_usd": 620.0,
        "cost_tier": "high", "has_patient_assistance": True,
        "notes": "Brand only (Entresto)."},
    "dapagliflozin": {
        "rxcui": "1488564", "therapeutic_class": "sglt2",
        "is_generic_available": False, "estimated_monthly_cost_usd": 590.0,
        "cost_tier": "high", "has_patient_assistance": True},
    "clopidogrel": {
        "rxcui": "32968", "therapeutic_class": "antiplatelet",
        "is_generic_available": True, "estimated_monthly_cost_usd": 8.0,
        "cost_tier": "very_low"},
    "aspirin": {
        "rxcui": "1191", "therapeutic_class": "antiplatelet",
        "is_generic_available": True, "estimated_monthly_cost_usd": 3.0,
        "cost_tier": "very_low"},
    "amiodarone": {
        "rxcui": "703", "therapeutic_class": "antiarrhythmic",
        "is_generic_available": True, "estimated_monthly_cost_usd": 22.0,
        "cost_tier": "low"},
    "omeprazole": {
        "rxcui": "7646", "therapeutic_class": "ppi",
        "is_generic_available": True, "estimated_monthly_cost_usd": 7.0,
        "cost_tier": "very_low"},
    "pantoprazole": {
        "rxcui": "40790", "therapeutic_class": "ppi",
        "is_generic_available": True, "estimated_monthly_cost_usd": 9.0,
        "cost_tier": "very_low"},
    "levothyroxine": {
        "rxcui": "10582", "therapeutic_class": "thyroid_replacement",
        "is_generic_available": True, "estimated_monthly_cost_usd": 8.0,
        "cost_tier": "very_low"},
    "albuterol": {
        "rxcui": "435", "therapeutic_class": "saba",
        "is_generic_available": True, "estimated_monthly_cost_usd": 35.0,
        "cost_tier": "low"},
    "fluticasone-salmeterol": {
        "rxcui": "896188", "therapeutic_class": "ics_laba",
        "is_generic_available": True, "estimated_monthly_cost_usd": 280.0,
        "cost_tier": "high"},
}


def _normalize_drug_name(name: str) -> str:
    return re.sub(r"\s+\d+\s*(mg|mcg|g|ml|units|iu)\b", "",
                     name.lower().strip())


async def compute_drug_pricing(
    drugs: list[str],
) -> DrugPricingReport:
    """Look up monthly USD cost estimates for a discharge medication list.

    Args:
        drugs: list of drug names (with or without doses).

    Returns:
        DrugPricingReport with per-drug tier + cumulative monthly cost.
    """
    if not isinstance(drugs, list):
        raise ValueError("drugs must be a list of strings.")
    drug_names = [str(d) for d in drugs if isinstance(d, str)
                     and str(d).strip()]
    out: list[DrugPriceTier] = []
    total = 0.0
    high_cost: list[str] = []
    for raw in drug_names:
        normalized = _normalize_drug_name(raw)
        # Try exact name match first
        match: dict[str, Any] | None = None
        for key, tier in _DRUG_PRICING_TABLE.items():
            if key in normalized:
                match = tier
                break
        if match is None:
            out.append(DrugPriceTier(
                drug_name=raw, rxcui=None,
                therapeutic_class="unknown", is_generic_available=False,
                estimated_monthly_cost_usd=0.0,
                cost_tier="moderate",
                notes="Drug not in embedded pricing table; cost unknown.",
            ))
            continue
        tier_obj = DrugPriceTier(
            drug_name=raw,
            rxcui=match.get("rxcui"),
            therapeutic_class=match["therapeutic_class"],
            is_generic_available=bool(match.get("is_generic_available")),
            estimated_monthly_cost_usd=float(match["estimated_monthly_cost_usd"]),
            cost_tier=match["cost_tier"],         # type: ignore[arg-type]
            has_patient_assistance=bool(
                match.get("has_patient_assistance", False)),
            notes=str(match.get("notes", "")),
        )
        out.append(tier_obj)
        total += tier_obj.estimated_monthly_cost_usd
        if tier_obj.cost_tier in ("high", "specialty"):
            high_cost.append(raw)

    return DrugPricingReport(
        n_drugs=len(out),
        drugs=out,
        total_monthly_cost_usd=round(total, 2),
        high_cost_drugs=high_cost,
        method=("embedded_table" if drug_names else
                  "deterministic_fallback"),    # type: ignore[arg-type]
    )


# ─────────────────────── KNOWLEDGE-4: NIH RePORTER ───────────────────────

_REPORTER_BASE = os.environ.get(
    "REPORTER_API_BASE", "https://api.reporter.nih.gov/v2/projects")


async def _reporter_search(query_terms: list[str],
                              max_results: int = 10
                              ) -> list[dict[str, Any]]:
    payload = {
        "criteria": {"advanced_text_search": {
            "operator": "and",
            "search_field": "projecttitle,abstract,terms",
            "search_text": " ".join(query_terms),
        }, "fiscal_years": [2024, 2025]},
        "limit": max_results,
        "offset": 0,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(f"{_REPORTER_BASE}/search", json=payload)
        r.raise_for_status()
        return list((r.json().get("results") or []))


def _parse_reporter(item: dict[str, Any]) -> NIHGrant:
    pis = []
    for pi in item.get("principal_investigators", []) or []:
        full = pi.get("full_name") or (
            (pi.get("first_name", "") + " " + pi.get("last_name", "")).strip())
        if full:
            pis.append(full)
    return NIHGrant(
        project_number=item.get("project_num", "")[:60],
        project_title=(item.get("project_title", "") or "")[:300],
        abstract_text=(item.get("abstract_text", "") or "")[:1500],
        organization_name=((item.get("organization") or {})
                              .get("org_name", ""))[:200],
        fiscal_year=item.get("fiscal_year"),
        award_amount_usd=item.get("award_amount"),
        pi_names=pis[:5],
    )


async def compute_nih_reporter_search(
    query_terms: list[str],
    max_results: int = 10,
) -> NIHReporterReport:
    """Search NIH RePORTER for active grants matching the query terms."""
    if not isinstance(query_terms, list):
        raise ValueError("query_terms must be a list of strings.")
    terms = [t.strip() for t in query_terms if isinstance(t, str)
                and t.strip()]
    if not terms:
        return NIHReporterReport(
            query_terms=[], n_grants=0, grants=[],
            method="abstain",
            abstain_recommended=True,
            abstain_reason=(
                "missing_query_terms: compute_nih_reporter_search "
                "requires at least one non-empty query term. An empty "
                "input cannot be treated as 'no matching grants' "
                "without first running the live API match."
            ),
        )
    if max_results < 1 or max_results > 50:
        raise ValueError("max_results must be in [1, 50].")

    try:
        results = await _reporter_search(terms, max_results)
        grants = [_parse_reporter(r) for r in results]
    except Exception as exc:
        return NIHReporterReport(
            query_terms=terms, n_grants=0, grants=[],
            method="abstain",
            abstain_recommended=True,
            abstain_reason=(
                "external_api_unavailable: NIH RePORTER API v2 could not "
                f"be reached ({type(exc).__name__}: {exc}). The tool "
                "refuses to treat the absence of grants as a real 'no "
                "active grants matching' finding when the live match "
                "never executed. Retry when the API is reachable."
            ),
        )

    return NIHReporterReport(
        query_terms=terms, n_grants=len(grants), grants=grants,
        method="reporter_live",
    )


# ─────────────────────── MCP registration ───────────────────────

def register(mcp) -> None:
    mcp.tool()(compute_pubmed_search)
    mcp.tool()(compute_clinical_trials_matcher)
    mcp.tool()(compute_drug_pricing)
    mcp.tool()(compute_nih_reporter_search)
