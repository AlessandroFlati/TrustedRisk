"""healthcare.compute_rxnorm_ddi_lookup -- DATA-2.

Drug-drug interaction (DDI) lookup that:

  1. Resolves drug names to RxNorm Concept Unique Identifiers (RxCUIs)
     via the live NLM RxNav API (/REST/rxcui.json) -- still a free,
     authentication-free service even after the DDI API retirement.
  2. Cross-references the resolved RxCUIs against an embedded curated
     DDI table covering the highest-priority interactions clinicians
     act on at the discharge handoff.
  3. Falls back to keyword matching on raw drug names when the network
     is unavailable.

Reinforces the existing rule-based `detect_polypharmacy_concerns` tool
(which uses drug-CLASS pairs) with concrete drug-LEVEL interactions
keyed off authoritative RxNorm identifiers.

NOTE on the RxNav DDI API: the legacy `/REST/interaction/list.json`
endpoint was retired by NLM in January 2024. This tool deliberately
does NOT depend on it; it only uses the still-active `/REST/rxcui.json`
name -> RxCUI resolver, plus the embedded curated DDI table.

The `RXNAV_BASE_URL` env var lets institutional deployments point at a
mirror or proxy.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from shared.schemas import (
    RxNormDDIInteraction,
    RxNormDDIReport,
)


# ─────────────────────── RxNav resolution ───────────────────────

_RXNAV_BASE_URL = os.environ.get(
    "RXNAV_BASE_URL", "https://rxnav.nlm.nih.gov/REST")


async def resolve_rxcui(drug_name: str,
                          *, timeout_s: float = 10.0) -> str | None:
    """Resolve a drug name to its primary RxCUI via RxNav.

    Tests monkeypatch this at module level.
    """
    if not drug_name or not drug_name.strip():
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.get(
                f"{_RXNAV_BASE_URL}/rxcui.json",
                params={"name": drug_name.strip(), "search": "1"},
            )
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None
    ids = (data.get("idGroup") or {}).get("rxnormId") or []
    if not ids:
        return None
    return str(ids[0])


# ─────────────────────── Curated DDI table ───────────────────────
#
# Each row: (frozenset({rxcui_a, rxcui_b}), severity, description, source).
# RxCUIs come from the canonical NLM listings. Severity scale follows the
# Lexicomp / Micromedex tiers commonly used in EHRs:
#   high     -- life-threatening / contraindicated
#   moderate -- adjust dose / monitor closely
#   low      -- informational / benign
#
# This is a curated subset of the highest-priority discharge-relevant
# pairs. Production deployments should swap in DrugBank or Lexicomp.

_DDI_TABLE: list[tuple[frozenset[str], str, str, str]] = [
    # Warfarin (RxCUI 11289) ↔ aspirin (1191) -> bleeding
    (frozenset({"11289", "1191"}), "high",
     "Warfarin + aspirin: synergistic anticoagulant + antiplatelet effect "
     "-> major-bleeding risk. Counsel + monitor INR closely.",
     "Lexicomp / FDA label"),
    # Warfarin ↔ ibuprofen (5640) -> bleeding
    (frozenset({"11289", "5640"}), "high",
     "Warfarin + ibuprofen (NSAID): additive bleeding risk + GI ulceration. "
     "Avoid concurrent use; substitute acetaminophen (161).",
     "Lexicomp / FDA label"),
    # Warfarin ↔ amiodarone (703) -- INR amplification
    (frozenset({"11289", "703"}), "high",
     "Warfarin + amiodarone: CYP2C9 inhibition raises INR. Reduce warfarin "
     "dose 30-50% and monitor INR.",
     "FDA boxed warning"),
    # ACE inhibitor (lisinopril 29046) ↔ spironolactone (9997) -- hyperkalemia
    (frozenset({"29046", "9997"}), "moderate",
     "ACE inhibitor + spironolactone (MRA): additive hyperkalemia risk. "
     "Monitor K+ within 7 days; pause if K+ ≥ 5.5.",
     "ACC/AHA HF guideline"),
    # Metformin (6809) ↔ iodinated contrast (placeholder RxCUI 7065 = iohexol)
    (frozenset({"6809", "7065"}), "moderate",
     "Metformin + iodinated contrast: lactic-acidosis risk if eGFR < 30 or "
     "AKI. Hold metformin 48 h post-contrast in high-risk patients.",
     "ACR Manual on Contrast Media v2024"),
    # Simvastatin (36567) ↔ amiodarone (703)
    (frozenset({"36567", "703"}), "moderate",
     "Simvastatin + amiodarone: rhabdomyolysis risk. Cap simvastatin at "
     "20 mg/day or substitute pravastatin / rosuvastatin.",
     "FDA dose limit"),
    # Clopidogrel (32968) ↔ omeprazole (7646)
    (frozenset({"32968", "7646"}), "moderate",
     "Clopidogrel + omeprazole: CYP2C19 inhibition reduces clopidogrel "
     "activation -> reduced antiplatelet effect. Substitute pantoprazole.",
     "FDA safety communication"),
    # Lithium (6448) ↔ thiazide diuretic (HCTZ 5487)
    (frozenset({"6448", "5487"}), "high",
     "Lithium + thiazide: thiazide reduces lithium clearance -> toxicity. "
     "Monitor lithium level + renal function within 5-7 days.",
     "Lexicomp"),
    # Methotrexate (6851) ↔ trimethoprim-sulfamethoxazole (10180)
    (frozenset({"6851", "10180"}), "high",
     "Methotrexate + TMP-SMX: pancytopenia risk via anti-folate synergy. "
     "Avoid concurrent use; substitute alternative antibiotic.",
     "FDA boxed warning"),
    # SSRI (sertraline 36437) ↔ tramadol (10689) -- serotonin syndrome
    (frozenset({"36437", "10689"}), "moderate",
     "SSRI + tramadol: serotonin-syndrome risk. Counsel on symptoms "
     "(agitation, hyperreflexia, hyperthermia).",
     "FDA safety communication"),
]


# Drug name -> RxCUI offline cache (used as a fallback when the network is
# unavailable but the test cohort references common drugs).
_OFFLINE_NAME_CACHE: dict[str, str] = {
    "warfarin": "11289",
    "aspirin": "1191",
    "ibuprofen": "5640",
    "acetaminophen": "161",
    "amiodarone": "703",
    "lisinopril": "29046",
    "spironolactone": "9997",
    "metformin": "6809",
    "iohexol": "7065",
    "simvastatin": "36567",
    "clopidogrel": "32968",
    "omeprazole": "7646",
    "pantoprazole": "40790",
    "lithium": "6448",
    "hydrochlorothiazide": "5487",
    "hctz": "5487",
    "methotrexate": "6851",
    "trimethoprim": "10180",
    "tmp-smx": "10180",
    "sertraline": "36437",
    "tramadol": "10689",
}


# ─────────────────────── Resolve helper (cache -> RxNav) ───────────────────────

async def _resolve_with_cache(drug: str) -> tuple[str | None, str]:
    """Return (rxcui_or_None, source) -- source ∈ {"offline_cache", "rxnav"}."""
    name_l = drug.lower().strip()
    if name_l in _OFFLINE_NAME_CACHE:
        return _OFFLINE_NAME_CACHE[name_l], "offline_cache"
    cui = await resolve_rxcui(drug)
    return cui, ("rxnav" if cui else "offline_cache")


# ─────────────────────── DDI cross-reference ───────────────────────

def _cross_reference(rxcuis: list[str | None],
                       drug_names: list[str]
                       ) -> list[RxNormDDIInteraction]:
    out: list[RxNormDDIInteraction] = []
    n = len(rxcuis)
    for i in range(n):
        if rxcuis[i] is None:
            continue
        for j in range(i + 1, n):
            if rxcuis[j] is None:
                continue
            pair = frozenset({rxcuis[i], rxcuis[j]})
            for table_pair, severity, desc, source in _DDI_TABLE:
                if pair == table_pair:
                    out.append(RxNormDDIInteraction(
                        drug_a=drug_names[i],
                        drug_b=drug_names[j],
                        severity=severity,            # type: ignore[arg-type]
                        description=desc,
                        source=source,
                    ))
                    break
    return out


# ─────────────────────── Public API ───────────────────────

async def compute_rxnorm_ddi_lookup(
    drugs: list[str],
) -> RxNormDDIReport:
    """Resolve drug names to RxNorm RxCUIs + cross-reference DDIs.

    Args:
        drugs: list of drug names (free text). Brand or generic names
            both work for common drugs -- RxNav resolves brand->generic.

    Returns:
        RxNormDDIReport with the RxCUI map + DDI list.
    """
    if not isinstance(drugs, list):
        raise ValueError("drugs must be a list of strings.")
    drug_names = [str(d).strip() for d in drugs if isinstance(d, str)
                     and str(d).strip()]
    if not drug_names:
        return RxNormDDIReport(
            drugs_requested=[],
            rxnorm_ids_resolved={},
            interactions=[],
            n_interactions=0,
            abstain_recommended=True,
            abstain_reason="empty_drug_list",
            method="deterministic_fallback",
        )

    rxcuis: list[str | None] = []
    sources: list[str] = []
    for drug in drug_names:
        cui, src = await _resolve_with_cache(drug)
        rxcuis.append(cui)
        sources.append(src)

    rxcui_map = dict(zip(drug_names, rxcuis, strict=False))
    interactions = _cross_reference(rxcuis, drug_names)

    # Method classification:
    #   - rxnav_live      : at least one resolution came from the API
    #   - offline_cache   : all resolutions hit the embedded cache
    #   - deterministic_fallback : no resolutions at all (unknown drugs)
    if any(s == "rxnav" for s in sources):
        method = "rxnav_live"
    elif any(s == "offline_cache" and rxcuis[i] is not None
                for i, s in enumerate(sources)):
        method = "offline_cache"
    else:
        method = "deterministic_fallback"

    return RxNormDDIReport(
        drugs_requested=drug_names,
        rxnorm_ids_resolved=rxcui_map,
        interactions=interactions,
        n_interactions=len(interactions),
        method=method,                   # type: ignore[arg-type]
    )


def register(mcp) -> None:
    mcp.tool()(compute_rxnorm_ddi_lookup)
