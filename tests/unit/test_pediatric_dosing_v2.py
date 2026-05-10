"""Phase 12.3 D2 -- pediatric dosing v2 expansion (12->50+ drugs)."""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.weight_based_dosing import (
    _DRUG_DB,
    compute_weight_based_dosing,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────
# Drug DB shape
# ─────────────────────────────────────────────────────────────────────

def test_drug_db_size_at_least_50():
    assert len(_DRUG_DB) >= 50


def test_drug_db_covers_emergency_drugs():
    """Resuscitation kit drugs must be present."""
    for d in ("epinephrine_im", "atropine_iv", "naloxone_iv",
                  "dextrose_d10w", "calcium_gluconate_iv",
                  "sodium_bicarbonate_iv", "magnesium_sulfate_iv"):
        assert d in _DRUG_DB, f"Missing emergency drug {d!r}"


def test_drug_db_covers_seizure_drugs():
    for d in ("midazolam_iv", "lorazepam_iv", "diazepam_pr",
                  "phenobarbital_iv", "levetiracetam_iv",
                  "fosphenytoin_iv"):
        assert d in _DRUG_DB, f"Missing seizure drug {d!r}"


def test_drug_db_covers_rsi_drugs():
    for d in ("rocuronium_iv", "succinylcholine_iv", "etomidate_iv",
                  "fentanyl_iv", "ketamine_iv"):
        assert d in _DRUG_DB, f"Missing RSI drug {d!r}"


def test_drug_db_covers_antimicrobial_classes():
    for d in ("amoxicillin", "amoxicillin_clavulanate", "cephalexin",
                  "ceftriaxone", "clindamycin", "trimethoprim_sulfa",
                  "doxycycline", "metronidazole", "fluconazole"):
        assert d in _DRUG_DB, f"Missing antimicrobial {d!r}"


# ─────────────────────────────────────────────────────────────────────
# Functional smokes on new drugs
# ─────────────────────────────────────────────────────────────────────

def test_clindamycin_dose_within_range():
    out = _run(compute_weight_based_dosing(
        drug="clindamycin", weight_kg=20, age_months=60,
        indication="ssti", route="PO",
    ))
    # 10-40 mg/kg/day -> for a 20kg child ~ 200-800 mg/day
    assert out.final_dose_mg >= 100.0
    assert out.final_dose_mg <= 900.0


def test_cefdinir_blocked_under_6_months():
    out = _run(compute_weight_based_dosing(
        drug="cefdinir", weight_kg=6, age_months=4,
        indication="acute_otitis_media",
    ))
    # Should abstain or surface contraindication
    assert out.abstain_recommended or out.contraindications


def test_rocuronium_dose_at_typical_RSI_paralysis():
    out = _run(compute_weight_based_dosing(
        drug="rocuronium_iv", weight_kg=15, age_months=36,
        indication="RSI_paralysis", route="IV",
    ))
    # 0.6-1.2 mg/kg -> 9-18 mg
    assert 8.0 <= out.final_dose_mg <= 19.0


def test_naloxone_first_dose_for_overdose():
    out = _run(compute_weight_based_dosing(
        drug="naloxone_iv", weight_kg=30, age_months=120,
        indication="opioid_overdose", route="IV",
    ))
    # 0.01-0.1 mg/kg -> 0.3-3 mg
    assert 0.2 <= out.final_dose_mg <= 3.5


def test_sulfa_allergy_blocks_tmp_smx():
    out = _run(compute_weight_based_dosing(
        drug="trimethoprim_sulfa", weight_kg=20, age_months=72,
        indication="uti", allergy_classes=["sulfa"],
    ))
    assert out.abstain_recommended or out.contraindications


def test_diphenhydramine_blocked_under_2_years_per_fda():
    out = _run(compute_weight_based_dosing(
        drug="diphenhydramine", weight_kg=10, age_months=18,
        indication="allergic_reaction",
    ))
    assert out.abstain_recommended or out.contraindications


def test_etomidate_blocked_under_12_months():
    out = _run(compute_weight_based_dosing(
        drug="etomidate_iv", weight_kg=8, age_months=8,
        indication="RSI_induction", route="IV",
    ))
    assert out.abstain_recommended or out.contraindications


# ─────────────────────────────────────────────────────────────────────
# Adult cap respected on a heavy adolescent
# ─────────────────────────────────────────────────────────────────────

def test_morphine_capped_at_adult_max_for_heavy_adolescent():
    out = _run(compute_weight_based_dosing(
        drug="morphine_iv", weight_kg=80, age_months=180,
        indication="pain_severe", route="IV",
    ))
    # Adult cap applied; scope of returned dose should not exceed adult max
    spec = _DRUG_DB["morphine_iv"]
    assert out.final_dose_mg <= (spec.adult_max_mg or 1e9)


# ─────────────────────────────────────────────────────────────────────
# DB consistency
# ─────────────────────────────────────────────────────────────────────

def test_every_drug_has_min_le_max():
    for name, spec in _DRUG_DB.items():
        assert spec.dose_mg_per_kg_min <= spec.dose_mg_per_kg_max, (
            f"{name}: min ({spec.dose_mg_per_kg_min}) > max "
            f"({spec.dose_mg_per_kg_max})"
        )


def test_every_drug_has_at_least_one_route():
    for name, spec in _DRUG_DB.items():
        assert len(spec.routes) >= 1, f"{name}: no routes"


def test_every_drug_has_at_least_one_indication():
    for name, spec in _DRUG_DB.items():
        assert len(spec.indications) >= 1, f"{name}: no indications"
