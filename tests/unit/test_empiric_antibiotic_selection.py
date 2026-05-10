"""Unit tests for compute_empiric_antibiotic_selection."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.empiric_antibiotic_selection import (
    compute_empiric_antibiotic_selection,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Urinary ───────────────────────

def test_uncomplicated_uti_picks_nitrofurantoin():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="urinary", severity="uncomplicated",
        patient_factors={"age": 30, "egfr_ml_min": 90},
    ))
    assert sel.top_pick_id == "nitrofurantoin_po"


def test_uncomplicated_uti_low_egfr_blocks_nitrofurantoin():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="urinary", severity="uncomplicated",
        patient_factors={"age": 75, "egfr_ml_min": 25},
    ))
    nitro = next((o for o in sel.options if o.regimen_id == "nitrofurantoin_po"), None)
    if nitro:
        assert any("eGFR" in c for c in nitro.contraindications)


def test_complicated_uti_low_esbl_picks_ceftriaxone():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="urinary", severity="complicated",
        patient_factors={"age": 65, "egfr_ml_min": 70},
        local_antibiogram={"esbl_prevalence_pct": 5},
    ))
    assert sel.top_pick_id == "ceftriaxone_iv"


def test_complicated_uti_high_esbl_picks_carbapenem():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="urinary", severity="complicated",
        patient_factors={"age": 65, "egfr_ml_min": 70, "prior_esbl_infection": True},
        local_antibiogram={"esbl_prevalence_pct": 22},
    ))
    assert sel.top_pick_id == "ertapenem_iv"


def test_local_antibiogram_used_flagged():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="urinary", severity="uncomplicated",
        local_antibiogram={"esbl_prevalence_pct": 8},
    ))
    assert sel.local_antibiogram_used is True


# ─────────────────────── Pneumonia ───────────────────────

def test_uncomplicated_cap_picks_amoxicillin_unless_allergy():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="pneumonia", severity="uncomplicated",
        patient_factors={"age": 40},
    ))
    assert sel.top_pick_id == "amoxicillin_po"


def test_pcn_allergy_skips_amoxicillin():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="pneumonia", severity="uncomplicated",
        patient_factors={"age": 40, "allergy_penicillin": True},
    ))
    amox = next((o for o in sel.options if o.regimen_id == "amoxicillin_po"), None)
    if amox:
        assert amox.score == 0.0


def test_inpatient_pneumonia_picks_combo():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="pneumonia", severity="complicated",
        patient_factors={"age": 70},
    ))
    assert sel.top_pick_id == "ceftriaxone_azithromycin"


def test_icu_pneumonia_with_mrsa_risk():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="pneumonia", severity="septic_shock",
        patient_factors={"age": 65, "mrsa_risk": True, "pseudomonas_risk": True,
                         "hap_vap": True},
    ))
    regimen_ids = [o.regimen_id for o in sel.options]
    assert "vancomycin_iv" in regimen_ids
    assert "cefepime_iv" in regimen_ids


# ─────────────────────── SSTI ───────────────────────

def test_uncomplicated_ssti_no_mrsa_picks_cephalexin():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="skin_soft_tissue", severity="uncomplicated",
        patient_factors={},
    ))
    assert sel.top_pick_id == "cephalexin_po"


def test_uncomplicated_ssti_mrsa_risk_picks_tmp_smx():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="skin_soft_tissue", severity="uncomplicated",
        patient_factors={"mrsa_risk": True},
    ))
    assert sel.top_pick_id == "tmp_smx_po"


def test_severe_ssti_picks_combo():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="skin_soft_tissue", severity="sepsis",
    ))
    assert sel.top_pick_id == "vancomycin_pip_tazo"


# ─────────────────────── Unsupported sources ───────────────────────

def test_intra_abdominal_abstains_for_now():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="intra_abdominal", severity="complicated",
    ))
    assert sel.abstain_recommended is True


def test_unknown_source_abstains():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="something_made_up",
    ))
    assert sel.abstain_recommended is True


def test_options_ranked_by_score():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="urinary", severity="complicated",
        patient_factors={"egfr_ml_min": 70, "prior_esbl_infection": True},
        local_antibiogram={"esbl_prevalence_pct": 25},
    ))
    scores = [o.score for o in sel.options]
    assert scores == sorted(scores, reverse=True)


def test_references_populated():
    sel = _run(compute_empiric_antibiotic_selection(
        infection_source="pneumonia", severity="uncomplicated",
        patient_factors={"age": 40},
    ))
    assert any("Metlay" in r or "ATS" in r or "IDSA" in r for r in sel.references)
