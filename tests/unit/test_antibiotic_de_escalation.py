"""Unit tests for compute_antibiotic_de_escalation."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.antibiotic_de_escalation import (
    compute_antibiotic_de_escalation,
)


def _run(coro):
    return asyncio.run(coro)


def test_pending_culture_abstains():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="piperacillin-tazobactam IV",
        pathogen=None,
        days_on_therapy=2,
    ))
    assert rep.abstain_recommended is True
    assert rep.abstain_reason == "cultures_pending"
    assert rep.de_escalation_recommended is False


def test_e_coli_amp_susceptible_de_escalates_to_amox():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="piperacillin-tazobactam 4.5g IV q8h",
        pathogen="e_coli",
        susceptibility={"ampicillin": "S"},
        days_on_therapy=3,
        clinical_factors={"afebrile_24h": True, "hemodynamically_stable": True,
                           "tolerating_po": True, "alert_oriented": True},
    ))
    assert rep.de_escalation_recommended is True
    assert "amoxicillin" in (rep.target_regimen or "").lower()
    assert rep.iv_to_po_switch_eligible is True


def test_iv_to_po_blocked_by_npo():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="ceftriaxone 1g IV q24h",
        pathogen="e_coli",
        susceptibility={"ciprofloxacin": "S"},
        days_on_therapy=3,
        clinical_factors={"afebrile_24h": True, "hemodynamically_stable": True,
                           "tolerating_po": False, "alert_oriented": True},
    ))
    assert rep.iv_to_po_switch_eligible is False
    assert any("tolerating PO" in u or "NPO" in u
                for u in rep.iv_to_po_criteria_unmet)


def test_iv_to_po_blocked_by_unstable_hemodynamics():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="vancomycin IV",
        pathogen="staphylococcus_aureus_mssa",
        susceptibility={"oxacillin": "S"},
        days_on_therapy=2,
        clinical_factors={"afebrile_24h": False, "hemodynamically_stable": False,
                           "tolerating_po": True, "alert_oriented": True},
    ))
    assert rep.iv_to_po_switch_eligible is False


def test_unknown_pathogen_abstains():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="some regimen",
        pathogen="pseudomonas_aeruginosa_some_weird_strain_xyz",
        days_on_therapy=3,
    ))
    # Pathogen not in narrowing DB
    assert rep.de_escalation_recommended is False or rep.target_regimen is None


def test_esbl_no_oral_option():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="piperacillin-tazobactam IV",
        pathogen="e_coli_esbl",
        susceptibility={"ertapenem": "S"},
        days_on_therapy=3,
        clinical_factors={"afebrile_24h": True, "hemodynamically_stable": True,
                           "tolerating_po": True, "alert_oriented": True},
    ))
    # ESBL has IV-only option in our table
    assert rep.iv_to_po_switch_eligible is False
    assert rep.target_regimen_route == "IV"


def test_susceptibility_pattern_recorded():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="ceftriaxone IV",
        pathogen="e_coli",
        susceptibility={"ampicillin": "R", "tmp_smx": "S", "ciprofloxacin": "S"},
        days_on_therapy=3,
    ))
    assert rep.susceptibility_pattern == {
        "ampicillin": "R", "tmp_smx": "S", "ciprofloxacin": "S",
    }


def test_duration_remaining_calculated():
    rep = _run(compute_antibiotic_de_escalation(
        current_regimen="ceftriaxone",
        pathogen="e_coli",
        susceptibility={"ampicillin": "S"},
        days_on_therapy=3,
        total_planned_duration_days=10,
    ))
    assert rep.duration_remaining_days == 7
    assert rep.duration_total_days == 10
