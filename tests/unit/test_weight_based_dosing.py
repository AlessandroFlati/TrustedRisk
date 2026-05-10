"""Unit tests for compute_weight_based_dosing."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.weight_based_dosing import compute_weight_based_dosing


def _run(coro):
    return asyncio.run(coro)


def test_amoxicillin_typical_toddler():
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=12.0, age_months=14,
        indication="acute_otitis_media",
    ))
    assert rep.calculated_dose_mg > 0
    assert rep.final_dose_mg <= 1000  # adult cap
    assert rep.volume_to_administer_ml is not None
    assert rep.abstain_recommended is False


def test_amoxicillin_capped_at_adult_max():
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=40.0, age_months=120,
        indication="strep_pharyngitis",
    ))
    assert rep.capped_at_adult_max is True
    assert rep.final_dose_mg == 1000.0


def test_penicillin_allergy_blocks_amoxicillin():
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=20.0, age_months=60,
        allergy_classes=["penicillin"],
    ))
    assert rep.abstain_recommended is True
    assert any("allerg" in c.lower() for c in rep.contraindications)


def test_ibuprofen_under_6_months_contraindicated():
    rep = _run(compute_weight_based_dosing(
        drug="ibuprofen", weight_kg=5.0, age_months=4,
        indication="fever",
    ))
    assert rep.abstain_recommended is True
    assert any("contraindication" in c.lower() for c in rep.contraindications)


def test_ceftriaxone_neonate_contraindicated():
    rep = _run(compute_weight_based_dosing(
        drug="ceftriaxone", weight_kg=3.5, age_months=0,
        indication="bacteremia", route="IV",
    ))
    assert rep.abstain_recommended is True
    assert any("28 days" in c or "neonate" in c.lower() or "bilirubin" in c.lower()
                for c in rep.contraindications)


def test_unknown_drug_abstains():
    rep = _run(compute_weight_based_dosing(
        drug="totally-fake-medication", weight_kg=20.0, age_months=60,
    ))
    assert rep.abstain_recommended is True
    assert "drug_not_in_database" in (rep.abstain_reason or "")


def test_unsupported_route_blocks():
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=12.0, age_months=24,
        route="IV",  # amoxicillin is PO-only in our DB
    ))
    assert rep.abstain_recommended is True
    assert any("route" in c.lower() for c in rep.contraindications)


def test_acetaminophen_pediatric_dose():
    rep = _run(compute_weight_based_dosing(
        drug="acetaminophen", weight_kg=15.0, age_months=36,
        indication="fever",
    ))
    # Midpoint of 10-15 mg/kg = 12.5; 12.5 * 15 = 187.5 mg
    assert 100 <= rep.final_dose_mg <= 250
    assert rep.volume_to_administer_ml is not None


def test_target_dose_above_max_blocks():
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=15.0, age_months=36,
        target_dose_mg_per_kg=120.0,  # above 90 mg/kg max
    ))
    assert rep.abstain_recommended is True
    assert any("exceed" in c.lower() for c in rep.contraindications)


def test_epinephrine_anaphylaxis():
    rep = _run(compute_weight_based_dosing(
        drug="epinephrine_im", weight_kg=20.0, age_months=60,
        indication="anaphylaxis", route="IM",
    ))
    # 0.01 mg/kg * 20 kg = 0.2 mg, under 0.5 cap
    assert rep.final_dose_mg == pytest.approx(0.2, rel=0.01)
    assert rep.capped_at_adult_max is False


def test_volume_calculation():
    """For amoxicillin 50 mg/mL: 600 mg dose -> 12 mL."""
    rep = _run(compute_weight_based_dosing(
        drug="amoxicillin", weight_kg=10.0, age_months=24,
        target_dose_mg_per_kg=60.0,
    ))
    # 60 * 10 = 600 mg; 600 / 50 = 12 mL
    assert rep.volume_to_administer_ml == 12.0
