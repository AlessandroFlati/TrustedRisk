"""Unit tests for aki_kdigo_stage + dialysis_initiation_decision."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.aki_kdigo_stage import compute_aki_kdigo_stage
from mcp_server.tools.dialysis_initiation_decision import (
    compute_dialysis_initiation_decision,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── KDIGO staging ───────────────────────

def test_no_aki_when_stable():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=1.05,
    ))
    assert rep.aki_stage == "no_aki"


def test_stage_1_creatinine_change():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=1.6,
    ))
    assert rep.aki_stage == "stage_1"


def test_stage_2_doubling():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=2.4,
    ))
    assert rep.aki_stage == "stage_2"
    assert rep.nephrology_consult_indicated is True


def test_stage_3_tripling():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=3.5,
    ))
    assert rep.aki_stage == "stage_3"


def test_stage_3_at_4_mg_dl():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=3.5, creatinine_current_mg_dl=4.2,
    ))
    assert rep.aki_stage == "stage_3"


def test_stage_2_uop_criterion():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=1.05,
        urine_output_ml_per_kg_per_hour=0.4, urine_output_window_hours=12,
    ))
    assert rep.aki_stage == "stage_2"


def test_etiology_pre_renal_low_fena():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=1.6,
        fena_pct=0.5,
    ))
    assert rep.aki_etiology_clue == "pre_renal"


def test_etiology_post_renal_hydronephrosis():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=2.5,
        hydronephrosis_present=True,
    ))
    assert rep.aki_etiology_clue == "post_renal"


def test_nephrotoxin_review_when_meds_present():
    rep = _run(compute_aki_kdigo_stage(
        creatinine_baseline_mg_dl=1.0, creatinine_current_mg_dl=1.5,
        nephrotoxic_medications_present=True,
    ))
    assert rep.nephrotoxin_review_recommended is True


# ─────────────────────── Dialysis initiation ───────────────────────

def test_emergent_acidosis():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="stage_3", ph=7.05,
    ))
    assert rep.dialysis_indicated is True
    assert rep.urgency == "emergent_immediately"
    assert any(i.startswith("A:") for i in rep.aeiou_indications_met)


def test_emergent_hyperkalemia():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="stage_3", potassium_meq_l=7.0,
        refractory_hyperkalemia=True,
    ))
    assert rep.dialysis_indicated is True
    assert any(i.startswith("E:") for i in rep.aeiou_indications_met)


def test_dialyzable_toxin_indicates():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="no_aki", dialyzable_toxin="methanol",
    ))
    assert rep.dialysis_indicated is True
    assert any("toxin" in i.lower() or i.startswith("I:")
                for i in rep.aeiou_indications_met)


def test_uremic_pericarditis_indicates():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="stage_3", uremic_pericarditis=True,
    ))
    assert rep.dialysis_indicated is True
    assert any(i.startswith("U:") for i in rep.aeiou_indications_met)


def test_hemodynamic_instability_picks_crrt():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="stage_3", ph=7.05, hemodynamically_unstable=True,
    ))
    assert rep.modality_suggested == "crrt"


def test_stable_picks_intermittent_hd():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="stage_3", ph=7.05, hemodynamically_unstable=False,
    ))
    assert rep.modality_suggested == "intermittent_hd"


def test_stage_3_alone_starrt_aki_defers():
    rep = _run(compute_dialysis_initiation_decision(
        aki_stage="stage_3",
    ))
    assert rep.dialysis_indicated is False
    assert rep.urgency == "non_urgent"
    assert rep.abstain_recommended is True
    assert "STARRT-AKI" in (rep.abstain_reason or "")


def test_no_aki_no_indications_no_dialysis():
    rep = _run(compute_dialysis_initiation_decision(aki_stage="no_aki"))
    assert rep.dialysis_indicated is False
    assert rep.urgency == "non_urgent"
