"""Unit tests for compute_chemo_dose_adjustment."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.chemo_dose_adjustment import (
    compute_chemo_dose_adjustment,
)


def _run(coro):
    return asyncio.run(coro)


def test_full_dose_when_all_normal():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="carboplatin_pemetrexed", cycle_number=4,
        egfr_ml_min=80, anc_per_ul=2500, platelets_per_ul=200_000,
        ecog_performance_status=1,
    ))
    assert rep.decision == "proceed_full_dose"
    assert rep.dose_reduction_pct == 0.0


def test_neutropenia_delays():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="carboplatin_pemetrexed", cycle_number=4,
        egfr_ml_min=80, anc_per_ul=1000, platelets_per_ul=200_000,
    ))
    assert rep.decision in ("delay_one_week", "hold_until_recovery")


def test_severe_neutropenia_holds():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="folfox", cycle_number=3,
        egfr_ml_min=80, anc_per_ul=400, platelets_per_ul=150_000,
    ))
    assert rep.decision == "hold_until_recovery"


def test_thrombocytopenia_delays():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="r_chop", cycle_number=2,
        egfr_ml_min=80, anc_per_ul=2000, platelets_per_ul=70_000,
    ))
    assert rep.decision == "delay_one_week"


def test_cisplatin_low_egfr_discontinues():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="cisplatin_etoposide", cycle_number=2,
        egfr_ml_min=25, anc_per_ul=2000, platelets_per_ul=200_000,
    ))
    assert rep.decision == "discontinue_consider_alternative"
    assert rep.dose_reduction_pct == 100.0


def test_cisplatin_moderate_renal_reduction():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="cisplatin_etoposide", cycle_number=2,
        egfr_ml_min=45, anc_per_ul=2000, platelets_per_ul=200_000,
    ))
    assert rep.decision == "proceed_reduced_dose"
    assert rep.dose_reduction_pct >= 25.0


def test_doxorubicin_high_bili_held():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="ac_t", cycle_number=2,
        egfr_ml_min=80, bilirubin_mg_dl=6.0,
        anc_per_ul=2000, platelets_per_ul=200_000,
    ))
    assert rep.decision == "discontinue_consider_alternative"


def test_doxorubicin_moderate_bili_reduced():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="ac_t", cycle_number=2,
        egfr_ml_min=80, bilirubin_mg_dl=2.0,
        anc_per_ul=2000, platelets_per_ul=200_000,
    ))
    assert rep.decision == "proceed_reduced_dose"


def test_ecog_3_discontinues():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="paclitaxel_weekly", cycle_number=3,
        egfr_ml_min=80, anc_per_ul=2000, platelets_per_ul=200_000,
        ecog_performance_status=3,
    ))
    assert rep.decision == "discontinue_consider_alternative"


def test_unknown_regimen_abstains():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="totally_made_up_regimen_xyz", cycle_number=1,
        egfr_ml_min=80, anc_per_ul=2000, platelets_per_ul=200_000,
    ))
    assert rep.abstain_recommended is True


def test_growth_factor_indicated_for_high_fn_regimen():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="ac_t", cycle_number=1,
        egfr_ml_min=80, anc_per_ul=2500, platelets_per_ul=200_000,
        ecog_performance_status=0,
    ))
    assert rep.growth_factor_indicated is True


def test_growth_factor_after_severe_neutropenia():
    rep = _run(compute_chemo_dose_adjustment(
        regimen="folfox", cycle_number=3,
        egfr_ml_min=80, anc_per_ul=400, platelets_per_ul=150_000,
    ))
    # ANC < 500 in cycle 3 should indicate growth factor support
    assert rep.growth_factor_indicated is True
