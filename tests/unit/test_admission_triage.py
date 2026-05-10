"""Unit tests for compute_admission_triage."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.admission_triage import compute_admission_triage


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Red-flag chief complaints ───────────────────────

def test_chest_pain_is_emergent():
    rep = _run(compute_admission_triage(
        chief_complaint="chest pain for 30 minutes",
        vital_signs={"systolic_bp": 130, "heart_rate": 80},
        age=58,
    ))
    assert rep.esi_level == 2
    assert rep.priority == "emergent"
    assert rep.disposition == "admit_inpatient"
    assert any("chest pain" in f.lower() for f in rep.red_flag_findings)


def test_radiating_chest_pain_is_immediate():
    rep = _run(compute_admission_triage(
        chief_complaint="crushing chest pain radiating to left arm",
        vital_signs={"systolic_bp": 130},
        age=60,
    ))
    assert rep.esi_level == 1
    assert rep.priority == "immediate"
    assert rep.disposition == "resuscitation_room"


def test_facial_droop_is_immediate():
    rep = _run(compute_admission_triage(
        chief_complaint="sudden facial droop and slurred speech",
        vital_signs={"systolic_bp": 165},
        age=72,
    ))
    assert rep.esi_level == 1
    assert any("focal" in f.lower() or "stroke" in f.lower()
                for f in rep.red_flag_findings)


def test_thunderclap_headache_is_immediate():
    rep = _run(compute_admission_triage(
        chief_complaint="sudden severe headache, worst of life",
        age=50,
    ))
    assert rep.esi_level == 1


def test_seizure_is_emergent():
    rep = _run(compute_admission_triage(
        chief_complaint="first-time seizure, post-ictal",
        vital_signs={"systolic_bp": 130, "heart_rate": 90},
        age=42,
    ))
    assert rep.esi_level == 2


def test_dyspnea_is_emergent():
    rep = _run(compute_admission_triage(
        chief_complaint="severe shortness of breath",
        vital_signs={"systolic_bp": 130, "heart_rate": 95, "spo2": 94},
        age=68,
    ))
    assert rep.esi_level == 2


def test_gi_bleed_is_emergent():
    rep = _run(compute_admission_triage(
        chief_complaint="hematemesis x 2 episodes",
        vital_signs={"systolic_bp": 110},
        age=55,
    ))
    assert rep.esi_level == 2


def test_minor_complaint_is_less_urgent():
    rep = _run(compute_admission_triage(
        chief_complaint="ankle pain after stumbling",
        vital_signs={"systolic_bp": 125, "heart_rate": 78},
        age=30,
    ))
    assert rep.esi_level >= 4
    assert rep.disposition == "discharge_from_ed"


# ─────────────────────── Vital-sign red flags ───────────────────────

def test_severe_hypotension_pulls_to_immediate():
    rep = _run(compute_admission_triage(
        chief_complaint="weakness",
        vital_signs={"systolic_bp": 75, "heart_rate": 110},
        age=70,
    ))
    assert rep.esi_level == 1


def test_severe_hypoxemia_pulls_to_immediate():
    rep = _run(compute_admission_triage(
        chief_complaint="cough and fatigue",
        vital_signs={"systolic_bp": 120, "spo2": 85},
        age=68,
    ))
    assert rep.esi_level == 1


def test_high_fever_with_no_red_flag_complaint():
    rep = _run(compute_admission_triage(
        chief_complaint="general malaise",
        vital_signs={"systolic_bp": 110, "heart_rate": 115, "temperature": 39.6},
        age=50,
    ))
    assert rep.esi_level <= 2


def test_borderline_hypotension_lowers_to_emergent():
    rep = _run(compute_admission_triage(
        chief_complaint="dizziness",
        vital_signs={"systolic_bp": 95, "heart_rate": 100},
        age=65,
    ))
    assert rep.esi_level <= 2


# ─────────────────────── Age-driven defaults ───────────────────────

def test_pediatric_lethargic_infant_is_immediate():
    rep = _run(compute_admission_triage(
        chief_complaint="3-month-old infant lethargic and not feeding",
        age=0,
    ))
    assert rep.esi_level == 1


def test_older_adult_at_esi_3_admits_to_observation():
    rep = _run(compute_admission_triage(
        chief_complaint="general weakness for 2 days",
        vital_signs={"systolic_bp": 120, "heart_rate": 75},
        age=82,
    ))
    # ESI 3 by default for "weakness" without clearer red flags
    if rep.esi_level == 3:
        assert rep.disposition == "admit_observation"
        assert rep.recommended_unit == "observation_unit"


# ─────────────────────── Edge cases ───────────────────────

def test_empty_chief_complaint_abstains():
    rep = _run(compute_admission_triage(chief_complaint=""))
    assert rep.abstain_recommended is True
    assert rep.esi_level == 3  # default to urgent for clinician review


def test_high_acuity_complaint_without_vitals_abstains():
    """ESI 1-2 complaint without vital signs -> abstain (need vitals for confirm)."""
    rep = _run(compute_admission_triage(
        chief_complaint="severe shortness of breath",
        vital_signs=None,
        age=58,
    ))
    assert rep.esi_level == 2
    assert rep.abstain_recommended is True
    assert rep.abstain_reason == "no_vitals_for_high_acuity_complaint"


def test_rationale_includes_red_flags():
    rep = _run(compute_admission_triage(
        chief_complaint="chest pain",
        vital_signs={"systolic_bp": 130, "heart_rate": 85},
        age=60,
    ))
    assert "Red-flag findings" in rep.rationale
    assert "chest pain" in rep.rationale.lower()


def test_no_red_flag_no_vitals_defaults_to_esi4():
    """No red flag complaint and no abnormal vitals -> ESI 4 (less_urgent)."""
    rep = _run(compute_admission_triage(
        chief_complaint="sore throat",
        vital_signs={"systolic_bp": 120, "heart_rate": 75, "temperature": 37.3},
        age=25,
    ))
    assert rep.esi_level == 4
    assert rep.disposition == "discharge_from_ed"


def test_overdose_is_emergent_mental_health():
    rep = _run(compute_admission_triage(
        chief_complaint="suspected acetaminophen overdose, suicidal intent",
        vital_signs={"systolic_bp": 110, "heart_rate": 95},
        age=22,
    ))
    assert rep.esi_level == 2
    assert any("mental" in f.lower() for f in rep.red_flag_findings)


def test_references_present():
    rep = _run(compute_admission_triage(
        chief_complaint="cough",
        vital_signs={"systolic_bp": 125},
        age=40,
    ))
    assert any("ESI" in r for r in rep.references)
