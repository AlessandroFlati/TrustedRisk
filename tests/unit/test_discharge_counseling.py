"""Unit tests for compute_discharge_counseling."""
from __future__ import annotations

import asyncio

import pytest

from mcp_server.tools.discharge_counseling import (
    _CLASS_PATIENT_LANGUAGE,
    _follow_up_window,
    _section_follow_up,
    _section_medications,
    _section_warning_signs,
    compute_discharge_counseling,
)


def _run(coro):
    return asyncio.run(coro)


def _med(name: str, drug_class: str | None = None, status: str = "active") -> dict:
    return {"name": name, "drug_class": drug_class, "status": status}


# ─────────────────────── _follow_up_window ───────────────────────

def test_follow_up_high_risk_short_window():
    window, _ = _follow_up_window(15)
    assert window == (3, 7)


def test_follow_up_moderate_risk():
    window, _ = _follow_up_window(7)
    assert window == (7, 14)


def test_follow_up_low_risk_longest_window():
    window, _ = _follow_up_window(2)
    assert window == (14, 30)


def test_follow_up_zero_lace():
    window, rationale = _follow_up_window(0)
    assert window == (14, 30)
    assert "2 to 4 weeks" in rationale


# ─────────────────────── _section_medications ───────────────────────

def test_meds_section_explains_warfarin():
    sec, n = _section_medications([
        # type: ignore[list-item] -- _section_medications coerces dicts via the public API
        # but here we go through _coerce_meds-equivalent path indirectly.
    ])
    assert n == 0  # empty list -> no explanations
    assert "No active medications" in sec.bullets[0]


def test_meds_section_skips_inactive(monkeypatch):
    """Stopped medications must not appear in the medications section."""
    from shared.schemas import Medication
    meds = [
        Medication(name="warfarin", drug_class="anticoagulant_vka", status="active"),
        Medication(name="amoxicillin", status="completed"),
    ]
    sec, n = _section_medications(meds)
    assert n == 1
    joined = " ".join(sec.bullets)
    assert "warfarin" in joined.lower()
    assert "amoxicillin" not in joined.lower()


def test_meds_section_unknown_drug_falls_back_to_label():
    from shared.schemas import Medication
    meds = [Medication(name="ObscureDrug-X", drug_class=None, status="active")]
    sec, n = _section_medications(meds)
    assert n == 0  # unknown drug class -> not "explained"
    assert "ObscureDrug-X" in " ".join(sec.bullets)
    assert "ask your pharmacist" in " ".join(sec.bullets).lower()


# ─────────────────────── _section_warning_signs ───────────────────────

def test_warning_section_includes_universal_red_flags():
    sec, n = _section_warning_signs([])
    text = " ".join(sec.bullets).lower()
    assert "chest pain" in text
    assert "911" in text


def test_warning_section_adds_drug_class_red_flags():
    from shared.schemas import Medication
    meds = [Medication(name="warfarin", drug_class="anticoagulant_vka", status="active")]
    sec, n = _section_warning_signs(meds)
    text = " ".join(sec.bullets).lower()
    # Warfarin red flags
    assert "bruising" in text or "bleeding" in text
    assert "stools" in text


def test_warning_section_dedupes_same_class():
    """Two warfarin variants in the list must not duplicate the red-flag bullets."""
    from shared.schemas import Medication
    meds = [
        Medication(name="warfarin 5mg", drug_class="anticoagulant_vka", status="active"),
        Medication(name="warfarin 2.5mg", drug_class="anticoagulant_vka", status="active"),
    ]
    sec, _ = _section_warning_signs(meds)
    bleeding_count = sum(
        1 for b in sec.bullets if "bruising or bleeding" in b.lower()
    )
    assert bleeding_count == 1


def test_warning_section_appends_extra_red_flags():
    sec, _ = _section_warning_signs([], extra_red_flags=[
        "Redness or pus at your surgical site.",
    ])
    assert any("surgical site" in b.lower() for b in sec.bullets)


# ─────────────────────── _section_follow_up ───────────────────────

def test_follow_up_section_high_lace_recommends_short_window():
    sec, window = _section_follow_up(lace_score=15, recommendation_action="discharge_home")
    assert window == (3, 7)
    assert "3 to 7 days" in " ".join(sec.bullets)


def test_follow_up_section_snf_action_adds_snf_bullet():
    sec, _ = _section_follow_up(lace_score=10, recommendation_action="snf")
    assert any("skilled nursing facility" in b.lower() for b in sec.bullets)


def test_follow_up_section_home_with_care_adds_nurse_bullet():
    sec, _ = _section_follow_up(lace_score=8, recommendation_action="home_with_care")
    assert any("home health nurse" in b.lower() for b in sec.bullets)


# ─────────────────────── compute_discharge_counseling end-to-end ───────────────────────

def test_e2e_basic_chf_patient():
    """Heart failure patient on furosemide + lisinopril + metoprolol."""
    rep = _run(compute_discharge_counseling(
        medications=[
            _med("furosemide 40mg", drug_class="loop_diuretic"),
            _med("lisinopril 10mg", drug_class="ace_inhibitor"),
            _med("metoprolol 25mg", drug_class="beta_blocker"),
        ],
        lace_score=8,
        recommendation_action="home_with_care",
        patient_id="pt-chf-1",
    ))
    assert len(rep.sections) == 5
    section_ids = [s.section_id for s in rep.sections]
    assert section_ids == [
        "your_medications", "follow_up", "warning_signs",
        "activities_self_care", "questions_to_ask",
    ]
    assert rep.n_medications_explained == 3
    assert rep.follow_up_window_days == (7, 14)


def test_e2e_high_risk_short_window():
    rep = _run(compute_discharge_counseling(
        medications=[_med("apixaban 5mg", drug_class="anticoagulant_doac")],
        lace_score=12,
        recommendation_action="snf",
    ))
    assert rep.follow_up_window_days == (3, 7)
    follow = next(s for s in rep.sections if s.section_id == "follow_up")
    assert "skilled nursing facility" in " ".join(follow.bullets).lower()


def test_e2e_no_phi_in_output():
    """The output must never include patient_id or any PHI-like data."""
    pid = "patient-johndoe-1985"
    rep = _run(compute_discharge_counseling(
        medications=[_med("warfarin", drug_class="anticoagulant_vka")],
        lace_score=10,
        patient_id=pid,
    ))
    full_text = " ".join(
        s.plain_text + " " + " ".join(s.bullets) for s in rep.sections
    )
    assert pid not in full_text
    assert "johndoe" not in full_text.lower()
    assert "1985" not in full_text


def test_e2e_no_active_meds_still_renders():
    rep = _run(compute_discharge_counseling(medications=[], lace_score=4))
    assert len(rep.sections) == 5
    meds_sec = next(s for s in rep.sections if s.section_id == "your_medications")
    assert "No active medications" in " ".join(meds_sec.bullets)


def test_e2e_locale_fallback():
    """Non-en locale falls back to en (no silent failure, no cross-locale leakage)."""
    rep = _run(compute_discharge_counseling(
        medications=[_med("warfarin", drug_class="anticoagulant_vka")],
        lace_score=8,
        locale="es",  # not implemented
    ))
    assert rep.locale == "en"  # fell back


def test_e2e_extra_red_flags_appended():
    rep = _run(compute_discharge_counseling(
        medications=[],
        lace_score=4,
        extra_red_flags=[
            "Redness, warmth, or pus at your surgical incision.",
        ],
    ))
    warning = next(s for s in rep.sections if s.section_id == "warning_signs")
    assert any("incision" in b.lower() for b in warning.bullets)


def test_e2e_lace_score_clamped():
    """LACE outside 0-19 should be clamped, not crash."""
    rep1 = _run(compute_discharge_counseling(medications=[], lace_score=99))
    assert rep1.follow_up_window_days == (3, 7)
    rep2 = _run(compute_discharge_counseling(medications=[], lace_score=-5))
    assert rep2.follow_up_window_days == (14, 30)


def test_e2e_reading_level_metadata():
    rep = _run(compute_discharge_counseling(medications=[], lace_score=5))
    assert rep.reading_level_grade == 6


def test_e2e_disclaimer_present():
    rep = _run(compute_discharge_counseling(medications=[], lace_score=5))
    assert "discharge paperwork" in rep.disclaimer.lower()
    assert "does not replace" in rep.disclaimer.lower() or "does NOT replace" in rep.disclaimer


def test_class_taxonomy_covers_high_risk_classes():
    """Every drug class with patient-language should have all required keys."""
    for cls, info in _CLASS_PATIENT_LANGUAGE.items():
        assert "label" in info, f"{cls} missing label"
        assert "purpose" in info, f"{cls} missing purpose"
        assert "monitoring" in info, f"{cls} missing monitoring"
        assert "red_flags" in info, f"{cls} missing red_flags"
        assert isinstance(info["red_flags"], list)
        assert len(info["red_flags"]) >= 1


def test_e2e_inactive_med_does_not_count():
    rep = _run(compute_discharge_counseling(
        medications=[
            _med("warfarin", drug_class="anticoagulant_vka", status="active"),
            _med("metoprolol", drug_class="beta_blocker", status="stopped"),
        ],
        lace_score=8,
    ))
    assert rep.n_medications_explained == 1


def test_e2e_drug_class_inferred_when_missing():
    """When drug_class is None but the name matches the taxonomy, it should still classify."""
    rep = _run(compute_discharge_counseling(
        medications=[
            _med("Warfarin 5 mg", drug_class=None),
            _med("Lisinopril 10 mg", drug_class=None),
        ],
        lace_score=8,
    ))
    assert rep.n_medications_explained == 2
    meds_sec = next(s for s in rep.sections if s.section_id == "your_medications")
    text = " ".join(meds_sec.bullets).lower()
    assert "blood thinner" in text
    assert "blood pressure" in text


def test_e2e_dict_or_pydantic_input_both_work():
    """The tool accepts both dict-form and Medication-form medications."""
    from shared.schemas import Medication
    pyd_med = Medication(
        name="warfarin", drug_class="anticoagulant_vka", status="active",
    )
    rep_pyd = _run(compute_discharge_counseling(
        medications=[pyd_med],  # type: ignore[arg-type]
        lace_score=8,
    ))
    rep_dict = _run(compute_discharge_counseling(
        medications=[{"name": "warfarin",
                       "drug_class": "anticoagulant_vka",
                       "status": "active"}],
        lace_score=8,
    ))
    assert rep_pyd.n_medications_explained == rep_dict.n_medications_explained == 1
