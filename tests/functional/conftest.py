"""Shared fixtures for functional tests.

Functional tests exercise CROSS-TOOL workflows on realistic clinical inputs.
Distinct from unit tests (single-tool boundaries) and integration tests
(live external services). Fixtures here build canonical patient bundles,
DecisionCards, and counseling documents that downstream test files reuse.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


# ─────────────────────── Async runner ───────────────────────

@pytest.fixture(scope="session")
def run_async():
    def _runner(coro):
        return asyncio.run(coro)
    return _runner


# ─────────────────────── LLM disabled by default ───────────────────────

@pytest.fixture(autouse=True)
def _disable_llm(monkeypatch):
    """All functional tests run on the deterministic floor by default.
    Tests that need LLM behavior monkeypatch _call_ollama explicitly."""
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


# ─────────────────────── Coefficients path ───────────────────────

@pytest.fixture(scope="session", autouse=True)
def _coefficients_path(tmp_path_factory):
    """Ensure TRUSTEDRISK_COEFFICIENTS_PATH points at the repo's data dir."""
    import os
    repo_root = Path(__file__).resolve().parent.parent.parent
    coef = repo_root / "data" / "coefficients.json"
    if coef.exists():
        os.environ.setdefault("TRUSTEDRISK_COEFFICIENTS_PATH", str(coef))


# ─────────────────────── Canonical FHIR Bundle: CHF discharge ───────────────────────

def _make_encounter(*, kind: str, start_iso: str, end_iso: str | None = None,
                       diag_codes: list[str] | None = None,
                       enc_id: str | None = None,
                       reason_text: str | None = None) -> dict:
    enc = {
        "resourceType": "Encounter",
        "id": enc_id or f"enc-{kind}-{start_iso[:10]}",
        "status": "finished" if end_iso else "in-progress",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                     "code": kind},
        "period": {"start": start_iso} if not end_iso
        else {"start": start_iso, "end": end_iso},
    }
    if reason_text:
        enc["reasonCode"] = [{"text": reason_text,
                                "coding": [{"display": reason_text}]}]
    if diag_codes:
        enc["diagnosis"] = [{"condition": {"reference": f"Condition/{c}"}}
                                for c in diag_codes]
    return enc


def _make_condition(code: str, *, system: str = "icd10cm",
                       text: str | None = None) -> dict:
    sys_url = ("http://hl7.org/fhir/sid/icd-10-cm" if system == "icd10cm"
               else "http://snomed.info/sct")
    return {
        "resourceType": "Condition",
        "id": f"cond-{code.replace('.', '')}",
        "code": {"coding": [{"system": sys_url, "code": code,
                                "display": text or code}],
                    "text": text or code},
    }


def _make_observation(*, loinc: str, value: float, unit: str = "",
                          when_iso: str | None = None,
                          obs_id: str | None = None) -> dict:
    return {
        "resourceType": "Observation",
        "id": obs_id or f"obs-{loinc}-{(when_iso or 'na')[:10]}",
        "status": "final",
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc}]},
        "valueQuantity": {"value": value, "unit": unit},
        "effectiveDateTime": when_iso or datetime.now(timezone.utc).isoformat(),
    }


def _make_med_request(*, name: str, rxcui: str | None = None) -> dict:
    coding: list[dict] = []
    if rxcui:
        coding.append({"system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                          "code": rxcui, "display": name})
    return {
        "resourceType": "MedicationRequest",
        "id": f"medreq-{name.replace(' ', '_').lower()}",
        "status": "active", "intent": "order",
        "medicationCodeableConcept": {"coding": coding, "text": name},
    }


@pytest.fixture
def chf_bundle() -> dict:
    """75-year-old female with CHF + DM + AKI, recent inpatient encounter."""
    today = datetime.now(timezone.utc)
    last_admit_start = (today - timedelta(days=10)).isoformat()
    last_admit_end = (today - timedelta(days=4)).isoformat()
    prior_ed = (today - timedelta(days=90)).isoformat()
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "pt-chf-001",
                "gender": "female", "birthDate": "1950-03-15",
                "address": [{"postalCode": "02118"}]}},
            {"resource": _make_encounter(kind="IMP",
                                              start_iso=last_admit_start,
                                              end_iso=last_admit_end,
                                              reason_text="acute decompensated heart failure")},
            {"resource": _make_encounter(kind="EMER",
                                              start_iso=prior_ed,
                                              end_iso=prior_ed,
                                              reason_text="dyspnea")},
            {"resource": _make_condition("I50.9", text="Heart failure unspecified")},
            {"resource": _make_condition("E11.9", text="Type 2 diabetes")},
            {"resource": _make_condition("I10", text="Essential hypertension")},
            {"resource": _make_condition("N17.9", text="Acute kidney injury")},
            {"resource": _make_observation(loinc="2160-0", value=1.8,
                                                unit="mg/dL",
                                                when_iso=last_admit_end)},
            {"resource": _make_observation(loinc="4548-4", value=8.4,
                                                unit="%",
                                                when_iso=last_admit_end)},
            {"resource": _make_observation(loinc="2823-3", value=4.6,
                                                unit="mmol/L",
                                                when_iso=last_admit_end)},
            {"resource": _make_observation(loinc="8480-6", value=144,
                                                unit="mm[Hg]",
                                                when_iso=last_admit_end)},
            {"resource": _make_observation(loinc="8462-4", value=86,
                                                unit="mm[Hg]",
                                                when_iso=last_admit_end)},
            {"resource": _make_med_request(name="warfarin 5 mg",
                                                rxcui="11289")},
            {"resource": _make_med_request(name="lisinopril 10 mg",
                                                rxcui="29046")},
            {"resource": _make_med_request(name="furosemide 40 mg",
                                                rxcui="4603")},
            {"resource": _make_med_request(name="metoprolol 50 mg",
                                                rxcui="6918")},
            {"resource": _make_med_request(name="metformin 500 mg",
                                                rxcui="6809")},
        ],
    }


# ─────────────────────── DKA bundle ───────────────────────

@pytest.fixture
def dka_bundle() -> dict:
    """28yo M with DKA -- first inpatient encounter."""
    today = datetime.now(timezone.utc)
    admit = (today - timedelta(days=3)).isoformat()
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "pt-dka-001",
                "gender": "male", "birthDate": "1996-07-04",
                "address": [{"postalCode": "60601"}]}},
            {"resource": _make_encounter(kind="EMER", start_iso=admit,
                                              reason_text="diabetic ketoacidosis")},
            {"resource": _make_condition("E10.10",
                                              text="Type 1 diabetes with ketoacidosis")},
            {"resource": _make_observation(loinc="2345-7", value=480,
                                                unit="mg/dL",
                                                when_iso=admit)},
            {"resource": _make_observation(loinc="2823-3", value=3.2,
                                                unit="mmol/L",
                                                when_iso=admit)},
            {"resource": _make_observation(loinc="2744-1", value=7.10,
                                                unit="{pH}",
                                                when_iso=admit)},
            {"resource": _make_observation(loinc="1925-7", value=11,
                                                unit="mmol/L",
                                                when_iso=admit)},
        ],
    }


# ─────────────────────── Pediatric bundle ───────────────────────

@pytest.fixture
def pediatric_bundle() -> dict:
    """4-month-old infant in ED, fever + lethargy."""
    today = datetime.now(timezone.utc)
    admit = (today - timedelta(hours=4)).isoformat()
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "pt-peds-001",
                "gender": "male",
                "birthDate": (today - timedelta(days=120)).date().isoformat()}},
            {"resource": _make_encounter(kind="EMER", start_iso=admit,
                                              reason_text="fever and lethargy")},
            {"resource": _make_observation(loinc="8310-5", value=39.4,
                                                unit="Cel", when_iso=admit)},
            {"resource": _make_observation(loinc="8867-4", value=180,
                                                unit="/min", when_iso=admit)},
            {"resource": _make_observation(loinc="9279-1", value=42,
                                                unit="/min", when_iso=admit)},
            {"resource": _make_observation(loinc="59408-5", value=92,
                                                unit="%", when_iso=admit)},
        ],
    }


# ─────────────────────── DecisionCard scaffolds ───────────────────────

@pytest.fixture
def chf_decision_card() -> dict:
    """A DecisionCard built for the CHF patient with home_with_care
    recommendation and the typical 4-tool trace."""
    return {
        "recommendation": {"action": "home_with_care", "confidence": "medium"},
        "reasoning": {
            "risk_estimate": {
                "model_name": "lace-plus-bayesian-v1",
                "model_version": "0.7.0",
                "outcome_id": "readmission_30d",
                "horizon_days": 30,
                "lace_raw_score": 12,
                "probability_mean": 0.34,
                "probability_ci95": [0.29, 0.39],
                "probability_ci_width": 0.10,
                "contributing_factors": [
                    {"name": "LACE_length_of_stay", "raw_value": 6,
                     "lace_points": 3, "weight": 0.25},
                    {"name": "LACE_acuity", "raw_value": 1,
                     "lace_points": 3, "weight": 0.25},
                    {"name": "LACE_comorbidity", "raw_value": 4,
                     "lace_points": 4, "weight": 0.30},
                    {"name": "LACE_ed_visits_6mo", "raw_value": 1,
                     "lace_points": 2, "weight": 0.20},
                ],
                "fhir_observations_used": ["obs-A", "obs-B", "obs-C"],
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "confidence": "preferred",
                "valid_for_minutes": 60,
                "valid_until": (datetime.now(timezone.utc)
                                  + timedelta(hours=1)).isoformat(),
            },
            "utility_analysis": {
                "dominant_action": "home_with_care",
                "dominance_confidence": 0.78,
            },
        },
        "validation": {
            "grounding": {
                "claim_text": "patient stable for home with care",
                "sub_claims": [{
                    "evidence_sources": [{
                        "source_id": "ACC-2017-1",
                        "source_type": "guideline_passage",
                        "excerpt": "GDMT for HFrEF includes ARNI/SGLT2-i...",
                    }]
                }],
                "overall_verdict": "supported",
            },
            "phi_check": {"risk_level": "none"},
        },
        "audit": {"request_id": "req-chf-001", "tools": [
            {"tool": "compute_readmission_risk"},
            {"tool": "compute_medication_reconciliation"},
            {"tool": "compute_discharge_counseling"},
        ]},
        "abstain": [],
        "self_critique": {"verdict": "approved", "rationale": "consistent",
                              "critic_role": "structural"},
    }


# ─────────────────────── Counseling document ───────────────────────

@pytest.fixture
def chf_counseling() -> dict:
    return {
        "patient_id": "pt-chf-001", "locale": "en", "reading_level_grade": 6,
        "sections": [
            {"section_id": "your_medications",
             "title": "Your medications",
             "plain_text": "You are starting warfarin to thin your blood + "
                              "lisinopril for blood pressure + furosemide for "
                              "fluid + metformin for blood sugar.",
             "bullets": [
                 "Warfarin 5 mg by mouth once a day in the evening.",
                 "Lisinopril 10 mg by mouth once a day in the morning.",
                 "Furosemide 40 mg by mouth twice a day.",
                 "Metoprolol 50 mg by mouth twice a day.",
                 "Metformin 500 mg by mouth twice a day with meals.",
             ]},
            {"section_id": "follow_up", "title": "Follow up",
             "plain_text": "See your primary care doctor in 7 days.",
             "bullets": ["Call PCP within 7 days for a heart-failure check."]},
            {"section_id": "warning_signs", "title": "Warning signs",
             "plain_text": "Call your nurse or 911 if any of these happen.",
             "bullets": [
                 "Trouble breathing that does not improve with rest.",
                 "Chest pain that does not go away.",
                 "Sudden severe headache or confusion.",
                 "Unusual bruising or bleeding that won't stop.",
             ]},
        ],
        "follow_up_window_days": [7, 14],
        "n_medications_explained": 5, "n_red_flags": 4,
    }


# ─────────────────────── HEDIS-eligible cohort ───────────────────────

@pytest.fixture
def hedis_eligible_cohort_bundle() -> dict:
    """A patient eligible for multiple HEDIS measures (60yo F with DM + HTN)."""
    today = datetime.now(timezone.utc)
    last_a1c = (today - timedelta(days=85)).isoformat()
    last_bp = (today - timedelta(days=14)).isoformat()
    return {
        "resourceType": "Bundle", "type": "collection",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "pt-hedis-001",
                            "gender": "female", "birthDate": "1965-04-12",
                            "address": [{"postalCode": "10027"}]}},
            {"resource": _make_condition("E11.9")},
            {"resource": _make_condition("I10")},
            {"resource": _make_observation(loinc="4548-4", value=7.4,
                                                unit="%", when_iso=last_a1c)},
            {"resource": _make_observation(loinc="8480-6", value=132,
                                                when_iso=last_bp)},
            {"resource": _make_observation(loinc="8462-4", value=82,
                                                when_iso=last_bp)},
            {"resource": {
                "resourceType": "Procedure",
                "id": "proc-mammo-1",
                "code": {"text": "Mammography"},
                "performedDateTime": (today - timedelta(days=300)).isoformat(),
            }},
            {"resource": {
                "resourceType": "Immunization",
                "id": "imm-flu-1",
                "vaccineCode": {"coding": [{
                    "system": "http://hl7.org/fhir/sid/cvx",
                    "code": "150",
                }]},
                "occurrenceDateTime": (today - timedelta(days=200)).isoformat(),
            }},
        ],
    }


# ─────────────────────── Realistic risk estimate ───────────────────────

@pytest.fixture
def chf_risk_estimate() -> dict:
    return {
        "model_name": "lace-plus-bayesian-v1",
        "model_version": "0.7.0",
        "outcome_id": "readmission_30d",
        "horizon_days": 30,
        "lace_raw_score": 12,
        "probability_mean": 0.34,
        "probability_ci95": [0.29, 0.39],
        "probability_ci_width": 0.10,
        "contributing_factors": [],
        "fhir_observations_used": [],
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "confidence": "preferred",
        "valid_for_minutes": 60,
        "valid_until": (datetime.now(timezone.utc)
                          + timedelta(hours=1)).isoformat(),
    }


# ─────────────────────── Sample HL7 v2 ADT message ───────────────────────

@pytest.fixture
def hl7v2_adt_a01() -> str:
    return (
        "MSH|^~\\&|EPIC|HOSPITAL|RECEIVER|FACILITY|20240429120000||"
        "ADT^A01|MSG12345|P|2.5\r"
        "PID|1||MRN12345^^^HOSPITAL^MR||DOE^JANE||19550101|F\r"
        "PV1|1|I|2N^201^A^HOSPITAL|||||||||||||||VISIT12345"
        "|||||||||||||||||||||||||||20240429120000\r"
        "DG1|1|I10|I50.9^Heart failure|Heart failure\r"
        "DG1|2|I10|E11.9^Type 2 diabetes|Type 2 diabetes\r"
        "AL1|1|DA|^Penicillin|MO|Hives\r"
    )


# ─────────────────────── Discharge summary free-text ───────────────────────

@pytest.fixture
def chf_discharge_summary_text() -> str:
    return (
        "75-year-old female with history of CHF, hypertension, and type 2 "
        "diabetes admitted for acute decompensated heart failure. "
        "Hospital course was complicated by AKI stage 1 which has resolved. "
        "Patient denies chest pain on discharge. "
        "Discharge meds: warfarin 5 mg daily, lisinopril 10 mg daily, "
        "furosemide 40 mg twice daily, metoprolol 50 mg twice daily, "
        "metformin 500 mg twice daily. "
        "NKDA. "
        "Disposition: discharge home with home health nursing referral. "
        "Follow up with PCP in 7 days. "
        "Defer to attending judgment regarding cardiology referral timing."
    )
