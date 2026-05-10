"""LLM-3 unit tests for compute_resolve_patient_from_query."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from mcp_server.tools import conversational_resolver as mod
from mcp_server.tools.conversational_resolver import (
    _extract_features_regex,
    compute_resolve_patient_from_query,
)


def _run(coro):
    return asyncio.run(coro)


# ─────────────────────── Regex feature extraction ───────────────────────

def test_chief_complaint_chest_pain_extracted():
    f = _extract_features_regex("75 yo M with crushing chest pain")
    assert "chest pain" in f.chief_complaint_terms
    assert f.age_min_years == 70 and f.age_max_years == 80
    assert f.sex == "male"
    assert f.confidence in ("medium", "high")


def test_just_admitted_anchors_to_six_hours():
    f = _extract_features_regex("the patient I just admitted with sepsis")
    assert f.time_anchor_hours == 6
    assert f.care_role == "admitting"
    assert "sepsis" in f.chief_complaint_terms


def test_yesterday_anchors_to_48h():
    f = _extract_features_regex("yesterday's stroke admission")
    assert f.time_anchor_hours == 48
    assert "stroke" in f.chief_complaint_terms


def test_age_only_query_low_confidence():
    f = _extract_features_regex("the 82-year-old")
    assert f.age_min_years == 77
    assert f.confidence == "low"


def test_empty_query_no_features():
    f = _extract_features_regex("")
    assert f.chief_complaint_terms == []
    assert f.confidence == "low"


def test_multiple_complaints_multiple_terms():
    f = _extract_features_regex(
        "55yo F with chest pain, dyspnea, and dizziness")
    assert {"chest pain", "shortness of breath", "syncope"} <= set(
        f.chief_complaint_terms)


def test_dka_alias_normalized():
    f = _extract_features_regex("DKA glucose 540 ketoacidosis")
    assert "dka" in f.chief_complaint_terms


def test_suicide_stem_matches_ideation():
    f = _extract_features_regex("28yo M with active suicidal ideation")
    assert "suicide ideation" in f.chief_complaint_terms


# ─────────────────────── Tool behavior with mocked FHIR ───────────────────────

class _FakeQuery:
    def __init__(self, results):
        self._results = results
        self._limit = None

    def search(self, **kwargs):
        return self

    def limit(self, n):
        self._limit = n
        return self

    async def fetch(self):
        return list(self._results)


class _FakeClient:
    def __init__(self, encounters_by_search=None,
                  patients_by_id=None):
        self._enc = encounters_by_search or []
        self._pts = patients_by_id or {}

    def resources(self, kind: str):
        if kind == "Encounter":
            return _FakeQuery(self._enc)
        if kind == "Patient":
            return _PatientByIdQuery(self._pts)
        return _FakeQuery([])


class _PatientByIdQuery:
    def __init__(self, by_id):
        self._by_id = by_id
        self._target_id = None

    def search(self, _id=None, **kwargs):
        self._target_id = _id
        return self

    def limit(self, n):
        return self

    async def fetch(self):
        if self._target_id is None:
            return []
        p = self._by_id.get(self._target_id)
        return [p] if p is not None else []


@pytest.fixture
def fhir_disabled_llm(monkeypatch):
    """Disable LLM enhancement so tests are deterministic."""
    monkeypatch.setenv("TRUSTEDRISK_DISABLE_LLM", "1")


def _enc(patient_id: str, *, complaint: str = "", start_iso: str = None):
    enc = {
        "resourceType": "Encounter",
        "id": f"enc-{patient_id}",
        "subject": {"reference": f"Patient/{patient_id}"},
        "period": {"start": start_iso or
                      datetime.now(timezone.utc).isoformat()},
        "reasonCode": [],
        "type": [],
    }
    if complaint:
        enc["reasonCode"].append(
            {"text": complaint,
             "coding": [{"display": complaint}]})
    return enc


def _patient(pid: str, age_years: int = 60, sex: str = "male"):
    today = datetime.now(timezone.utc).date()
    dob = today.replace(year=today.year - age_years)
    return {"resourceType": "Patient", "id": pid,
              "birthDate": dob.isoformat(), "gender": sex}


def test_empty_query_abstains(fhir_disabled_llm):
    r = _run(compute_resolve_patient_from_query(""))
    assert r.abstain_recommended is True
    assert r.abstain_reason == "empty_query"
    assert r.candidates == []


def test_no_candidates_abstains(fhir_disabled_llm, monkeypatch):
    async def fake_get_client():
        return _FakeClient(encounters_by_search=[], patients_by_id={})
    monkeypatch.setattr(mod, "get_fhir_client", fake_get_client)
    r = _run(compute_resolve_patient_from_query(
        "patient with chest pain admitted today"))
    assert r.abstain_recommended is True
    assert r.abstain_reason == "no_candidates_in_time_window"


def test_single_high_confidence_match_resolves(fhir_disabled_llm, monkeypatch):
    enc = _enc("pt-001", complaint="chest pain")
    pt = _patient("pt-001", age_years=58, sex="male")

    async def fake_get_client():
        return _FakeClient(encounters_by_search=[enc],
                              patients_by_id={"pt-001": pt})
    monkeypatch.setattr(mod, "get_fhir_client", fake_get_client)
    r = _run(compute_resolve_patient_from_query(
        "the 58yo M with crushing chest pain I just admitted"))
    assert r.resolved_patient_id == "pt-001"
    assert r.abstain_recommended is False
    top = r.candidates[0]
    assert top.match_score >= 0.6
    assert any("chest pain" in m or "chief_complaint" in m
                  for m in top.matched_features)
    assert any("age:58" in m for m in top.matched_features)
    assert any("sex:male" in m for m in top.matched_features)


def test_ambiguous_match_abstains(fhir_disabled_llm, monkeypatch):
    """Two very similar candidates -> abstain (no clear winner)."""
    e1 = _enc("pt-001", complaint="chest pain")
    e2 = _enc("pt-002", complaint="chest pain")
    pts = {"pt-001": _patient("pt-001", 58, "male"),
             "pt-002": _patient("pt-002", 58, "male")}

    async def fake_get_client():
        return _FakeClient(encounters_by_search=[e1, e2], patients_by_id=pts)
    monkeypatch.setattr(mod, "get_fhir_client", fake_get_client)
    r = _run(compute_resolve_patient_from_query(
        "58yo M with chest pain admitted today"))
    assert r.resolved_patient_id is None
    assert r.abstain_recommended is True
    assert r.abstain_reason == "ambiguous_top_match"
    assert r.n_candidates == 2
    # Phase 3.1 -- ambiguous matches promote to A2A INPUT_REQUIRED with
    # a candidates list so the BYO orchestrator can ask the user to pick
    assert r.task_state == "input_required"
    assert r.clarification_request is not None
    cr = r.clarification_request
    assert cr.expected_answer_kind == "patient_id"
    assert set(cr.candidates) == {"pt-001", "pt-002"}


def test_ranking_by_match_score(fhir_disabled_llm, monkeypatch):
    """Candidates returned ordered by descending score."""
    e_match = _enc("pt-good", complaint="chest pain")
    e_partial = _enc("pt-other", complaint="abdominal pain")
    pts = {
        "pt-good": _patient("pt-good", 58, "male"),
        "pt-other": _patient("pt-other", 30, "female"),
    }

    async def fake_get_client():
        return _FakeClient(encounters_by_search=[e_partial, e_match],
                              patients_by_id=pts)
    monkeypatch.setattr(mod, "get_fhir_client", fake_get_client)
    r = _run(compute_resolve_patient_from_query(
        "58yo M with chest pain"))
    assert [c.patient_id for c in r.candidates][0] == "pt-good"
    scores = [c.match_score for c in r.candidates]
    assert scores == sorted(scores, reverse=True)


def test_limit_caps_candidates(fhir_disabled_llm, monkeypatch):
    encounters = [_enc(f"pt-{i:03d}", complaint="chest pain")
                    for i in range(20)]
    pts = {f"pt-{i:03d}": _patient(f"pt-{i:03d}", 58, "male")
             for i in range(20)}

    async def fake_get_client():
        return _FakeClient(encounters_by_search=encounters,
                              patients_by_id=pts)
    monkeypatch.setattr(mod, "get_fhir_client", fake_get_client)
    r = _run(compute_resolve_patient_from_query(
        "58yo M with chest pain", limit=3))
    assert len(r.candidates) == 3


def test_time_window_override(fhir_disabled_llm, monkeypatch):
    captured: dict = {}

    class _CapturingQuery:
        def search(self, **kwargs):
            captured.update(kwargs)
            return _FakeQuery([])
        def limit(self, n): return self
        async def fetch(self): return []

    class _CapturingClient:
        def resources(self, kind: str):
            return _CapturingQuery() if kind == "Encounter" \
                else _PatientByIdQuery({})

    async def fake_get_client():
        return _CapturingClient()
    monkeypatch.setattr(mod, "get_fhir_client", fake_get_client)

    _run(compute_resolve_patient_from_query(
        "chest pain", time_window_hours=72))
    cutoff_str = captured.get("_lastUpdated", "")
    assert cutoff_str.startswith("ge")
    iso = cutoff_str[2:].replace("Z", "+00:00")
    parsed = datetime.fromisoformat(iso)
    expected = datetime.now(timezone.utc) - timedelta(hours=72)
    delta = abs((parsed - expected).total_seconds())
    assert delta < 60


def test_invalid_extraction_method_raises(fhir_disabled_llm):
    with pytest.raises(ValueError, match="include_extraction_method"):
        _run(compute_resolve_patient_from_query(
            "chest pain", include_extraction_method="hallucinate"))


def test_llm_only_mode_abstains_when_unavailable(fhir_disabled_llm):
    """With TRUSTEDRISK_DISABLE_LLM=1, llm_only mode must abstain."""
    r = _run(compute_resolve_patient_from_query(
        "chest pain", include_extraction_method="llm_only"))
    assert r.abstain_recommended is True
    assert r.abstain_reason == "llm_unavailable_in_llm_only_mode"


def test_resolver_is_registered_in_bundle():
    """Sanity: the new tool appears in the context_resolution bundle."""
    from mcp_server.tools import BUNDLES
    assert "context_resolution" in BUNDLES
    assert "compute_resolve_patient_from_query" in BUNDLES["context_resolution"]
