"""Standalone smoke-test for compute_readmission_risk core logic.
Bypasses MCP/SHARP context. Loads coefficients.json + runs LACE on a stub bundle.
Confirms the W1->runtime hand-off works end-to-end.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "mcp_server" / "tools"))


# Inline-import only the helpers (skip MCP/SHARP/schemas dependencies)
def _score_los(days: float) -> int:
    if days < 1: return 0
    if days < 2: return 1
    if days < 3: return 2
    if days < 4: return 3
    if days < 7: return 4
    if days < 14: return 5
    return 7


def _score_charlson(n: int) -> int:
    if n <= 0: return 0
    if n >= 4: return 5
    return int(n)


def _rtype(entry):
    res = entry.get("resource") if isinstance(entry, dict) else None
    return str(res.get("resourceType", "")) if isinstance(res, dict) else ""


def _compute_lace_components(bundle):
    entries = bundle.get("entry") or []
    encounters = [e.get("resource", {}) for e in entries if _rtype(e) == "Encounter"]
    conditions = [e.get("resource", {}) for e in entries if _rtype(e) == "Condition"]

    los_days = 0.0
    for enc in encounters:
        cls = enc.get("class") or {}
        if isinstance(cls, dict) and cls.get("code") == "IMP":
            period = enc.get("period") or {}
            start = period.get("start", "")
            end = period.get("end", "")
            if start and end:
                ds = datetime.fromisoformat(start.replace("Z", "+00:00"))
                de = datetime.fromisoformat(end.replace("Z", "+00:00"))
                delta = (de - ds).total_seconds() / 86400.0
                if delta > los_days:
                    los_days = delta

    l_points = _score_los(los_days)
    a_raw = int(any((e.get("class") or {}).get("code") == "IMP" for e in encounters))
    a_points = 3 if a_raw else 0
    c_raw = len(conditions)
    c_points = _score_charlson(c_raw)
    e_raw = sum(1 for e in encounters if (e.get("class") or {}).get("code") != "IMP")
    e_points = min(4, e_raw)
    return {
        "l": l_points, "l_raw": int(los_days),
        "a": a_points, "a_raw": a_raw,
        "c": c_points, "c_raw": c_raw,
        "e": e_points, "e_raw": e_raw,
    }


def _make_bundle(*, los_days: int, n_conditions: int, n_ed_visits: int, is_acute: bool):
    entries = [{"resource": {"resourceType": "Patient", "id": "pt-1", "birthDate": "1960-01-01", "gender": "female"}}]
    if is_acute:
        entries.append({"resource": {"resourceType": "Encounter", "id": "enc-imp",
                                      "class": {"code": "IMP"},
                                      "period": {"start": "2025-12-01T08:00:00Z",
                                                 "end": f"2025-12-{1+los_days:02d}T11:00:00Z"}}})
    for i in range(n_ed_visits):
        entries.append({"resource": {"resourceType": "Encounter", "id": f"enc-ed-{i}", "class": {"code": "EMER"}}})
    for i in range(n_conditions):
        entries.append({"resource": {"resourceType": "Condition", "id": f"cond-{i}"}})
    return {"resourceType": "Bundle", "type": "collection", "entry": entries}


def main() -> None:
    coef_path = ROOT / "data" / "coefficients.json"
    if not coef_path.exists():
        print(f"FAIL: coefficients.json not found at {coef_path}")
        sys.exit(1)
    coef = json.loads(coef_path.read_text())
    lookup = coef["runtime_coefficients"]["lookup_table"]
    model_name = coef["model_name"]
    confidence = coef.get("confidence", "?")
    print(f"=== TrustedRisk readmission-risk smoke test ===")
    print(f"model:      {model_name}")
    print(f"version:    {coef.get('model_version')}")
    print(f"confidence: {confidence}")
    print()

    cases = [
        ("Low risk:      30yo, 1d-IMP, 1 cond, 0 ED",   _make_bundle(los_days=1, n_conditions=1, n_ed_visits=0, is_acute=True)),
        ("Medium:        65yo, 5d-IMP, 4 cond, 2 ED",   _make_bundle(los_days=5, n_conditions=4, n_ed_visits=2, is_acute=True)),
        ("High:          80yo, 14d-IMP, 5 cond, 4 ED",  _make_bundle(los_days=14, n_conditions=5, n_ed_visits=4, is_acute=True)),
        ("Elective:      70yo, 3d-non-acute, 3 cond",   _make_bundle(los_days=3, n_conditions=3, n_ed_visits=0, is_acute=False)),
    ]

    for label, bundle in cases:
        comp = _compute_lace_components(bundle)
        total = comp["l"] + comp["a"] + comp["c"] + comp["e"]
        total = max(0, min(19, total))
        entry = lookup[str(total)]
        ci = entry["prob_ci95"]
        print(f"{label}")
        print(f"  components: L={comp['l']}, A={comp['a']}, C={comp['c']}, E={comp['e']} -> total={total}")
        print(f"  posterior:  prob_mean={entry['prob_mean']:.4f}, ci95=[{ci[0]:.4f}, {ci[1]:.4f}], ci_width={entry['ci_width']:.4f}")
        print(f"  bin:        {entry['lace_bin']}")
        print()


if __name__ == "__main__":
    main()
