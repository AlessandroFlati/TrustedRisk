"""Parse MIMIC-IV demo 2.2 admissions into a LACE-feature CSV ready for
external_validation.py.

MIMIC-IV demo schema (`data/mimic-iv-demo/...-2.2/hosp/`):
  admissions.csv.gz       -- one row per (subject_id, hadm_id) with admittime,
                            dischtime, admission_type, race, insurance
  patients.csv.gz         -- anchor_age (age at first admission anchor year)
  diagnoses_icd.csv.gz    -- ICD codes per admission (used for Charlson proxy)
  transfers.csv.gz        -- ED visits inferable when careunit contains "ED"

LACE feature mapping (van Walraven 2010):
  L (Length of stay)      -- dischtime - admittime in days
  A (Acuity)              -- admission_type ∈ {URGENT, EMERGENT, ...} -> 3 pts
  C (Charlson comorbidity)-- count of distinct ICD chapters mapped to Charlson
  E (ED visits 6mo)       -- count of prior ED admissions in the 180d before

30-day readmission outcome:
  Look for any subsequent admission within 30 days of dischtime; flag = 1.

Output: data/mimic_iv_lace_cohort.csv with columns:
  patient_id, age, race, insurance, lace_los, lace_acuity, lace_charlson,
  lace_ed_visits_6mo, lace_total, outcome_30d_readmit
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

MIMIC_BASE = ROOT / "data" / "mimic-iv-demo" / "mimic-iv-clinical-database-demo-2.2" / "hosp"


# ─────────────────────────────────────────────────────────────────────
# Feature scoring
# ─────────────────────────────────────────────────────────────────────

def _score_los(days: float) -> int:
    if days < 1: return 0
    if days < 2: return 1
    if days < 3: return 2
    if days < 4: return 3
    if days < 7: return 4
    if days < 14: return 5
    return 7


def _score_charlson_count(n: int) -> int:
    if n <= 0: return 0
    if n >= 4: return 5
    return n


# ICD-10 chapter prefixes that count toward Charlson comorbidities
# (rough mapping -- full Charlson would map specific code lists).
_CHARLSON_PREFIXES = {
    "I", "C", "K", "N", "E", "G", "J", "M",  # cardio, cancer, GI, renal, endo, neuro, resp, msk
}


def _race_norm(s: str) -> str:
    """Map MIMIC race strings to TrustedRisk fairness audit categories."""
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    if "black" in s or "african" in s:
        return "black"
    if "white" in s:
        return "white"
    if "hispanic" in s or "latino" in s:
        return "hispanic"
    if "asian" in s:
        return "asian"
    return "unknown"


def _insurance_norm(s: str) -> str:
    if not isinstance(s, str):
        return "unknown"
    s = s.lower()
    if "medicaid" in s:
        return "medicaid"
    if "medicare" in s:
        return "medicare"
    if "private" in s or "blue" in s or "uhc" in s or "aetna" in s:
        return "private"
    if "self" in s or "uninsured" in s:
        return "uninsured"
    return s.strip().lower()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    if not MIMIC_BASE.exists():
        print(f"ERROR: {MIMIC_BASE} not found. Run download first.")
        return 2

    print(f"Reading {MIMIC_BASE}...")
    adm = pd.read_csv(MIMIC_BASE / "admissions.csv.gz",
                       parse_dates=["admittime", "dischtime"])
    pat = pd.read_csv(MIMIC_BASE / "patients.csv.gz")
    diag = pd.read_csv(MIMIC_BASE / "diagnoses_icd.csv.gz")
    print(f"  admissions: {len(adm)} | patients: {len(pat)} | diagnoses: {len(diag)}")

    # Sort admissions chronologically per subject for prior-admission queries
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)

    # Charlson proxy: count distinct ICD-10 chapter prefixes per admission
    diag["chapter"] = diag["icd_code"].astype(str).str[0]
    diag["is_charlson"] = diag["chapter"].isin(_CHARLSON_PREFIXES)
    charlson_count = (
        diag[diag["is_charlson"]]
        .groupby("hadm_id")["chapter"]
        .nunique()
        .rename("lace_charlson_count")
    )

    # Anchor age per subject
    age_lookup = pat.set_index("subject_id")["anchor_age"].to_dict()

    # Pre-compute admit times per subject for ED-visit and readmit lookups
    by_subject: dict[int, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    for _, r in adm.iterrows():
        by_subject.setdefault(r["subject_id"], []).append(
            (r["admittime"], r["dischtime"], str(r.get("admission_type", "")))
        )

    rows: list[dict[str, Any]] = []
    for _, r in adm.iterrows():
        sid = int(r["subject_id"])
        hadm = int(r["hadm_id"])
        admit = r["admittime"]
        disch = r["dischtime"]
        if pd.isna(admit) or pd.isna(disch):
            continue
        los_days = (disch - admit).total_seconds() / 86400.0
        if los_days < 0:
            continue

        adm_type = str(r.get("admission_type", "")).upper()
        is_acute = ("URGENT" in adm_type or "EMERGENCY" in adm_type
                      or "EMERGENT" in adm_type)
        a_pts = 3 if is_acute else 0

        # Charlson
        n_charlson = int(charlson_count.get(hadm, 0))
        c_pts = _score_charlson_count(n_charlson)

        # ED visits in past 6 months -- count subject's prior emergency admissions
        prior_admits = [
            (a_t, d_t, t) for (a_t, d_t, t) in by_subject.get(sid, [])
            if a_t < admit and (admit - a_t) <= pd.Timedelta(days=180)
        ]
        ed_count = sum(
            1 for (_, _, t) in prior_admits
            if t and ("EMERGENCY" in t.upper() or "EMERGENT" in t.upper()
                       or "URGENT" in t.upper())
        )
        e_pts = min(4, ed_count)

        l_pts = _score_los(los_days)
        lace_total = l_pts + a_pts + c_pts + e_pts

        # 30-day readmission outcome
        future_admits = [
            a_t for (a_t, _, _) in by_subject.get(sid, [])
            if a_t > disch and (a_t - disch) <= pd.Timedelta(days=30)
        ]
        readmit_30d = 1 if future_admits else 0

        age = int(age_lookup.get(sid, 60))
        if age < 18 or age > 120:
            # MIMIC anchor_age can be >100 for de-identified seniors; clamp
            age = max(18, min(120, age))

        rows.append({
            "patient_id": f"mimic-{sid}-{hadm}",
            "age": age,
            "race": _race_norm(str(r.get("race", ""))),
            "insurance": _insurance_norm(str(r.get("insurance", ""))),
            "lace_los": int(round(los_days)),
            "lace_acuity": 1 if is_acute else 0,
            "lace_charlson": n_charlson,
            "lace_ed_visits_6mo": ed_count,
            "lace_total": lace_total,
            "outcome_30d_readmit": readmit_30d,
        })

    out_df = pd.DataFrame(rows)
    out_path = ROOT / "data" / "mimic_iv_lace_cohort.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path.relative_to(ROOT)} ({len(out_df)} rows)")
    print(f"  outcome rate: {out_df['outcome_30d_readmit'].mean():.3f}")
    print(f"  mean LACE: {out_df['lace_total'].mean():.2f}")
    print(f"  race distribution: {dict(out_df['race'].value_counts())}")
    print(f"  insurance distribution: {dict(out_df['insurance'].value_counts())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
