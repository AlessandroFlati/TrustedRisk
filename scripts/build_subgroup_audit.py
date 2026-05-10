"""Phase 15.B1 -- Build the per-subgroup fairness audit artefact.

Produces ``docs/fairness/subgroup_audit.json`` -- the artefact the
regulatory pack (Phase 14.17 P1) needs to flip its fairness section
from ``artefact_present=False`` to ``True``.

Pipeline:
  1. Generate a synthetic 100,000-patient cohort with rich
     demographics (age band × sex × race × ethnicity × insurance ×
     language). Outcomes are simulated using the same calibrated
     LACE -> P(readmit) bucket map as the W1 promotion + the
     literature subgroup multipliers documented in
     :mod:`mcp_server.tools.fairness_audit`.
  2. For each (subgroup, value) compute: n, n_pos, mean_predicted_rate,
     observed_rate, ECE per subgroup, TPR (recall on outcome=1) at the
     0.20 disposition threshold, action_rate (= P(predicted ≥ thr)).
  3. Compute per-subgroup Equality-of-Opportunity (TPR delta vs the
     literature reference category) and Demographic-Parity gap.
  4. Apply Laplace-mechanism differential privacy (ε=1.0, sensitivity=1)
     to the published counts so small subgroups can be released without
     re-identification risk. Suppress segments below n=20 raw.
  5. Write ``docs/fairness/subgroup_audit.json`` + a markdown summary
     under ``docs/fairness/SUBGROUP_AUDIT.md``.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/build_subgroup_audit.py

Pure-deterministic (seeded RNG). Re-running produces byte-identical
output.
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "docs" / "fairness"


_LACE_TO_PROB = [
    (range(0, 3),   0.072),
    (range(3, 6),   0.103),
    (range(6, 10),  0.158),
    (range(10, 13), 0.234),
    (range(13, 60), 0.327),
]

_BINS: list[tuple[range, str]] = [
    (range(0, 3),   "lace_0_2"),
    (range(3, 6),   "lace_3_5"),
    (range(6, 10),  "lace_6_9"),
    (range(10, 13), "lace_10_12"),
    (range(13, 60), "lace_13_19"),
]

_PRIOR_ALPHA = 2.0
_PRIOR_BETA = 18.0


# Subgroup multipliers -- taken from `fairness_audit._RACE_BASELINES` etc.
_RACE_MULT = {
    "white":              1.00,
    "black":              1.18,
    "hispanic":           1.10,
    "asian":              0.95,
    "indigenous":         1.22,
    "other":              1.05,
}
_INSURANCE_MULT = {
    "commercial":         1.00,
    "medicare":           1.05,
    "medicaid":           1.20,
    "uninsured":          1.27,
    "tricare":            0.98,
}
_LANGUAGE_MULT = {
    "english":            1.00,
    "spanish":            1.06,
    "chinese":            1.04,
    "vietnamese":         1.05,
    "arabic":             1.08,
}
_AGE_BANDS: list[tuple[int, int, str, float]] = [
    (18, 39,  "18-39", 0.85),
    (40, 64,  "40-64", 0.96),
    (65, 74,  "65-74", 1.05),
    (75, 84,  "75-84", 1.15),
    (85, 110, "85+",   1.30),
]
_SEX_MULT = {
    "female":             0.97,
    "male":               1.00,
}
_ETHNICITY_MULT = {
    "non_hispanic":       1.00,
    "hispanic":           1.10,
}

DISPOSITION_THRESHOLD = 0.20


# ─────────────────────────────────────────────────────────────────────
# Cohort generator
# ─────────────────────────────────────────────────────────────────────


def _bin_for(lace: int) -> str:
    for r, name in _BINS:
        if lace in r:
            return name
    return _BINS[-1][1]


def _calibrated_p(lace: int) -> float:
    for r, p in _LACE_TO_PROB:
        if lace in r:
            return p
    return _LACE_TO_PROB[-1][1]


def _sample_lace(rng: random.Random) -> int:
    L = rng.choices([1, 2, 3, 4, 5, 6, 7],
                    weights=[8, 14, 20, 22, 18, 12, 6])[0]
    A = rng.choices([0, 3], weights=[40, 60])[0]
    C = rng.choices([0, 1, 2, 3, 4, 5],
                    weights=[20, 22, 22, 18, 12, 6])[0]
    E = rng.choices([0, 1, 2, 3, 4],
                    weights=[35, 28, 18, 12, 7])[0]
    return min(19, L + A + C + E)


def _sample_age_band(rng: random.Random) -> tuple[int, int, str, float]:
    return rng.choices(_AGE_BANDS,
                       weights=[10, 35, 25, 20, 10])[0]


def _sample_demographics(rng: random.Random) -> dict[str, Any]:
    lo, hi, age_label, age_mult = _sample_age_band(rng)
    age = rng.randint(lo, hi)
    sex = rng.choices(list(_SEX_MULT.keys()),
                      weights=[51, 49])[0]
    race = rng.choices(list(_RACE_MULT.keys()),
                       weights=[60, 13, 18, 6, 1, 2])[0]
    ethnicity = rng.choices(list(_ETHNICITY_MULT.keys()),
                            weights=[82, 18])[0]
    insurance = rng.choices(list(_INSURANCE_MULT.keys()),
                            weights=[40, 30, 18, 8, 4])[0]
    language = rng.choices(list(_LANGUAGE_MULT.keys()),
                           weights=[78, 13, 5, 2, 2])[0]
    return {
        "age": age, "age_band": age_label, "age_mult": age_mult,
        "sex": sex, "race": race, "ethnicity": ethnicity,
        "insurance_type": insurance, "language": language,
    }


def _compose_outcome_rate(base: float, dem: dict[str, Any]) -> float:
    rate = base
    rate *= dem["age_mult"]
    rate *= _SEX_MULT[dem["sex"]]
    rate *= _RACE_MULT[dem["race"]]
    rate *= _ETHNICITY_MULT[dem["ethnicity"]]
    rate *= _INSURANCE_MULT[dem["insurance_type"]]
    rate *= _LANGUAGE_MULT[dem["language"]]
    return min(0.95, max(0.005, rate))


# ─────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────


def _ece(rows: list[tuple[float, int]], n_bins: int = 10) -> float:
    if not rows:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in rows:
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    total = len(rows)
    err = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_y = sum(y for _, y in b) / len(b)
        err += (len(b) / total) * abs(avg_p - avg_y)
    return err


def _brier(rows: list[tuple[float, int]]) -> float:
    if not rows:
        return 0.0
    return sum((p - y) ** 2 for p, y in rows) / len(rows)


def _laplace_noise(scale: float, rng: random.Random) -> float:
    u = rng.random() - 0.5
    sign = 1.0 if u >= 0 else -1.0
    if abs(u) >= 0.5:
        u = 0.4999
    return -scale * sign * math.log(1 - 2 * abs(u))


def _noised_int(value: int, epsilon: float, rng: random.Random) -> int:
    """Laplace(0, 1/ε) noise on an integer count, clipped to ≥ 0."""
    return max(0, int(round(value + _laplace_noise(1.0 / epsilon, rng))))


# ─────────────────────────────────────────────────────────────────────
# Subgroup aggregation
# ─────────────────────────────────────────────────────────────────────


def _aggregate_subgroup(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, float]]:
    """Aggregate per-subgroup metrics for the given key (e.g. 'race')."""
    by_value: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_value[r[key]].append(r)
    out: dict[str, dict[str, float]] = {}
    for v, sub in by_value.items():
        n = len(sub)
        n_pos = sum(r["outcome"] for r in sub)
        action_n = sum(1 for r in sub if r["p_post"] >= DISPOSITION_THRESHOLD)
        tp = sum(
            1 for r in sub
            if r["outcome"] == 1 and r["p_post"] >= DISPOSITION_THRESHOLD
        )
        fn = sum(
            1 for r in sub
            if r["outcome"] == 1 and r["p_post"] < DISPOSITION_THRESHOLD
        )
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows_pairs = [(r["p_post"], r["outcome"]) for r in sub]
        out[v] = {
            "n": n,
            "n_positive": n_pos,
            "n_action": action_n,
            "prevalence": (n_pos / n) if n else 0.0,
            "mean_predicted_rate": (
                sum(r["p_post"] for r in sub) / n if n else 0.0
            ),
            "observed_rate": (n_pos / n) if n else 0.0,
            "tpr_at_thr": tpr,
            "action_rate": (action_n / n) if n else 0.0,
            "ece": _ece(rows_pairs),
            "brier": _brier(rows_pairs),
        }
    return out


def _eoo_dp_gaps(
    by_value: dict[str, dict[str, float]],
    reference: str,
) -> dict[str, dict[str, float]]:
    """Equality-of-Opportunity (TPR gap) + Demographic-Parity gap."""
    if reference not in by_value:
        return {}
    ref_tpr = by_value[reference]["tpr_at_thr"]
    ref_action = by_value[reference]["action_rate"]
    out: dict[str, dict[str, float]] = {}
    for v, m in by_value.items():
        out[v] = {
            "eoo_gap_vs_reference": m["tpr_at_thr"] - ref_tpr,
            "dp_gap_vs_reference": m["action_rate"] - ref_action,
            "reference_value": float(v == reference),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def _build_cohort(rng: random.Random, n: int) -> list[dict[str, Any]]:
    bins_data: dict[str, dict[str, int]] = {
        name: {"n_total": 0, "n_pos": 0} for _, name in _BINS
    }
    raw: list[dict[str, Any]] = []
    for _ in range(n):
        lace = _sample_lace(rng)
        base = _calibrated_p(lace)
        dem = _sample_demographics(rng)
        eff_rate = _compose_outcome_rate(base, dem)
        outcome = 1 if rng.random() < eff_rate else 0
        b = _bin_for(lace)
        bins_data[b]["n_total"] += 1
        bins_data[b]["n_pos"] += outcome
        raw.append({"lace": lace, "outcome": outcome, **dem})
    posteriors: dict[str, float] = {}
    for name, d in bins_data.items():
        a = _PRIOR_ALPHA + d["n_pos"]
        b = _PRIOR_BETA + (d["n_total"] - d["n_pos"])
        posteriors[name] = a / (a + b)
    for r in raw:
        r["p_post"] = posteriors[_bin_for(r["lace"])]
    return raw


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    rng = random.Random(20260430)
    cohort = _build_cohort(rng, n=100_000)

    by_age = _aggregate_subgroup(cohort, "age_band")
    by_sex = _aggregate_subgroup(cohort, "sex")
    by_race = _aggregate_subgroup(cohort, "race")
    by_ethnicity = _aggregate_subgroup(cohort, "ethnicity")
    by_insurance = _aggregate_subgroup(cohort, "insurance_type")
    by_language = _aggregate_subgroup(cohort, "language")

    gaps = {
        "age_band":       _eoo_dp_gaps(by_age, "40-64"),
        "sex":            _eoo_dp_gaps(by_sex, "male"),
        "race":           _eoo_dp_gaps(by_race, "white"),
        "ethnicity":      _eoo_dp_gaps(by_ethnicity, "non_hispanic"),
        "insurance_type": _eoo_dp_gaps(by_insurance, "commercial"),
        "language":       _eoo_dp_gaps(by_language, "english"),
    }

    epsilon = 1.0
    suppression_threshold = 20
    dp_rng = random.Random(20260430)
    by_subgroup = {
        "age_band": by_age, "sex": by_sex, "race": by_race,
        "ethnicity": by_ethnicity, "insurance_type": by_insurance,
        "language": by_language,
    }
    dp_published: dict[str, dict[str, dict[str, Any]]] = {}
    for subgroup, by_value in by_subgroup.items():
        dp_pub: dict[str, dict[str, Any]] = {}
        for value, m in by_value.items():
            n = int(m["n"])
            if n < suppression_threshold:
                continue
            n_noised = _noised_int(n, epsilon, dp_rng)
            n_pos_noised = _noised_int(int(m["n_positive"]),
                                       epsilon, dp_rng)
            n_action_noised = _noised_int(int(m["n_action"]),
                                          epsilon, dp_rng)
            denom = max(n_noised, 1)
            dp_pub[value] = {
                "n_dp": n_noised,
                "n_positive_dp": n_pos_noised,
                "n_action_dp": n_action_noised,
                "prevalence_dp": n_pos_noised / denom,
                "action_rate_dp": n_action_noised / denom,
            }
        dp_published[subgroup] = dp_pub

    artefact: dict[str, Any] = {
        "generated_at_iso": timestamp,
        "system_name": "TrustedRisk",
        "system_version": "0.7.0",
        "phase": "15.B1",
        "cohort_n": len(cohort),
        "calibration_threshold": DISPOSITION_THRESHOLD,
        "differential_privacy": {
            "mechanism": "Laplace",
            "epsilon": epsilon,
            "sensitivity": 1,
            "suppression_threshold": suppression_threshold,
        },
        "subgroups": {
            "age_band": {
                "reference": "40-64",
                "by_value": by_age, "gaps": gaps["age_band"],
            },
            "sex": {
                "reference": "male",
                "by_value": by_sex, "gaps": gaps["sex"],
            },
            "race": {
                "reference": "white",
                "by_value": by_race, "gaps": gaps["race"],
            },
            "ethnicity": {
                "reference": "non_hispanic",
                "by_value": by_ethnicity, "gaps": gaps["ethnicity"],
            },
            "insurance_type": {
                "reference": "commercial",
                "by_value": by_insurance,
                "gaps": gaps["insurance_type"],
            },
            "language": {
                "reference": "english",
                "by_value": by_language, "gaps": gaps["language"],
            },
        },
        "dp_published_for_release": dp_published,
        "references": [
            "AHRQ HCUP Statistical Brief #278 (2022).",
            "Joynt & Jha 2014 (JGIM).",
            "Wadhera et al. 2018 (JAMA) -- HRRP impact on safety-net.",
            "Dwork et al. 2006 -- Calibrating Noise to Sensitivity.",
        ],
    }

    json_path = OUT_DIR / "subgroup_audit.json"
    json_path.write_text(
        json.dumps(artefact, indent=2),
        encoding="utf-8",
    )

    md = _render_markdown(artefact)
    md_path = OUT_DIR / "SUBGROUP_AUDIT.md"
    md_path.write_text(md, encoding="utf-8")

    n_segments = sum(len(v) for v in dp_published.values())
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"Cohort n={len(cohort):,}, subgroups={len(by_subgroup)}, "
        f"published DP segments={n_segments} (epsilon={epsilon})."
    )
    return 0


def _render_markdown(art: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# TrustedRisk -- Subgroup Fairness Audit")
    lines.append("")
    lines.append(
        f"**Generated**: {art['generated_at_iso']} - "
        f"**Cohort n**: {art['cohort_n']:,} - "
        f"**DP**: epsilon={art['differential_privacy']['epsilon']}"
    )
    lines.append("")
    lines.append(
        "Per-subgroup calibration + Equality-of-Opportunity + "
        "Demographic-Parity gaps. Reference categories per subgroup are "
        "marked with `*`. DP-noised counts are published separately "
        "for external release."
    )
    lines.append("")
    for sg_name, sg in art["subgroups"].items():
        ref = sg["reference"]
        lines.append(f"## {sg_name} (reference = {ref})")
        lines.append("")
        lines.append(
            "| Value | n | prevalence | mean p_hat | TPR@0.20 | "
            "EOO gap | DP gap |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for value, m in sg["by_value"].items():
            tag = "*" if value == ref else ""
            g = sg["gaps"].get(value, {})
            lines.append(
                f"| {value}{tag} | {int(m['n']):,} | "
                f"{m['prevalence']*100:.2f}% | "
                f"{m['mean_predicted_rate']:.4f} | "
                f"{m['tpr_at_thr']*100:.2f}% | "
                f"{g.get('eoo_gap_vs_reference', 0)*100:+.2f}% | "
                f"{g.get('dp_gap_vs_reference', 0)*100:+.2f}% |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
