"""External validation harness for the calibrated readmission risk model.

Validates `compute_readmission_risk` (LACE + Beta-Binomial calibration) on
an EXTERNAL cohort (different distribution than the Synthea training data).

Inputs:
  - External cohort CSV/parquet with columns:
      patient_id, age, race, insurance, lace_los, lace_acuity,
      lace_charlson, lace_ed_visits_6mo, outcome_30d_readmit
  - Or: --synthetic flag -> generates a 500-patient cross-shifted cohort
    (LACE distribution skewed toward higher acuity, mortality rate +30%
    vs Synthea baseline). This is a stand-in for MIMIC-IV demo until the
    user provides the real CSV.

Outputs:
  - docs/validation/external_validation_report.json
  - docs/validation/reliability_diagram.png (when matplotlib is available)
  - docs/validation/calibration_metrics.csv
  - Console summary

Metrics computed:
  - Brier score (overall + per subgroup)
  - Expected Calibration Error (ECE) with 10 equal-width bins
  - Maximum Calibration Error (MCE)
  - AUROC (binary classifier interpretation)
  - Reliability diagram: predicted vs observed in 10 bins
  - Per-subgroup metrics: race, insurance, age band
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@dataclass
class ValidationMetrics:
    n: int
    brier_score: float
    ece: float
    mce: float
    auroc: float
    mean_predicted: float
    observed_rate: float
    bins: list[dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Synthetic cohort generator (cross-shifted vs Synthea training)
# ─────────────────────────────────────────────────────────────────────

def generate_synthetic_external_cohort(n: int = 500, seed: int = 1234
                                          ) -> list[dict[str, Any]]:
    """Generate a 500-patient external cohort with shifted LACE distribution
    and amplified outcome rate vs Synthea baseline.

    Why cross-shifted: the W1 calibrated coefficients were fit on a
    Synthea synthetic cohort (n=7780) with mean LACE ~7. We simulate an
    external population (e.g. urban safety-net hospital) with mean LACE ~9
    and a higher 30-day readmission rate (~18% vs Synthea's 11%) -- this
    stresses the calibration model's generalisation.
    """
    rng = np.random.default_rng(seed)
    cohort: list[dict[str, Any]] = []
    for i in range(n):
        # LACE features, shifted upward
        los_days = max(0, int(rng.normal(loc=5.5, scale=3.5)))
        ed_visits = int(rng.poisson(lam=1.2))
        acuity = int(rng.binomial(n=1, p=0.85))   # 85% acute admissions
        charlson_n = int(rng.poisson(lam=3.0))      # mean 3 comorbidities
        # Demographics -- diverse mix to stress fairness
        race = rng.choice(
            ["white", "black", "hispanic", "asian", "unknown"],
            p=[0.40, 0.30, 0.18, 0.07, 0.05],
        )
        age = int(rng.normal(loc=68, scale=15))
        age = max(18, min(95, age))
        insurance = rng.choice(
            ["medicare", "medicaid", "private", "uninsured"],
            p=[0.50, 0.25, 0.20, 0.05],
        )
        # Generate outcome from a logistic with LACE features + small race effect
        # so the validation can detect drift
        l = _score_los(los_days)
        a = 3 if acuity else 0
        c = _score_charlson(charlson_n)
        e = min(4, ed_visits)
        lace_total = l + a + c + e

        # Logit base from LACE; Black race +0.3, low SES (medicaid/uninsured) +0.2
        logit = -2.5 + 0.20 * lace_total
        if race == "black":
            logit += 0.30
        if insurance in ("medicaid", "uninsured"):
            logit += 0.20
        prob = 1.0 / (1.0 + np.exp(-logit))
        outcome = int(rng.random() < prob)

        cohort.append({
            "patient_id": f"ext-{i:04d}",
            "age": age, "race": str(race), "insurance": str(insurance),
            "lace_los": los_days, "lace_acuity": int(acuity),
            "lace_charlson": charlson_n, "lace_ed_visits_6mo": ed_visits,
            "lace_total": lace_total,
            "outcome_30d_readmit": outcome,
        })
    return cohort


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
    return n


# ─────────────────────────────────────────────────────────────────────
# Score model on external cohort
# ─────────────────────────────────────────────────────────────────────

def predict_lookup(lace_total: int, lookup: dict[str, Any]) -> dict[str, Any]:
    """Predict via the calibrated lookup table (avoids running the full
    async tool stack; we only need the calibration mapping)."""
    entry = lookup.get(str(lace_total))
    if entry is None:
        # Fallback: clamp
        entry = lookup.get("19" if lace_total > 19 else "0",
                            {"prob_mean": 0.10})
    return {
        "prob_mean": float(entry["prob_mean"]),
        "prob_ci95": list(entry.get("prob_ci95", [0.05, 0.20])),
    }


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

def compute_brier_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def compute_ece_mce(y_pred: np.ndarray, y_true: np.ndarray, n_bins: int = 10
                      ) -> tuple[float, float, list[dict[str, Any]]]:
    """Equal-width binning ECE + MCE + per-bin diagnostic info."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_info: list[dict[str, Any]] = []
    n_total = len(y_pred)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_pred >= lo) & (y_pred <= hi)
        else:
            mask = (y_pred >= lo) & (y_pred < hi)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            bin_info.append({
                "bin_lo": float(lo), "bin_hi": float(hi),
                "n": 0, "mean_predicted": None, "observed_rate": None,
                "calibration_gap": None,
            })
            continue
        mean_pred = float(y_pred[mask].mean())
        obs_rate = float(y_true[mask].mean())
        gap = abs(mean_pred - obs_rate)
        ece += (n_in_bin / n_total) * gap
        mce = max(mce, gap)
        bin_info.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "n": n_in_bin, "mean_predicted": mean_pred,
            "observed_rate": obs_rate, "calibration_gap": gap,
        })
    return float(ece), float(mce), bin_info


def compute_auroc(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Manual AUROC implementation (avoid scikit-learn dep). Trapezoidal
    integration over the ROC curve."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    n_pos = int(y_sorted.sum())
    n_neg = len(y_sorted) - n_pos
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / n_pos
    fpr = fps / n_neg
    # Prepend (0,0) and append (1,1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))


def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray
                       ) -> ValidationMetrics:
    brier = compute_brier_score(y_pred, y_true)
    ece, mce, bins = compute_ece_mce(y_pred, y_true)
    auroc = compute_auroc(y_pred, y_true)
    return ValidationMetrics(
        n=len(y_pred),
        brier_score=brier, ece=ece, mce=mce, auroc=auroc,
        mean_predicted=float(y_pred.mean()),
        observed_rate=float(y_true.mean()),
        bins=bins,
    )


# ─────────────────────────────────────────────────────────────────────
# Subgroup analysis with bootstrap CIs
# ─────────────────────────────────────────────────────────────────────

def bootstrap_metric(y_pred: np.ndarray, y_true: np.ndarray,
                       metric_fn, n_boot: int = 500, seed: int = 42
                       ) -> tuple[float, float, float]:
    """Returns (point_estimate, lo_ci95, hi_ci95)."""
    rng = np.random.default_rng(seed)
    n = len(y_pred)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples.append(metric_fn(y_pred[idx], y_true[idx]))
        except Exception:
            continue
    samples_arr = np.array([s for s in samples if not np.isnan(s)])
    if len(samples_arr) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(samples_arr))
    lo = float(np.quantile(samples_arr, 0.025))
    hi = float(np.quantile(samples_arr, 0.975))
    return point, lo, hi


def subgroup_metrics(cohort: list[dict[str, Any]], y_pred: np.ndarray,
                       y_true: np.ndarray, key: str
                       ) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    values = {row[key] for row in cohort}
    for v in sorted(map(str, values)):
        mask = np.array([str(row[key]) == v for row in cohort])
        if mask.sum() < 5:
            continue
        sub_pred = y_pred[mask]
        sub_true = y_true[mask]
        m = compute_metrics(sub_pred, sub_true)
        # Bootstrap AUROC + Brier
        auroc_pt, auroc_lo, auroc_hi = bootstrap_metric(
            sub_pred, sub_true, compute_auroc,
        )
        brier_pt, brier_lo, brier_hi = bootstrap_metric(
            sub_pred, sub_true, compute_brier_score,
        )
        out[v] = {
            "n": m.n,
            "brier_score": m.brier_score,
            "brier_ci95": [brier_lo, brier_hi],
            "ece": m.ece,
            "auroc": m.auroc,
            "auroc_ci95": [auroc_lo, auroc_hi],
            "mean_predicted": m.mean_predicted,
            "observed_rate": m.observed_rate,
            "calibration_gap": m.mean_predicted - m.observed_rate,
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# Reliability diagram
# ─────────────────────────────────────────────────────────────────────

def write_reliability_diagram(bins: list[dict[str, Any]],
                                 out_path: Path) -> bool:
    """Try matplotlib; return True if written. Skip silently if unavailable."""
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [(b["bin_lo"] + b["bin_hi"]) / 2.0 for b in bins if b["n"] > 0]
    ys = [b["observed_rate"] for b in bins if b["n"] > 0]
    sizes = [max(20, b["n"] * 2) for b in bins if b["n"] > 0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(xs, ys, s=sizes, c="tab:blue", alpha=0.7,
                label="Bin (size = count)")
    for b in bins:
        if b["n"] > 0:
            x = (b["bin_lo"] + b["bin_hi"]) / 2.0
            ax.annotate(f'n={b["n"]}', (x, b["observed_rate"]),
                          textcoords="offset points", xytext=(5, 5),
                          fontsize=8)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed rate")
    ax.set_title("Reliability diagram (10 equal-width bins)")
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=str, default=None,
                          help="External cohort CSV path. If absent, --synthetic is used.")
    parser.add_argument("--synthetic", action="store_true",
                          help="Use the synthetic cross-shifted cohort.")
    parser.add_argument("--n-synthetic", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=str,
                          default="docs/validation")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load coefficients
    coef_path = (ROOT / "data" / "coefficients.json")
    coef = json.loads(coef_path.read_text(encoding="utf-8"))
    lookup = (coef.get("runtime_coefficients") or {}).get("lookup_table", {})
    if not lookup:
        print("ERROR: coefficients.json missing runtime_coefficients.lookup_table")
        return 2

    # Load cohort
    if args.input:
        import csv
        cohort: list[dict[str, Any]] = []
        with open(args.input, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["lace_total"] = int(row.get("lace_total")
                                          or _score_los(int(row["lace_los"]))
                                          + (3 if int(row["lace_acuity"]) else 0)
                                          + _score_charlson(int(row["lace_charlson"]))
                                          + min(4, int(row["lace_ed_visits_6mo"])))
                row["age"] = int(row["age"])
                row["outcome_30d_readmit"] = int(row["outcome_30d_readmit"])
                cohort.append(row)
        cohort_source = f"external CSV: {args.input}"
    else:
        cohort = generate_synthetic_external_cohort(n=args.n_synthetic,
                                                      seed=args.seed)
        cohort_source = (f"synthetic cross-shifted cohort (n={args.n_synthetic}, "
                          f"seed={args.seed})")
    print(f"Loaded cohort: {cohort_source} ({len(cohort)} rows)")

    # Predict
    y_pred = np.array([
        predict_lookup(int(row["lace_total"]), lookup)["prob_mean"]
        for row in cohort
    ])
    y_true = np.array([int(row["outcome_30d_readmit"]) for row in cohort])

    overall = compute_metrics(y_pred, y_true)
    print(f"\nOverall metrics:")
    print(f"  N            = {overall.n}")
    print(f"  Brier        = {overall.brier_score:.4f}")
    print(f"  ECE          = {overall.ece:.4f}")
    print(f"  MCE          = {overall.mce:.4f}")
    print(f"  AUROC        = {overall.auroc:.4f}")
    print(f"  Mean pred    = {overall.mean_predicted:.4f}")
    print(f"  Observed     = {overall.observed_rate:.4f}")
    print(f"  Calib gap    = {overall.mean_predicted - overall.observed_rate:+.4f}")

    # Subgroup analyses
    by_race = subgroup_metrics(cohort, y_pred, y_true, "race")
    by_insurance = subgroup_metrics(cohort, y_pred, y_true, "insurance")
    # Age bands
    age_band_lookup = {row["patient_id"]:
                          ("18-64" if int(row["age"]) < 65 else
                           "65-74" if int(row["age"]) < 75 else
                           "75-84" if int(row["age"]) < 85 else "85+")
                       for row in cohort}
    cohort_with_band = [
        {**row, "age_band": age_band_lookup[row["patient_id"]]}
        for row in cohort
    ]
    by_age = subgroup_metrics(cohort_with_band, y_pred, y_true, "age_band")

    print("\nBy race:")
    for k, v in by_race.items():
        print(f"  {k:20} n={v['n']:4d}  brier={v['brier_score']:.4f}  "
               f"AUROC={v['auroc']:.3f} [CI {v['auroc_ci95'][0]:.3f}-{v['auroc_ci95'][1]:.3f}]  "
               f"calib_gap={v['calibration_gap']:+.3f}")
    print("\nBy insurance:")
    for k, v in by_insurance.items():
        print(f"  {k:20} n={v['n']:4d}  brier={v['brier_score']:.4f}  "
               f"AUROC={v['auroc']:.3f}  calib_gap={v['calibration_gap']:+.3f}")
    print("\nBy age band:")
    for k, v in by_age.items():
        print(f"  {k:20} n={v['n']:4d}  brier={v['brier_score']:.4f}  "
               f"AUROC={v['auroc']:.3f}  calib_gap={v['calibration_gap']:+.3f}")

    # Reliability diagram
    rd_path = out_dir / "reliability_diagram.png"
    written = write_reliability_diagram(overall.bins, rd_path)
    if written:
        print(f"\nReliability diagram -> {rd_path.relative_to(ROOT)}")

    # Calibration metrics CSV
    csv_path = out_dir / "calibration_metrics.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("subgroup_dim,subgroup_value,n,brier_score,ece,auroc,"
                 "auroc_ci95_lo,auroc_ci95_hi,mean_predicted,observed_rate,calibration_gap\n")
        for k, v in by_race.items():
            f.write(f"race,{k},{v['n']},{v['brier_score']:.6f},{v['ece']:.6f},"
                     f"{v['auroc']:.6f},{v['auroc_ci95'][0]:.6f},{v['auroc_ci95'][1]:.6f},"
                     f"{v['mean_predicted']:.6f},{v['observed_rate']:.6f},"
                     f"{v['calibration_gap']:.6f}\n")
        for k, v in by_insurance.items():
            f.write(f"insurance,{k},{v['n']},{v['brier_score']:.6f},{v['ece']:.6f},"
                     f"{v['auroc']:.6f},{v['auroc_ci95'][0]:.6f},{v['auroc_ci95'][1]:.6f},"
                     f"{v['mean_predicted']:.6f},{v['observed_rate']:.6f},"
                     f"{v['calibration_gap']:.6f}\n")
        for k, v in by_age.items():
            f.write(f"age_band,{k},{v['n']},{v['brier_score']:.6f},{v['ece']:.6f},"
                     f"{v['auroc']:.6f},{v['auroc_ci95'][0]:.6f},{v['auroc_ci95'][1]:.6f},"
                     f"{v['mean_predicted']:.6f},{v['observed_rate']:.6f},"
                     f"{v['calibration_gap']:.6f}\n")
    print(f"Calibration metrics -> {csv_path.relative_to(ROOT)}")

    # Aggregate report JSON
    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort_source": cohort_source,
        "model_name": coef.get("model_name"),
        "model_version": coef.get("model_version"),
        "selected_spec": coef.get("selected_model"),
        "training_cohort": "Synthea synthetic n=7780 + 100 MIMIC-IV demo cross-check",
        "validation_cohort_n": overall.n,
        "overall": {
            "brier_score": overall.brier_score,
            "ece": overall.ece,
            "mce": overall.mce,
            "auroc": overall.auroc,
            "mean_predicted": overall.mean_predicted,
            "observed_rate": overall.observed_rate,
            "calibration_gap": overall.mean_predicted - overall.observed_rate,
            "bins": overall.bins,
        },
        "by_race": by_race,
        "by_insurance": by_insurance,
        "by_age_band": by_age,
    }
    out_path = out_dir / "external_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report -> {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
