"""SCALE-1 -- Generate a Synthea FHIR cohort and evaluate calibration.

Synthea (https://github.com/synthetichealth/synthea) is GPL-licensed (FREE).
This script:

  1. Downloads `synthea-with-dependencies.jar` from GitHub releases
     (cache via SYNTHEA_JAR env var or default path)
  2. Generates N synthetic patients with FHIR R4 output
  3. Loads each Patient bundle
  4. Calls `evaluate_synthea_cohort(...)` to compute ECE / AUROC / Brier
  5. Emits `docs/validation/SYNTHEA_<N>.md`

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
        .venv/Scripts/python.exe scripts/synthea_cohort_runner.py \
        --n 1000 --output-dir data/synthea_runs/run_1k

Re-uses an existing FHIR output directory if --skip-generate is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

# Ensure src/ is on the path when run from the repo root
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from a2a_agent.synthea_evaluator import evaluate_synthea_cohort  # noqa: E402


_DEFAULT_JAR_URL = (
    "https://github.com/synthetichealth/synthea/releases/download/"
    "v3.3.1/synthea-with-dependencies.jar"
)


def _ensure_synthea_jar(target_dir: Path) -> Path:
    """Return the path to a synthea-with-dependencies.jar, downloading
    if necessary. Honors $SYNTHEA_JAR override."""
    override = os.environ.get("SYNTHEA_JAR")
    if override:
        p = Path(override)
        if p.exists():
            print(f"[synthea] using SYNTHEA_JAR={p}")
            return p
        print(f"[synthea] SYNTHEA_JAR set but missing at {p}; downloading.")

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "synthea-with-dependencies.jar"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"[synthea] reusing cached jar at {target}")
        return target

    print(f"[synthea] downloading {_DEFAULT_JAR_URL} -> {target}")
    urllib.request.urlretrieve(_DEFAULT_JAR_URL, target)
    return target


def _generate_population(
    *, n: int, seed: int, output_dir: Path, jar_path: Path,
    java_xmx: str = "-Xmx4g",
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", java_xmx, "-jar", str(jar_path),
        "-p", str(n),
        "-s", str(seed),
        "--exporter.fhir.export", "true",
        "--exporter.fhir.transaction_bundle", "true",
        "--exporter.baseDirectory", str(output_dir),
    ]
    print("[synthea] running:", " ".join(cmd))
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(res.stdout[-2000:] if res.stdout else "")
    sys.stderr.write(res.stderr[-2000:] if res.stderr else "")
    if res.returncode != 0:
        raise RuntimeError(f"synthea exited with code {res.returncode}")


def _load_bundles(fhir_dir: Path) -> list[dict]:
    """Walk the Synthea FHIR output directory + load each Patient bundle."""
    bundles: list[dict] = []
    for path in fhir_dir.rglob("*.json"):
        if path.name in {"hospitalInformation*.json",
                            "practitionerInformation*.json"}:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        if doc.get("resourceType") != "Bundle":
            continue
        bundles.append(doc)
    return bundles


def _emit_markdown_report(report, output_path: Path,
                              n_generated: int, seed: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    body = [
        f"# Synthea Cohort Evaluation -- {report.cohort_label}\n",
        f"**Generated**: cohort_label={report.cohort_label}, "
        f"n_generated={n_generated}, seed={seed}\n",
        "## Summary\n",
        f"- n_patients (loaded): **{report.n_patients}**",
        f"- n_with_readmission_label: **{report.n_with_readmission_label}**",
        f"- n_readmitted: **{report.n_readmitted}**",
        f"- base_rate_readmission: **{report.base_rate_readmission:.4f}**",
        f"- coefficients_version: **{report.coefficients_version}**\n",
        "## Calibration Metrics\n",
        f"- **ECE** (10-bin): {report.ece if report.ece is None else f'{report.ece:.4f}'}",
        f"- **AUROC**: {report.auroc if report.auroc is None else f'{report.auroc:.4f}'}",
        f"- **Brier**: {report.brier_score if report.brier_score is None else f'{report.brier_score:.4f}'}\n",
        "## Calibration Table (5 bins)\n",
        "| Predicted bin | n | Predicted mean | Observed mean |",
        "|---|---|---|---|",
    ]
    for bucket, stats in report.calibration_buckets.items():
        body.append(
            f"| {bucket} | {int(stats['n'])} | "
            f"{stats['predicted_mean']:.4f} | {stats['observed_mean']:.4f} |"
        )
    body.append("\n## Rationale\n")
    body.append(report.rationale)
    body.append("\n## References\n")
    for ref in report.references:
        body.append(f"- {ref}")

    output_path.write_text("\n".join(body), encoding="utf-8")
    print(f"[synthea] wrote report -> {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path,
                          default=_ROOT / "data" / "synthea_runs" / "run_1k")
    parser.add_argument("--jar-cache", type=Path,
                          default=_ROOT / "data" / "synthea_runs")
    parser.add_argument("--report", type=Path,
                          default=_ROOT / "docs" / "validation" / "SYNTHEA_1000.md")
    parser.add_argument("--skip-generate", action="store_true",
                          help="Reuse FHIR output directory; skip the JAR call.")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--java-xmx", default="-Xmx4g")
    args = parser.parse_args()

    fhir_dir = args.output_dir / "fhir"

    if not args.skip_generate:
        jar_path = _ensure_synthea_jar(args.jar_cache)
        _generate_population(
            n=args.n, seed=args.seed, output_dir=args.output_dir,
            jar_path=jar_path, java_xmx=args.java_xmx,
        )

    if not fhir_dir.exists():
        print(f"[synthea] FHIR output dir missing: {fhir_dir}",
                file=sys.stderr)
        return 2

    print(f"[synthea] loading bundles from {fhir_dir}")
    bundles = _load_bundles(fhir_dir)
    print(f"[synthea] loaded {len(bundles)} bundles")
    if not bundles:
        print("[synthea] no bundles found -- abort.", file=sys.stderr)
        return 3

    report = evaluate_synthea_cohort(
        bundles, cohort_label=f"synthea_n{args.n}_seed{args.seed}",
        horizon_days=args.horizon_days,
    )
    _emit_markdown_report(report, args.report,
                              n_generated=args.n, seed=args.seed)
    print(report.rationale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
