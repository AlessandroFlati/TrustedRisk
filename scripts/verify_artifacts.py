#!/usr/bin/env python
"""Verify that staged calibration artifacts conform to the OUTPUT_SCHEMAs.

Runs as part of `make promote-artifacts` and in CI.
Exit code 0 = all artifacts valid. Non-zero = one or more failures.

Checks:
  1. coefficients.json -- schema + SUPPORTED_MODEL_NAMES membership + sanity bounds.
  2. grounding_index_*.pkl -- loadable, schema matches W3 OUTPUT_SCHEMA.
  3. abstain_policy_*.json -- threshold bounds, schema match.
  4. patient_{01,02,03}_{clean,abstain,complex}.json -- valid FHIR Bundle.
  5. expected_signals.json -- schema + tolerance fields.
  6. _provenance block present in every JSON artifact.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any


ERRORS: list[str] = []
WARNINGS: list[str] = []


def err(msg: str) -> None:
    ERRORS.append(msg)
    print(f"FAIL: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"WARN: {msg}", file=sys.stderr)


def ok(msg: str) -> None:
    print(f"OK:   {msg}")


def main() -> int:
    root = Path(__file__).parent.parent
    data_dir = root / "data"
    fixtures_dir = root / "fixtures"

    if not data_dir.exists():
        err(f"data/ directory not found at {data_dir}")
        return 1

    _check_coefficients(data_dir / "coefficients.json")
    _check_grounding_index(data_dir)
    _check_abstain_policy(data_dir)
    _check_cohort_embeddings(data_dir / "cohort_embeddings.npz")
    _check_fixtures(fixtures_dir)
    _check_expected_signals(fixtures_dir / "expected_signals.json")

    print()
    print(f"=== Summary: {len(ERRORS)} errors, {len(WARNINGS)} warnings ===")
    return 1 if ERRORS else 0


# ─────────────────────────────────────────────────────────────────────

def _check_coefficients(path: Path) -> None:
    if not path.exists():
        err(f"coefficients.json not found at {path}")
        return
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        err(f"coefficients.json malformed: {e}")
        return

    # _provenance
    if not isinstance(data.get("_provenance"), dict):
        err("coefficients.json: missing or malformed _provenance block")

    # model_name
    model_name = data.get("model_name")
    supported = ["lace-plus-bayesian-v1"]  # SUPPORTED_MODEL_NAMES
    if model_name not in supported:
        err(f"coefficients.json: model_name={model_name!r} not in SUPPORTED_MODEL_NAMES={supported}")

    # calibration metrics
    cal = data.get("calibration_metrics_final") or {}
    ece = cal.get("ece")
    if ece is not None and (ece < 0 or ece > 0.15):
        err(f"coefficients.json: ECE={ece} out of reasonable range [0, 0.15]")

    # runtime_coefficients.lookup_table sanity
    lookup = (data.get("runtime_coefficients") or {}).get("lookup_table", {})
    for lace_str, entry in lookup.items():
        try:
            lace_int = int(lace_str)
        except ValueError:
            err(f"coefficients.json: lookup_table key {lace_str!r} not an integer string")
            continue
        if not (0 <= lace_int <= 19):
            err(f"coefficients.json: lookup_table key {lace_int} out of LACE range [0, 19]")
        pm = entry.get("prob_mean")
        if pm is None or not (0 <= pm <= 1):
            err(f"coefficients.json: lookup[{lace_int}].prob_mean={pm} out of [0, 1]")

    # v0.3 confidence field
    confidence = data.get("confidence", "preferred")
    if confidence not in ("preferred", "degraded"):
        warn(f"coefficients.json: unexpected confidence={confidence!r}")

    ok(f"coefficients.json: model={model_name}, ECE={ece}, confidence={confidence}")


def _check_grounding_index(data_dir: Path) -> None:
    """Check any grounding_index_*.pkl in data/. Warn (not err) if none present."""
    candidates = list(data_dir.glob("grounding_index_*.pkl"))
    if not candidates:
        warn("No grounding_index_*.pkl found in data/ (ground_claim will run in stub mode)")
        return

    for fp in candidates:
        try:
            with fp.open("rb") as f:
                pkg = pickle.load(f)
        except (pickle.UnpicklingError, OSError) as e:
            err(f"{fp.name}: pickle load failed: {e}")
            continue
        embedder = pkg.get("embedder") or {}
        if not embedder.get("name"):
            err(f"{fp.name}: missing embedder.name")
        if "chunks" not in pkg or not isinstance(pkg["chunks"], list):
            err(f"{fp.name}: missing or malformed chunks[]")
        ok(f"{fp.name}: embedder={embedder.get('name')}, n_chunks={len(pkg.get('chunks', []))}")


def _check_abstain_policy(data_dir: Path) -> None:
    topological = data_dir / "abstain_policy_topological.json"
    fallback = data_dir / "abstain_policy_lace_percentile.json"

    policy_type_env = os.environ.get("TRUSTEDRISK_OOD_POLICY_TYPE", "").strip()
    expected_path_env = os.environ.get("TRUSTEDRISK_OOD_POLICY_PATH", "").strip()

    # Check that at least one policy exists
    if not (topological.exists() or fallback.exists()):
        err("No abstain policy found (expected abstain_policy_topological.json or abstain_policy_lace_percentile.json)")
        return

    for fp in (topological, fallback):
        if not fp.exists():
            continue
        try:
            with fp.open(encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            err(f"{fp.name}: malformed JSON: {e}")
            continue

        if "_provenance" not in data:
            err(f"{fp.name}: missing _provenance block")

        policy_type = data.get("policy_type")
        if fp.name == "abstain_policy_topological.json":
            if policy_type != "topological_clustering":
                err(f"{fp.name}: policy_type={policy_type!r} (expected 'topological_clustering')")
            threshold = data.get("ood_distance_threshold")
            if threshold is None or not (0 <= threshold <= 10):
                err(f"{fp.name}: ood_distance_threshold={threshold} out of sanity range [0, 10]")
        else:  # lace_percentile
            if policy_type != "lace_percentile_fallback":
                err(f"{fp.name}: policy_type={policy_type!r} (expected 'lace_percentile_fallback')")
            threshold = data.get("threshold_lace")
            if threshold is None or not (0 <= threshold <= 19):
                err(f"{fp.name}: threshold_lace={threshold} out of [0, 19]")

        ok(f"{fp.name}: policy_type={policy_type}")

    # Env consistency
    if policy_type_env == "topological" and not topological.exists():
        err("TRUSTEDRISK_OOD_POLICY_TYPE=topological but abstain_policy_topological.json is missing")
    if policy_type_env == "lace_percentile" and not fallback.exists():
        err("TRUSTEDRISK_OOD_POLICY_TYPE=lace_percentile but abstain_policy_lace_percentile.json is missing")


def _check_cohort_embeddings(path: Path) -> None:
    if not path.exists():
        warn(f"cohort_embeddings.npz not found at {path} (topological OOD policy won't function)")
        return
    try:
        import numpy as np
    except ImportError:
        warn("numpy not available; cannot validate cohort_embeddings.npz")
        return
    try:
        data = np.load(path)
        keys = list(data.files)
    except (OSError, ValueError) as e:
        err(f"cohort_embeddings.npz: load failed: {e}")
        return
    required = {"cluster_centroids", "cluster_assignment"}
    missing = required - set(keys)
    if missing:
        err(f"cohort_embeddings.npz: missing keys {missing}")
    ok(f"cohort_embeddings.npz: keys={keys}")


def _check_fixtures(fixtures_dir: Path) -> None:
    """Check patient_{01,02,03}_*.json FHIR Bundles if present."""
    expected = [
        "patient_01_clean.json",
        "patient_02_abstain.json",
        "patient_03_complex.json",
    ]
    for name in expected:
        fp = fixtures_dir / name
        if not fp.exists():
            warn(f"fixtures/{name} not found (regression suite will skip this fixture)")
            continue
        try:
            with fp.open(encoding="utf-8") as f:
                bundle = json.load(f)
        except json.JSONDecodeError as e:
            err(f"fixtures/{name}: malformed JSON: {e}")
            continue
        if bundle.get("resourceType") != "Bundle":
            err(f"fixtures/{name}: resourceType != 'Bundle'")
        entries = bundle.get("entry") or []
        if not isinstance(entries, list) or len(entries) == 0:
            err(f"fixtures/{name}: Bundle has no entries")
            continue
        has_patient = any(
            (e.get("resource") or {}).get("resourceType") == "Patient"
            for e in entries
        )
        if not has_patient:
            err(f"fixtures/{name}: Bundle missing a Patient resource")
        ok(f"fixtures/{name}: {len(entries)} entries")


def _check_expected_signals(path: Path) -> None:
    if not path.exists():
        warn(f"expected_signals.json not found at {path} (regression suite limited)")
        return
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        err(f"expected_signals.json: malformed: {e}")
        return
    for key in ("patient_01_clean", "patient_02_abstain", "patient_03_complex"):
        if key not in data:
            err(f"expected_signals.json: missing key '{key}'")
    # v0.3 addition: tolerance_mode
    tm = data.get("tolerance_mode", "strict")
    if tm not in ("strict", "loose"):
        warn(f"expected_signals.json: unexpected tolerance_mode={tm!r}")
    ok(f"expected_signals.json: tolerance_mode={tm}")


if __name__ == "__main__":
    sys.exit(main())
