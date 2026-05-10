"""Phase 17.CON4 - Build every TrustedRisk artefact in one pass.

Runs the artefact generators in dependency order:

  1. build_subgroup_audit          -> docs/fairness/...
  2. run_prospective_eval          -> docs/prospective/...
  3. generate_hrrp_benchmark       -> docs/validation/HRRP_BENCHMARK.md
  4. generate_model_card           -> docs/research/MODEL_CARD + DATASHEET
  5. generate_framework_crosswalks -> docs/regulatory/FRAMEWORK_CROSSWALKS
  6. generate_regulatory_pack      -> docs/regulatory/REGULATORY_PACK
  7. generate_guideline_crosswalk  -> docs/guidelines/GUIDELINE_CROSSWALK
  8. extend_specialist_bundle_coverage -> apps/specialist_*/agent_card.json
  9. generate_federation_registry  -> docs/federation/...
 10. e2e_showcase_v7               -> docs/e2e/v7/index.html
 11. generate_storymode            -> docs/showcase/STORYMODE.html
 12. generate_cost_simulation      -> docs/economics/...
 13. generate_federated_learning_report -> docs/federated/...
 14. run_redteam_v4                -> docs/adversarial/red_team_v4.json
 15. generate_counterfactual_ui    -> docs/ui/counterfactual.html
 16. generate_openapi              -> docs/api/openapi.json
 17. generate_module_catalog       -> docs/MODULE_CATALOG.md

Returns exit code 0 only when every step succeeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


_PIPELINE: list[tuple[str, list[Path]]] = [
    ("scripts/build_subgroup_audit.py", [
        ROOT / "docs" / "fairness" / "subgroup_audit.json",
        ROOT / "docs" / "fairness" / "SUBGROUP_AUDIT.md",
    ]),
    ("scripts/run_prospective_eval.py --n 2000", [
        ROOT / "docs" / "prospective" / "prospective_eval.json",
        ROOT / "docs" / "prospective" / "PROSPECTIVE_EVAL.md",
    ]),
    ("scripts/generate_hrrp_benchmark.py", [
        ROOT / "docs" / "validation" / "HRRP_BENCHMARK.md",
        ROOT / "docs" / "validation" / "hrrp_benchmark.json",
    ]),
    ("scripts/generate_model_card.py", [
        ROOT / "docs" / "research" / "MODEL_CARD.md",
        ROOT / "docs" / "research" / "DATASHEET.md",
    ]),
    ("scripts/generate_framework_crosswalks.py", [
        ROOT / "docs" / "regulatory" / "FRAMEWORK_CROSSWALKS.md",
    ]),
    ("scripts/generate_regulatory_pack.py", [
        ROOT / "docs" / "regulatory" / "REGULATORY_PACK.md",
        ROOT / "docs" / "regulatory" / "regulatory_pack.json",
    ]),
    ("scripts/generate_guideline_crosswalk.py", [
        ROOT / "docs" / "guidelines" / "GUIDELINE_CROSSWALK.md",
    ]),
    ("scripts/extend_specialist_bundle_coverage.py", []),
    ("scripts/generate_federation_registry.py", [
        ROOT / "docs" / "federation" / "FEDERATION_REGISTRY.md",
        ROOT / "docs" / "federation" / "marketplace_manifest.json",
    ]),
    ("scripts/e2e_showcase_v7.py", [
        ROOT / "docs" / "e2e" / "v7" / "index.html",
    ]),
    ("scripts/generate_storymode.py", [
        ROOT / "docs" / "showcase" / "STORYMODE.html",
    ]),
    ("scripts/generate_cost_simulation.py", [
        ROOT / "docs" / "economics" / "COST_SIMULATION.md",
    ]),
    ("scripts/generate_federated_learning_report.py", [
        ROOT / "docs" / "federated" / "FEDERATED_LEARNING.md",
    ]),
    ("scripts/run_redteam_v4.py", [
        ROOT / "docs" / "adversarial" / "red_team_v4.json",
    ]),
    ("scripts/generate_counterfactual_ui.py", [
        ROOT / "docs" / "ui" / "counterfactual.html",
    ]),
    ("scripts/generate_openapi.py", [
        ROOT / "docs" / "api" / "openapi.json",
    ]),
    ("scripts/generate_module_catalog.py", [
        ROOT / "docs" / "MODULE_CATALOG.md",
    ]),
]


def _run_step(
    cmdline: str, expected_outputs: list[Path],
) -> tuple[bool, str, float]:
    parts = cmdline.split()
    cmd = [sys.executable] + [str(ROOT / parts[0])] + parts[1:]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env,
            capture_output=True, text=True, timeout=900,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout after 900s", time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        return False, (
            proc.stderr.strip().splitlines()[-1]
            if proc.stderr.strip() else "non-zero exit"
        ), elapsed
    missing = [p for p in expected_outputs if not p.exists()]
    if missing:
        return False, f"missing outputs: {missing}", elapsed
    return True, "ok", elapsed


def main() -> int:
    print(
        f"# Building TrustedRisk artefacts ({len(_PIPELINE)} steps)"
    )
    n_ok = 0
    n_fail = 0
    failed: list[tuple[str, str]] = []
    for cmdline, outputs in _PIPELINE:
        print(f"  -> {cmdline}", end=" ", flush=True)
        ok, msg, elapsed = _run_step(cmdline, outputs)
        status = "OK" if ok else "FAIL"
        print(f"[{status}, {elapsed:.1f}s] {msg}")
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            failed.append((cmdline, msg))
    print()
    print(f"# Summary: {n_ok} OK / {n_fail} FAIL "
          f"({len(_PIPELINE)} total)")
    if failed:
        print("# Failures:")
        for cmd, msg in failed:
            print(f"  - {cmd}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
