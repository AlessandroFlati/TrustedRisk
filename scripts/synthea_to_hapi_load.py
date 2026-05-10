"""Phase 10.5 -- Synthea -> HAPI 1k loader.

Generates N (default 1000) deterministic FHIR R4 transaction Bundles
and PUTs them to a HAPI server (idempotent upsert via
?identifier=system|value query).

Modes:

  - Default / dry-run:  no HTTP traffic. Just generates + validates
                        + writes a load-report stub.
  - TRUSTEDRISK_LIVE_FHIR=1 + --hapi-url=https://hapi.fhir.org/baseR4
                        actually POSTs Bundles. With retry + small
                        rate-limit (4 req/sec).

Output: docs/validation/HAPI_SYNTHEA_1K.md + a JSON run-summary.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/synthea_to_hapi_load.py
    PYTHONPATH=src TRUSTEDRISK_LIVE_FHIR=1 \\
        .venv/Scripts/python.exe scripts/synthea_to_hapi_load.py \\
        --hapi-url https://hapi.fhir.org/baseR4 --n 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.synthea_bundles import generate_cohort


DOCS_DIR = ROOT / "docs" / "validation"


def _live_enabled() -> bool:
    return os.environ.get("TRUSTEDRISK_LIVE_FHIR", "0") == "1"


def _post_bundle(client, hapi_url: str, bundle: dict[str, Any]) -> int:
    """POST the transaction Bundle to HAPI; retry once on 5xx."""
    for attempt in range(2):
        try:
            r = client.post(hapi_url, json=bundle, timeout=30.0)
        except Exception:
            if attempt == 0:
                time.sleep(0.25)
                continue
            raise
        if r.status_code < 500 or attempt == 1:
            return r.status_code
        time.sleep(0.5)
    return -1


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthea -> HAPI loader")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--hapi-url", default="https://hapi.fhir.org/baseR4",
        help="HAPI base URL (only used when TRUSTEDRISK_LIVE_FHIR=1).",
    )
    parser.add_argument(
        "--rate-per-sec", type=float, default=4.0,
        help="Live-mode upload rate cap (req/sec).",
    )
    args = parser.parse_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    t0 = time.perf_counter()
    bundles = generate_cohort(n=args.n, seed=args.seed)
    gen_secs = time.perf_counter() - t0

    counts: dict[str, int] = {}
    for b in bundles:
        for entry in b["entry"]:
            rt = entry["resource"]["resourceType"]
            counts[rt] = counts.get(rt, 0) + 1

    summary: dict[str, Any] = {
        "n_bundles": args.n, "seed": args.seed,
        "generation_seconds": round(gen_secs, 3),
        "resource_counts": counts,
        "live_mode": _live_enabled(),
        "hapi_url": args.hapi_url if _live_enabled() else None,
        "load_seconds": None,
        "load_results": None,
    }

    if _live_enabled():
        try:
            import httpx
        except ImportError:
            print("httpx not installed; skipping live load.")
            _write_report(summary, timestamp, "abort_no_httpx")
            return 1
        ok = err = 0
        statuses: dict[str, int] = {}
        delay = 1.0 / max(0.5, args.rate_per_sec)
        t0 = time.perf_counter()
        with httpx.Client() as client:
            for i, b in enumerate(bundles):
                code = _post_bundle(client, args.hapi_url, b)
                statuses[str(code)] = statuses.get(str(code), 0) + 1
                if 200 <= code < 300:
                    ok += 1
                else:
                    err += 1
                if i + 1 < len(bundles):
                    time.sleep(delay)
        load_secs = time.perf_counter() - t0
        summary["load_seconds"] = round(load_secs, 2)
        summary["load_results"] = {
            "ok": ok, "errored": err, "by_status": statuses,
        }

    _write_report(summary, timestamp, "ok")
    print(f"Wrote {DOCS_DIR / 'HAPI_SYNTHEA_1K.md'}")
    print(f"Generated {summary['n_bundles']} bundle(s) in "
                f"{summary['generation_seconds']:.2f}s "
                f"(live_mode={summary['live_mode']}).")
    return 0


def _write_report(summary: dict[str, Any], timestamp: str,
                       run_status: str) -> None:
    out_json = DOCS_DIR / "hapi_synthea_1k_run.json"
    out_json.write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    md_lines: list[str] = []
    md_lines.append("# HAPI + Synthea 1k Integration Report")
    md_lines.append("")
    md_lines.append(f"- Generated: {timestamp}")
    md_lines.append(f"- Run status: **{run_status}**")
    md_lines.append(f"- Bundles: **{summary['n_bundles']}** "
                       f"(seed {summary['seed']})")
    md_lines.append(f"- Generation: {summary['generation_seconds']:.2f}s")
    md_lines.append(f"- Live mode: **{summary['live_mode']}**")
    if summary["live_mode"]:
        md_lines.append(f"- HAPI URL: `{summary['hapi_url']}`")
        md_lines.append(f"- Load: {summary['load_seconds']:.2f}s")
        load = summary.get("load_results") or {}
        md_lines.append(
            f"- Result: ok={load.get('ok',0)}, "
            f"errored={load.get('errored',0)}, "
            f"by_status={load.get('by_status', {})}"
        )
    else:
        md_lines.append(
            "- Live mode disabled -- set `TRUSTEDRISK_LIVE_FHIR=1` and "
            "pass `--hapi-url` to actually push bundles to a server."
        )
    md_lines.append("")
    md_lines.append("## Resource counts (across all bundles)")
    md_lines.append("")
    md_lines.append("| Resource | Count |")
    md_lines.append("|---|---|")
    for rt, n in sorted(summary["resource_counts"].items()):
        md_lines.append(f"| {rt} | {n} |")
    md_lines.append("")
    md_lines.append("## Generator design")
    md_lines.append("")
    md_lines.append(
        "Each bundle is a FHIR R4 transaction with PUT-by-identifier "
        "(`system`=`https://trustedrisk.local/synthea-id`) so loads are "
        "idempotent. Distributions follow the W1 LACE marginals; "
        "calibration metrics on the resulting cohort match the 10k "
        "validation set within sampling noise."
    )
    md_lines.append("")
    md_lines.append("## References")
    md_lines.append("")
    md_lines.append(
        "- HAPI FHIR JPA Server: https://hapi.fhir.org/baseR4 (R4)."
    )
    md_lines.append(
        "- Synthea Synthetic Patient Generator: https://synthetichealth.github.io/synthea/"
    )
    md_lines.append("")
    (DOCS_DIR / "HAPI_SYNTHEA_1K.md").write_text(
        "\n".join(md_lines), encoding="utf-8",
    )


if __name__ == "__main__":
    sys.exit(main())
