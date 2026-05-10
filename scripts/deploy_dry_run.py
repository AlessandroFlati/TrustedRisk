"""Phase 12.8 C1 -- Cloud Run deployment dry-run validator.

Validates that every service entry in `scripts/deploy_cloudrun.sh`:
  1. Imports its declared `AGENT_MODULE:app` symbol successfully.
  2. Surfaces a `/healthz` endpoint that returns 200 in-process.
  3. Has a unique `SERVICE_NAME`.

Does NOT invoke gcloud / docker / build steps -- that's the user's
manual handoff per the standing scope rule. This script is the
'pre-deploy lint' that catches signature drift before the user runs
`gcloud run deploy` against their account.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/deploy_dry_run.py
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DEPLOY_SH = ROOT / "scripts" / "deploy_cloudrun.sh"
DOCS_DIR = ROOT / "docs" / "deployment"


def _parse_services_array(text: str) -> list[tuple[str, str, str]]:
    """Pull (service_name, agent_module, min_instances) from
    `SERVICES=( "name|module|min" ... )` in the bash deploy script."""
    m = re.search(r"SERVICES=\(\s*([\s\S]*?)\)", text)
    if not m:
        raise RuntimeError("SERVICES=(...) array not found in deploy script.")
    lines = m.group(1).splitlines()
    rows: list[tuple[str, str, str]] = []
    for line in lines:
        ln = line.strip().strip('"').strip("'")
        if not ln or ln.startswith("#"):
            continue
        # strip leading/trailing quotes left from `"a|b|c"` patterns
        ln = ln.strip().strip('"')
        parts = ln.split("|")
        if len(parts) != 3:
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def _load_app(agent_module: str) -> Any:
    """Import `module:attr` and return the ASGI app object. Raises
    on any import failure or missing attribute."""
    if ":" not in agent_module:
        raise ValueError(
            f"AGENT_MODULE must be in 'module:attr' form, got "
            f"{agent_module!r}"
        )
    mod_name, attr = agent_module.split(":", 1)
    mod = importlib.import_module(mod_name)
    if not hasattr(mod, attr):
        raise AttributeError(
            f"Module {mod_name!r} has no attribute {attr!r}"
        )
    target = getattr(mod, attr)
    # If `attr` is a factory like `build_http_app`, call it.
    if callable(target) and not hasattr(target, "router"):
        target = target()
    return target


def _validate_one(service_name: str, agent_module: str) -> dict[str, Any]:
    from starlette.testclient import TestClient
    out: dict[str, Any] = {
        "service_name": service_name, "agent_module": agent_module,
        "import_ok": False, "healthz_status": None,
        "agent_card_status": None, "error": None,
    }
    try:
        app = _load_app(agent_module)
        out["import_ok"] = True
        client = TestClient(app)
        try:
            r = client.get("/healthz")
            out["healthz_status"] = r.status_code
        except Exception as exc:    # noqa: BLE001
            out["error"] = f"healthz: {type(exc).__name__}: {exc}"
        try:
            r = client.get("/.well-known/agent-card.json")
            out["agent_card_status"] = r.status_code
        except Exception:
            # Some services (cds_hooks) don't ship an agent-card; that's OK.
            out["agent_card_status"] = None
    except Exception as exc:    # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    text = DEPLOY_SH.read_text(encoding="utf-8")
    services = _parse_services_array(text)
    if not services:
        print("ERROR: no services parsed from deploy_cloudrun.sh")
        return 2

    rows: list[dict[str, Any]] = []
    n_ok = n_fail = 0
    for service_name, agent_module, _min in services:
        row = _validate_one(service_name, agent_module)
        rows.append(row)
        if row["import_ok"] and row["healthz_status"] == 200:
            n_ok += 1
        else:
            n_fail += 1

    # Uniqueness
    duplicate_names: list[str] = []
    seen_names: set[str] = set()
    for s, _, _ in services:
        if s in seen_names:
            duplicate_names.append(s)
        seen_names.add(s)

    artefact = {
        "captured_at_iso": timestamp,
        "n_services": len(services),
        "n_ok": n_ok, "n_fail": n_fail,
        "duplicate_service_names": duplicate_names,
        "rows": rows,
    }
    (DOCS_DIR / "deploy_dry_run.json").write_text(
        json.dumps(artefact, indent=2, default=str), encoding="utf-8",
    )

    md: list[str] = []
    md.append("# Cloud Run Deployment -- Pre-Deploy Dry-Run Validation")
    md.append("")
    md.append(f"**Phase 12.8 -- captured {timestamp}**")
    md.append("")
    md.append(
        f"- Services in `scripts/deploy_cloudrun.sh`: **{len(services)}**\n"
        f"- Imports + `/healthz` 200: **{n_ok}**\n"
        f"- Failures: **{n_fail}**\n"
        f"- Duplicate service names: "
        f"**{len(duplicate_names)}** "
        f"({', '.join(duplicate_names) if duplicate_names else 'none'})"
    )
    md.append("")
    md.append("## Per-service validation")
    md.append("")
    md.append("| Service | Module | Import | /healthz | Agent card | Notes |")
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        notes = r["error"] or "-"
        md.append(
            f"| {r['service_name']} | `{r['agent_module']}` | "
            f"{'✓' if r['import_ok'] else '✗'} | "
            f"{r['healthz_status']} | "
            f"{r['agent_card_status'] if r['agent_card_status'] else '--'} | "
            f"{notes} |"
        )
    md.append("")
    md.append("## How to deploy for real")
    md.append("")
    md.append(
        "The user must run `gcloud auth login` + "
        "`gcloud config set project <id>` once on their workstation, "
        "then:\n\n"
        "```bash\n"
        "make deploy                # all 18 services\n"
        "make deploy-mcp            # only the MCP server\n"
        "make deploy-composer       # only the composer\n"
        "DEPLOY_TARGETS=quality,pophealth,appeals \\\n"
        "    bash scripts/deploy_cloudrun.sh   # subset deploy\n"
        "```\n\n"
        "The dry-run validation in this file confirms every service "
        "boots in-process before the user pays for a Cloud Run rollout."
    )
    md.append("")
    (DOCS_DIR / "DEPLOY_DRY_RUN.md").write_text(
        "\n".join(md), encoding="utf-8",
    )
    print(f"Wrote {DOCS_DIR / 'DEPLOY_DRY_RUN.md'}")
    print(f"Validated {n_ok}/{len(services)} services successfully.")
    return 0 if n_fail == 0 and not duplicate_names else 1


if __name__ == "__main__":
    sys.exit(main())
