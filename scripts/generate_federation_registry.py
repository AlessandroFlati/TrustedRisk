"""Phase 17.AI - Render the federation registry artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.federation_registry import (
    build_federation_registry,
    render_marketplace_manifest,
    render_registry_md,
)


def main() -> int:
    out_dir = ROOT / "docs" / "federation"
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = build_federation_registry(apps_dir=ROOT / "apps")
    (out_dir / "FEDERATION_REGISTRY.md").write_text(
        render_registry_md(reg), encoding="utf-8")
    (out_dir / "federation_registry.json").write_text(
        json.dumps(reg.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    manifest = render_marketplace_manifest(reg)
    (out_dir / "marketplace_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"{reg.n_specialists} specialists, "
        f"{len(reg.bundles_covered)} bundles covered, "
        f"{len(reg.bundles_missing)} missing "
        f"({reg.coverage_percent:.1f}% coverage)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
