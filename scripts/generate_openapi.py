"""Phase 17.V - Render the OpenAPI 3.1 spec for the federation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.openapi_generator import build_openapi


def main() -> int:
    out_dir = ROOT / "docs" / "api"
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = build_openapi(server_url="https://trustedrisk.local")
    payload = spec.model_dump(mode="json")
    json_path = out_dir / "openapi.json"
    json_path.write_text(json.dumps(payload, indent=2),
                         encoding="utf-8")
    print(f"Wrote {json_path}")
    print(
        f"openapi {spec.openapi}, "
        f"{len(spec.paths)} paths, "
        f"{len(spec.components.get('schemas', {}))} schemas, "
        f"{len(spec.tags)} tags."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
