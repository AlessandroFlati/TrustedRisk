"""Phase 14.17 P1 -- Render the regulatory pack to disk.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_regulatory_pack.py

Writes:
  - docs/regulatory/REGULATORY_PACK.md     (human-readable)
  - docs/regulatory/regulatory_pack.json   (machine-ingestible)

Pure-deterministic. Same artefact set -> byte-identical output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.regulatory_pack import (
    build_regulatory_pack, render_regulatory_pack_md,
)


def main() -> int:
    out_dir = ROOT / "docs" / "regulatory"
    out_dir.mkdir(parents=True, exist_ok=True)

    pack = build_regulatory_pack(project_root=ROOT)

    md = render_regulatory_pack_md(pack)
    md_path = out_dir / "REGULATORY_PACK.md"
    md_path.write_text(md, encoding="utf-8")

    json_path = out_dir / "regulatory_pack.json"
    json_path.write_text(
        json.dumps(pack.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(
        f"Sections: {pack.n_sections}, "
        f"artefact coverage: {pack.overall_artefact_coverage*100:.1f}%."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
