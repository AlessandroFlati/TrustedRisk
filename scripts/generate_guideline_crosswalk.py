"""Phase 17.P - Render the clinical guideline crosswalk artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.guideline_crosswalk import (
    build_guideline_crosswalk, render_guideline_crosswalk_md,
)


def main() -> int:
    out_dir = ROOT / "docs" / "guidelines"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = build_guideline_crosswalk()
    (out_dir / "GUIDELINE_CROSSWALK.md").write_text(
        render_guideline_crosswalk_md(rep), encoding="utf-8")
    (out_dir / "guideline_crosswalk.json").write_text(
        json.dumps(rep.model_dump(mode="json"), indent=2),
        encoding="utf-8")
    print(
        f"Mapped {rep.n_rows} tools "
        f"({rep.n_with_specific_guideline} with tool-specific "
        f"guideline, "
        f"{rep.n_rows - rep.n_with_specific_guideline} on bundle "
        f"defaults)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
