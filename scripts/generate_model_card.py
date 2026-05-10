"""Phase 16.F1 - Render the Model Card + Datasheet artefacts.

Run:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/generate_model_card.py

Outputs:
    docs/research/MODEL_CARD.md
    docs/research/model_card.json
    docs/research/DATASHEET.md
    docs/research/datasheet.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from a2a_agent.model_card import (
    build_datasheet, build_model_card,
    render_datasheet_md, render_model_card_md,
)


def main() -> int:
    out_dir = ROOT / "docs" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)

    card = build_model_card(project_root=ROOT)
    (out_dir / "MODEL_CARD.md").write_text(
        render_model_card_md(card), encoding="utf-8")
    (out_dir / "model_card.json").write_text(
        json.dumps(card.model_dump(mode="json"), indent=2),
        encoding="utf-8")

    ds = build_datasheet()
    (out_dir / "DATASHEET.md").write_text(
        render_datasheet_md(ds), encoding="utf-8")
    (out_dir / "datasheet.json").write_text(
        json.dumps(ds.model_dump(mode="json"), indent=2),
        encoding="utf-8")

    print(f"Wrote {out_dir / 'MODEL_CARD.md'} ({card.n_sections} sections)")
    print(f"Wrote {out_dir / 'DATASHEET.md'} ({ds.n_sections} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
