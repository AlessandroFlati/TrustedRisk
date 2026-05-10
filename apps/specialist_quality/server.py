"""trustedrisk-quality — Phase 10.1 HEDIS / CMS Stars Rating specialist (port 8782).

Outpatient-quality + value-based-care surface. 3 tools:
compute_quality_measures_aggregate (STARS-1),
compute_stars_rating_forecast (STARS-2),
compute_care_gap_priority_ranking (STARS-3).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_quality.server:app --port 8782
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-quality",
    port=8782,
    agent_card_path="apps/specialist_quality/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_QUALITY_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
