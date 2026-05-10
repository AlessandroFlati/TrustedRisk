"""trustedrisk-population — Phase 1 federation specialist (port 8773).

Quality-management / population-health surface. Skill catalog focused
on cost-effectiveness analysis (EVOI per intervention), subgroup
fairness audit, equity dashboard with differential-privacy noise.
Cohort-/research-level scope (vs the patient-level scope of the other
four specialists).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_population.server:app --port 8773
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-population",
    port=8773,
    agent_card_path="apps/specialist_population/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_POPULATION_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
