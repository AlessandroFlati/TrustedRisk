"""trustedrisk-discharge — Phase 1 federation specialist (port 8770).

Hospitalist / case-manager surface. Skill catalog focused on the
adult inpatient discharge workflow + outpatient medication review:
LACE-calibrated readmission risk, medication reconciliation, fairness
audit, discharge counseling, multilingual rendering, HEDIS care-gap
detection, PROM-influenced action ranking.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_discharge.server:app --port 8770

Marketplace registration: register the public URL serving
`/.well-known/agent-card.json` from this app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-discharge",
    port=8770,
    agent_card_path="apps/specialist_discharge/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_DISCHARGE_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
