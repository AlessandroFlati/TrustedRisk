"""trustedrisk-pediatric -- federation specialist (port 8774).

Pediatric inpatient surface. Skill catalog scoped to early-warning
PEWS + weight-based pediatric dosing.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_pediatric.server:app --port 8774
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-pediatric",
    port=8774,
    agent_card_path="apps/specialist_pediatric/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_PEDIATRIC_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
