"""trustedrisk-acute — Phase 1 federation specialist (port 8771).

Emergency department / acute-care surface. Skill catalog focused on
ED triage, deterioration nowcast, stroke + ACS reperfusion, trauma /
massive transfusion, DKA, contrast-safety, AKI staging, MEOWS,
geriatric falls + delirium.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_acute.server:app --port 8771
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-acute",
    port=8771,
    agent_card_path="apps/specialist_acute/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_ACUTE_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
