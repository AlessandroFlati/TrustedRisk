"""trustedrisk-pophealth — Phase 10.2 population-health specialist (port 8783).

ACO + public-health surveillance. 3 tools:
compute_syndromic_surveillance (POPHEALTH-1),
compute_vaccine_reminder_cohort (POPHEALTH-2),
compute_outbreak_heatmap (POPHEALTH-3 — DP-noised).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_pophealth.server:app --port 8783
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-pophealth",
    port=8783,
    agent_card_path="apps/specialist_pophealth/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_POPHEALTH_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
