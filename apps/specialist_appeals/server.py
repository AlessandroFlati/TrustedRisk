"""trustedrisk-appeals — Phase 10.3 insurance-appeals specialist (port 8784).

Patient-advocate / RCM surface. 3 tools:
compute_denial_letter_parse (APPEALS-1),
compute_appeal_letter_draft (APPEALS-2),
compute_appeal_escalation_path (APPEALS-3).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_appeals.server:app --port 8784
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-appeals",
    port=8784,
    agent_card_path="apps/specialist_appeals/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_APPEALS_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
