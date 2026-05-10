"""trustedrisk-preadmit — Phase 7.5 pre-arrival triage specialist (port 8781).

Patient-facing intake. 3 tools: compute_symptom_red_flag_check,
compute_when_to_seek_care, compute_symptom_followup_questions.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_preadmit.server:app --port 8781
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-preadmit",
    port=8781,
    agent_card_path="apps/specialist_preadmit/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_PREADMIT_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
