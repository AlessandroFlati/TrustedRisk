"""trustedrisk-scribe — Phase 2.2 Clinical Documentation specialist (port 8776).

Pain-point: clinician documentation burden is the #1 cited driver of
burnout in the Mayo Clinic 2024 survey (~ 2-3 h/day per clinician).
Skill catalog focused on the four SCRIBE-1/2/3/4 tools: progress note,
discharge summary, consult letter, admission H&P. Every tool ships a
deterministic-template floor with mandatory cite-back to FHIR resource
IDs; optional LLM polish (`TRUSTEDRISK_SCRIBE_LLM_POLISH=1`) paraphrases
prose while preserving the cite-back map.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_scribe.server:app --port 8776
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-scribe",
    port=8776,
    agent_card_path="apps/specialist_scribe/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_SCRIBE_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
