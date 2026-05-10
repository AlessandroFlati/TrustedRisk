"""trustedrisk-evidence — Phase 1 federation specialist (port 8772).

Clinician evidence-retrieval surface. Skill catalog focused on
grounded differential diagnosis, conversational SHARP patient
resolution, treatment selection (guideline-grounded), external
knowledge (PubMed + ClinicalTrials.gov + NIH RePORTER + drug
pricing), chart intelligence (NER + NegEx + structurer).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_evidence.server:app --port 8772
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-evidence",
    port=8772,
    agent_card_path="apps/specialist_evidence/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_EVIDENCE_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
