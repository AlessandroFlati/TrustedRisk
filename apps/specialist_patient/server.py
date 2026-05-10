"""trustedrisk-patient — Phase 2.3 Post-discharge Q&A specialist (port 8777).

Demonstrates A2A composition explicitly: the BYO orchestrator on the
Prompt Opinion platform consults `trustedrisk-discharge` for the
DecisionCard, then calls THIS specialist's `compute_discharge_qa` /
`compute_medication_what_if` / `compute_caregiver_handoff` tools to
produce the patient-language layer.

Multi-turn continuity is supported: each tool accepts an optional
`multi_turn_context_id` corresponding to the A2A `contextId`, so the
BYO orchestrator can keep a coherent conversation across turns.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_patient.server:app --port 8777
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-patient",
    port=8777,
    agent_card_path="apps/specialist_patient/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_PATIENT_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
