"""trustedrisk-coder — Phase 7.1 auto-coding specialist (port 8778).

Pain-point: revenue cycle, $30-50 B/yr in coding errors. 4 tools:
ICD-10 suggest + CPT suggest + HCPCS J-code suggest + coding audit
(documented_not_coded / coded_not_documented / specificity_loss).

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_coder.server:app --port 8778
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-coder",
    port=8778,
    agent_card_path="apps/specialist_coder/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_CODER_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
