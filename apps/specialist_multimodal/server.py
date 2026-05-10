"""trustedrisk-multimodal — Phase 11.4 multi-modal specialist (port 8786).

Signal + imaging surface. 2 tools:
compute_ecg_qt_analyzer (MULTIMODAL-1),
compute_dicom_sr_ingest (MULTIMODAL-2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-multimodal",
    port=8786,
    agent_card_path="apps/specialist_multimodal/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get(
        "TRUSTEDRISK_MULTIMODAL_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
