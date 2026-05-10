"""trustedrisk-pa — Phase 2.1 Prior Authorization specialist (port 8775).

Pain-point: $40 B/yr US administrative cost; AMA top-3 clinician
burden; payer denial rate ~15-20%. Skill catalog focused on the four
PA-1/2/3/4 tools: evidence pack, payer-rules match, appeal-
likelihood, letter draft.

Run:
    PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \\
        .venv/Scripts/python.exe -m uvicorn \\
        apps.specialist_pa.server:app --port 8775

Marketplace registration: register the public URL serving
`/.well-known/agent-card.json` from this app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app


_CFG = SpecialistConfig(
    name="trustedrisk-pa",
    port=8775,
    agent_card_path="apps/specialist_pa/agent_card.json",
)

app = build_specialist_app(_CFG)


def main() -> None:
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    host = os.environ.get("TRUSTEDRISK_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTEDRISK_PA_PORT", str(_CFG.port)))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
