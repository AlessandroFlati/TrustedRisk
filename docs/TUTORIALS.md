# TrustedRisk Tutorials

Practical walkthroughs for the most common integration tasks. Each
section is a self-contained recipe: copy-paste, swap the variables,
run.

---

## §1. Add a new specialist to the federation

A specialist is an A2A agent that publishes a focused skill catalogue
on its own port while sharing the underlying MCP backend. Adding one
takes ~ 15 minutes.

### 1.1 Pick the bundle subset and port

```python
# scripts/generate_specialist_cards.py -- add an entry to SPECIALISTS
{
    "id": "specialist_my_new",
    "name": "trustedrisk-my-new",
    "tagline": "One-line marketing description",
    "port": 8782,                               # next free
    "base_url_placeholder":
        "https://<deploy-url>/a2a/trustedrisk-my-new",
    "skills": ["my_skill_id", "phi_check"],     # IDs from the master agent-card
    "bundles": ["my_bundle_id"],                 # IDs from BUNDLES
},
```

### 1.2 Create the specialist app

```bash
mkdir apps/specialist_my_new
```

```python
# apps/specialist_my_new/__init__.py
"""trustedrisk-my-new specialist."""
from .server import app
__all__ = ["app"]
```

```python
# apps/specialist_my_new/server.py
"""trustedrisk-my-new specialist (port 8782)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apps._shared import SpecialistConfig, build_specialist_app

_CFG = SpecialistConfig(
    name="trustedrisk-my-new", port=8782,
    agent_card_path="apps/specialist_my_new/agent_card.json",
)

app = build_specialist_app(_CFG)
```

### 1.3 Generate the agent-card

```bash
PYTHONPATH=src .venv/Scripts/python.exe \
    scripts/generate_specialist_cards.py
```

### 1.4 Test the boot

```bash
PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
    .venv/Scripts/python.exe -m uvicorn \
    apps.specialist_my_new.server:app --port 8782
```

```bash
curl http://localhost:8782/.well-known/agent-card.json
curl http://localhost:8782/healthz
```

### 1.5 Wire into Cloud Run + the federation test

- Add a service entry in `scripts/deploy_cloudrun.sh` SERVICES array.
- Add a row in `tests/integration/test_specialist_federation.py`
  SPECIALISTS list + the fixture's import + return dict.

---

## §2. Add a new MCP tool to a bundle

Tools live in `src/mcp_server/tools/`. Each is a standalone module
with a `compute_<name>` async function + a `register(mcp)` hook.

### 2.1 Define the schema

```python
# src/shared/schemas.py
class MyNewToolReport(BaseModel):
    primary_field: str
    rationale: str
    references: list[str] = Field(default_factory=list)
```

### 2.2 Implement the tool

```python
# src/mcp_server/tools/my_new_tool.py
"""healthcare.compute_my_new_tool -- what it does."""
from __future__ import annotations
from shared.schemas import MyNewToolReport


async def compute_my_new_tool(
    primary_input: str,
    optional_input: int | None = None,
) -> MyNewToolReport:
    """One-line description (becomes the MCP `description`).

    Args:
        primary_input: ...
        optional_input: ...

    Returns:
        MyNewToolReport with ...
    """
    # Pure-deterministic body. LLM polish is OPTIONAL and goes through
    # `a2a_agent.llm_polish.resolve_polish_client()` with the
    # cite-back post-check enforced.
    return MyNewToolReport(
        primary_field="value",
        rationale="why",
        references=["..."],
    )


def register(mcp) -> None:
    mcp.tool()(compute_my_new_tool)
```

### 2.3 Wire the tool

```python
# src/mcp_server/tools/__init__.py
from . import (
    ...,
    my_new_tool,
)

__all__ = [..., "my_new_tool"]

BUNDLES = {
    ...,
    "my_bundle_id": [
        "compute_my_new_tool",
        "ground_claim",
        "detect_phi",
    ],
}

def register_all(mcp):
    ...
    my_new_tool.register(mcp)
```

```python
# src/mcp_server/scopes.py
BUNDLE_SCOPES = {
    ...,
    "my_bundle_id": [
        "patient/Patient.rs",
        "patient/Condition.rs",
    ],
}
```

### 2.4 Test

```python
# tests/unit/test_my_new_tool.py
import asyncio
from mcp_server.tools.my_new_tool import compute_my_new_tool


def test_tool_basic():
    out = asyncio.run(compute_my_new_tool("hello"))
    assert out.primary_field == "value"
```

---

## §3. Swap calibrated artifacts

The calibrated coefficients live at `data/coefficients.json`. The
runtime reads them through `TRUSTEDRISK_COEFFICIENTS_PATH`. To
deploy a new calibration:

### 3.1 Validate locally

```bash
TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients_v0_8.json \
    .venv/Scripts/python.exe scripts/external_validation.py \
    --synthetic
```

Check that:
- ECE ≤ 0.08 (target ≤ 0.05)
- AUROC change ≤ 0.05 vs prior calibration
- No subgroup calibration drift > 30 % vs prior

### 3.2 Reproducibility check

```bash
TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients_v0_8.json \
    .venv/Scripts/python.exe -m pytest tests/golden/ -v
```

100 % of the 600 golden cohort cases must still pass.

### 3.3 Promote

```bash
mv data/coefficients_v0_8.json data/coefficients.json
git commit data/coefficients.json -m "calibrate: spec_003 (Q2 2026)"
```

### 3.4 PCCP-friendly metadata

The bundled `model_version` field on every RiskEstimate output ties
each decision to a specific calibration. `compute_data_lineage`
emits an `integrity_hash` over the coefficients + training-data manifest,
so the decision can be re-validated against the calibration in effect
at the time it was made -- closing the FDA Pre-determined Change Control
Plan loop.

---

## §4. Integrate with an EHR (Epic / Cerner / etc.)

TrustedRisk is FHIR-server-agnostic per SHARP §3.5 -- same deployment
works against any FHIR R4 endpoint.

### 4.1 Configure the SHARP context

Per request, send:

```
X-FHIR-Server-URL: https://your-ehr.example.com/api/FHIR/R4
X-FHIR-Access-Token: <bearer-token-from-SMART-launch>
X-Patient-ID: Patient/<id>
X-FHIR-Refresh-Token: <optional, for offline access>
X-FHIR-Refresh-Url: <optional refresh endpoint>
```

A SMART-on-FHIR app launches via the standard EHR launch sequence;
the resulting access token is forwarded on every MCP / A2A call.

### 4.2 Multi-tenant OAuth

Enable the OAuth proxy and configure per-tenant FHIR allowlist:

```bash
TRUSTEDRISK_OAUTH_ENABLED=1
TRUSTEDRISK_OAUTH_CLIENT_ID=<your-tenant-id>
TRUSTEDRISK_OAUTH_ALLOWED_FHIR_SERVERS=https://epic.example.com/fhir,https://cerner.example.com/fhir
```

Cross-tenant FHIR queries are blocked at the middleware layer (audit:
`tests/unit/test_oauth.py`, `tests/unit/test_sharp_multitenant.py`).

### 4.3 Audit + reproducibility

The audit log (`a2a_agent.audit`) logs every decision with PHI-redacted
SHA-256 hashing. Bind a Postgres / Redis-backed log adapter for
multi-replica deployments -- the in-process JSONL log is for
single-replica Cloud Run only.

---

## §5. Run the composer end-to-end

```bash
PYTHONPATH=src TRUSTEDRISK_COEFFICIENTS_PATH=data/coefficients.json \
    .venv/Scripts/python.exe -m uvicorn \
    apps.composer.server:app --port 8780 &

curl http://localhost:8780/api/workflows
curl -X POST http://localhost:8780/api/run/chf_admission \
    -H "Content-Type: application/json" \
    -d '{
      "inputs": {
        "chief_complaint": "shortness of breath",
        "vital_signs": {"heart_rate": 122, "spo2": 91, ...},
        "age": 72,
        "patient_id": "Patient/example",
        "fhir_bundle": {...},
        "medications": ["furosemide", "lisinopril"]
      }
    }'
```

The response is a complete trace: per-step `tool_name`, `specialist`,
`duration_ms`, output dump, error (if any).

---

## §6. Enable real LLM polish

Default = deterministic floor (no LLM). To enable polish per-tool:

```bash
TRUSTEDRISK_PA_LLM_POLISH=1            # PA letter polish
TRUSTEDRISK_SCRIBE_LLM_POLISH=1        # scribe note polish
TRUSTEDRISK_PATIENT_LLM_POLISH=1       # patient Q&A polish
```

Resolution chain (`a2a_agent.llm_polish.resolve_polish_client`):
1. Ollama at `localhost:11434` (default; `TRUSTEDRISK_OLLAMA_MODEL`
   = `llama3.1:8b`)
2. Gemini API when `GOOGLE_API_KEY` is set
   (`TRUSTEDRISK_GEMINI_MODEL` = `gemini-2.0-flash-exp` default)
3. Null client (no polish)

Hallucination post-check is built in: every polish call asserts that
cite-back IDs / drug names / ICD codes / dose values present in the
source survive verbatim in the output. If any drops, the polish is
**rejected** and the deterministic floor is published instead.

---

## §7. Deploy to Google Cloud Run

```bash
# One-time
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com

# Deploy all 12 services
make deploy

# Or selectively
DEPLOY_TARGETS=composer,discharge bash scripts/deploy_cloudrun.sh
```

Each specialist gets its own Cloud Run service, public URL, and
agent-card endpoint. Register the agent-card URLs at
`https://app.promptopinion.ai/` (External Agents -> Add Connection).

Free-tier sustains the entire federation at low traffic
(2 M requests/month, scale-to-zero default).

---

## §8. Common gotchas

| Issue | Cause | Fix |
|---|---|---|
| `FhirAuthRequired` 401 | Access token expired mid-task; refresh failed | Send `X-FHIR-Refresh-Token` + `X-FHIR-Refresh-Url` (offline access) |
| 403 `missing_fhir_context` | SHARP middleware can't find `X-FHIR-Server-URL` | Add the header; check casing -- `X-FHIR-Server-URL`, not `x-fhir-server-url` (HTTP is case-insensitive but middleware reads exact-case for clarity) |
| Composer step abstains, halts workflow | Default `stop_on_abstain=True` | Mark step `optional=True` OR pass `stop_on_abstain=False` to `execute_workflow` |
| Polish never fires despite `*_LLM_POLISH=1` | No LLM configured (Ollama down + no GOOGLE_API_KEY) | Check `GET /api/llm/polish/status` from the playground |
| Specialist returns `tools_registered=79` everywhere | Same shared backend; specialist filters surface, not registration | This is by design -- see `docs/ARCHITECTURE.md` §1 |
| Rate-limited 429 with low traffic | Burst cap default 120 | Bump `TRUSTEDRISK_RATE_LIMIT_BURST` |
