# TrustedRisk Local Demo

**Updated:** 2026-04-30 05:53 UTC

This page describes how to bring up the **full 14-specialist federation** locally and exercise the Phase 10 expansion (HEDIS Stars + Population Health + Insurance Appeals).

---

## Prerequisites

- Python 3.13 with the project virtualenv activated
- `data/coefficients.json` present (calibration artefact)
- `make install` (or `pip install -e .[dev]`) completed

---

## One-shot full federation

```bash
make demo-up-full
```

Brings up the 14 specialists + composer in the background under `/tmp/trustedrisk-pids/<id>.pid` (logs in `/tmp/trustedrisk-logs/`):

| Specialist | Port |
|---|---|
| trustedrisk-discharge | 8770 |
| trustedrisk-acute | 8771 |
| trustedrisk-evidence | 8772 |
| trustedrisk-population | 8773 |
| trustedrisk-pedi-mh | 8774 |
| trustedrisk-pa | 8775 |
| trustedrisk-scribe | 8776 |
| trustedrisk-patient | 8777 |
| trustedrisk-coder | 8778 |
| trustedrisk-pgx | 8779 |
| **composer** | **8780** |
| trustedrisk-preadmit | 8781 |
| **trustedrisk-quality (10.1)** | **8782** |
| **trustedrisk-pophealth (10.2)** | **8783** |
| **trustedrisk-appeals (10.3)** | **8784** |

Subset launch is supported -- pass the ids on the command line:

```bash
bash scripts/demo_up_full.sh quality pophealth appeals
```

---

## Health check

```bash
make demo-status
```

Sample output:

```
id           port   pid          status     tools/bundles
------------ ------ ------------ ---------- -----
discharge    8770   34521        running    89/30
acute        8771   34522        running    89/30
...
quality      8782   34534        running    89/30
pophealth    8783   34535        running    89/30
appeals      8784   34536        running    89/30
```

Every specialist must report **89 tools / 30 bundles** (drift is a regression).

---

## Exercising the Phase-10 surface

### HEDIS / CMS Stars (port 8782)

```bash
curl -s http://127.0.0.1:8782/.well-known/agent-card.json | jq '.skills[].id'
```

Use the canned notebook walk-through:

```bash
jupyter notebook docs/notebooks/02_hedis_stars_workflow.ipynb
```

### Population health + outbreak (port 8783)

The bundle ships `compute_syndromic_surveillance`, `compute_vaccine_reminder_cohort`, and `compute_outbreak_heatmap` (DP-noised). The agent-card declares the `patient/Immunization.rs` SMART-on-FHIR scope on top of the standard set.

### Insurance appeals (port 8784)

The bundle ships the deterministic 3-step appeal pipeline (`compute_denial_letter_parse -> compute_appeal_letter_draft -> compute_appeal_escalation_path`) with a built-in 7-payer keyword table (UnitedHealthcare / Anthem / Aetna / Cigna / Humana / Medicare / Medicaid / generic).

---

## Adversarial smoke

Re-run the v2 red-team campaign:

```bash
make redteam-v2
```

Outputs:

- `docs/adversarial/RED_TEAM_RESULTS.md` -- markdown report
- `docs/adversarial/red_team_run.json` -- full structured `RedTeamReport`

Current build: **110/110 pass** against `detect_phi`.

---

## Synthea 1k integration

```bash
make synthea-1k
```

Generates 1,000 deterministic FHIR R4 transaction Bundles in ~ 0.04 s and writes the load report to `docs/validation/HAPI_SYNTHEA_1K.md`. Set `TRUSTEDRISK_LIVE_FHIR=1` to also push them to a live HAPI server (default `https://hapi.fhir.org/baseR4`).

---

## Stopping the federation

Each specialist runs as a backgrounded `uvicorn` with its PID under `/tmp/trustedrisk-pids/`:

```bash
for f in /tmp/trustedrisk-pids/*.pid; do
  kill "$(cat "$f")" 2>/dev/null
  rm "$f"
done
```

(`make demo-down` continues to control the docker-compose stack.)

---

## Test harness

The full-federation smoke test lives at `tests/integration/test_full_federation_smoke.py` and runs **without subprocesses** -- it exercises the same ASGI factories that `demo-up-full` launches. Run with:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m pytest tests/integration/test_full_federation_smoke.py -v
```

Current build: **3,166+ tests passing across the suite.**
