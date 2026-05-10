# TrustedRisk Makefile — operational one-liners for W3 Day 1 promotion flow.

.PHONY: help install dev-install test verify port promote-artifacts serve docker-build docker-run demo-up demo-up-full demo-down demo-status hapi-up hapi-down hapi-seed deploy deploy-mcp deploy-composer deploy-specialists clean redteam-v2 synthea-1k download-fixtures

help:
	@echo "TrustedRisk operational targets:"
	@echo "  install           — install runtime deps"
	@echo "  dev-install       — install runtime + dev deps"
	@echo "  test              — run pytest suite"
	@echo "  verify            — run scripts/verify_artifacts.py"
	@echo "  port              — copy internal calibration workflow outputs into data/ + fixtures/"
	@echo "  promote-artifacts — port + verify + test regression (atomic unit for W3 Day 1)"
	@echo "  serve             — run the MCP server locally on TRUSTEDRISK_PORT"
	@echo "  docker-build      — build the trustedrisk Docker image"
	@echo "  docker-run        — run the image with docker-compose"
	@echo "  deploy            — Cloud Run deploy: 1 MCP + 8 specialists + 1 composer"
	@echo "  deploy-mcp        — Cloud Run deploy: only the main MCP server"
	@echo "  deploy-composer   — Cloud Run deploy: only the composer agent"
	@echo "  deploy-specialists— Cloud Run deploy: only the 8 specialists"
	@echo "  demo-up-full      — bring up the full 14-specialist federation locally"
	@echo "  demo-status       — print health status of each running specialist"
	@echo "  redteam-v2        — run the 110-prompt v2 red-team against detect_phi"
	@echo "  synthea-1k        — generate the Synthea-1k FHIR R4 cohort report"
	@echo "  download-fixtures — fetch the 3 FHIR bundle fixtures (~94 MB) into fixtures/"
	@echo "  clean             — remove data/, fixtures/ (destructive)"

install:
	pip install -e .

dev-install:
	pip install -e .[dev]

test:
	pytest tests/ -v

verify:
	@echo "=== verify_artifacts.py ==="
	python scripts/verify_artifacts.py

port:
	@echo "=== port_calibration_artifacts.sh ==="
	bash scripts/port_calibration_artifacts.sh

promote-artifacts: port verify
	@echo "=== Running regression suite ==="
	pytest tests/regression/ -v || (echo "Regression failed; artifacts staged but not committed" && exit 1)
	@echo "=== Artifacts promoted successfully ==="

serve:
	python -m mcp_server.server

docker-build:
	docker build -t trustedrisk:latest .

docker-run:
	docker-compose up -d

# Bring up trustedrisk-mcp + hapi together (the local-integration setup).
demo-up:
	docker-compose up -d
	@echo "MCP:  http://localhost:8080/mcp"
	@echo "HAPI: http://localhost:8081/fhir/metadata (wait ~60s for first boot)"

demo-down:
	docker-compose down

hapi-up:
	docker-compose up -d hapi
	@echo "HAPI starting at http://localhost:8081/fhir/metadata (wait ~60s)"

hapi-down:
	docker-compose stop hapi

# Seed HAPI with the 3 W2 demo patient bundles (clean / abstain / complex).
# Requires fixtures/ to be populated (run `make port` first).
hapi-seed:
	@for f in fixtures/patient_*.json; do \
	  [ -f "$$f" ] || continue; \
	  echo "POSTing $$f"; \
	  curl -s -X POST -H "Content-Type: application/fhir+json" \
	    --data-binary @"$$f" \
	    http://localhost:8081/fhir > /dev/null && echo "  ok"; \
	done

clean:
	@echo "WARNING: this removes data/ and fixtures/"
	@read -p "Confirm (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf data/ fixtures/

# ─── Phase 6.3 Cloud Run deploy ────────────────────────────────────
# All targets require `gcloud auth login` + a chosen project. The
# script is idempotent — re-running updates the existing services.

deploy:
	bash scripts/deploy_cloudrun.sh

deploy-mcp:
	DEPLOY_TARGETS=mcp bash scripts/deploy_cloudrun.sh

deploy-composer:
	DEPLOY_TARGETS=composer bash scripts/deploy_cloudrun.sh

deploy-specialists:
	DEPLOY_TARGETS=discharge,acute,evidence,population,pedi-mh,pa,scribe,patient \
	    bash scripts/deploy_cloudrun.sh

# ─── Phase 10.10 — Local full-federation demo ──────────────────────
# Brings up all 14 specialists (8770-8784) + composer (8780) using the
# uvicorn ASGI factory directly. PIDs are kept under /tmp/trustedrisk-pids
# so `make demo-down` can stop them.

demo-up-full:
	bash scripts/demo_up_full.sh

demo-status:
	bash scripts/demo_status.sh

# ─── Phase 10.4 / 10.5 driver convenience targets ──────────────────

redteam-v2:
	PYTHONPATH=src python scripts/run_redteam_v2.py

redteam-v3:
	PYTHONPATH=src TRUSTEDRISK_DISABLE_LLM=1 python scripts/run_redteam_v3.py

synthea-1k:
	PYTHONPATH=src python scripts/synthea_to_hapi_load.py --n 1000

synthea-100k:
	PYTHONPATH=src python scripts/synthea_100k_recalibrate.py

mimic-recal:
	PYTHONPATH=src python scripts/mimic_iv_recalibrate.py

deploy-dry-run:
	PYTHONPATH=src python scripts/deploy_dry_run.py

scenario-counterfactuals:
	PYTHONPATH=src python scripts/generate_scenario_counterfactuals_doc.py

streamlit-demo:
	PYTHONPATH=src streamlit run apps/streamlit_demo/app.py

# ─── Fixture download (FHIR bundle test data) ──────────────────────
# The 3 FHIR Bundle fixtures (~94 MB total) are too large to ship in-tree.
# They live in the GitHub Release v0.8.0 and are pulled on demand. Set
# TRUSTEDRISK_FIXTURES_URL to override the default release URL.

download-fixtures:
	python scripts/download_fixtures.py
