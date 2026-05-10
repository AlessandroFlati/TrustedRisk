#!/usr/bin/env bash
# Phase 6.3 — Cloud Run deploy for the TrustedRisk federation.
#
# Idempotent. Builds a single image (one Dockerfile, AGENT_MODULE
# selected at runtime), pushes to Artifact Registry, then deploys 10
# Cloud Run services — 1 main MCP server + 9 specialists + 1 composer.
#
# Prerequisites (one-time):
#   gcloud auth login
#   gcloud config set project YOUR_PROJECT_ID
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
#       artifactregistry.googleapis.com
#   echo -n "$GOOGLE_API_KEY" | gcloud secrets create google-api-key \
#       --data-file=-      # only when LLM polish is wanted
#
# Run:
#   bash scripts/deploy_cloudrun.sh                # deploys all services
#   DEPLOY_TARGETS=mcp,composer bash scripts/deploy_cloudrun.sh
#                                                    # deploys a subset
#
# Note: the user must run gcloud manually per the standing scope rule.
# This script is provided so the *invocation* is one shell command —
# but it must be run by hand against the user's Google Cloud account.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-trustedrisk}"
IMAGE_TAG="${IMAGE_TAG:-v0.7.0}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/trustedrisk:${IMAGE_TAG}"

if [ -z "${PROJECT_ID}" ]; then
    echo "ERROR: PROJECT_ID is not set and gcloud has no default project."
    echo "       Run \`gcloud config set project YOUR_PROJECT_ID\` first."
    exit 1
fi

# ─────────────────────── Service catalogue ───────────────────────
#
# Each entry: SERVICE_NAME|AGENT_MODULE|MIN_INSTANCES
# Updated in Phase 12.8 to cover the 15-specialist + composer + cds_hooks
# federation. The MIN_INSTANCES default is 0 (cold-start) for Cloud Run
# free-tier friendliness; bump to 1 for the production-tier services.
SERVICES=(
    "trustedrisk-mcp|mcp_server.server:build_http_app|0"
    "trustedrisk-discharge|apps.specialist_discharge.server:app|0"
    "trustedrisk-acute|apps.specialist_acute.server:app|0"
    "trustedrisk-evidence|apps.specialist_evidence.server:app|0"
    "trustedrisk-population|apps.specialist_population.server:app|0"
    "trustedrisk-pediatric|apps.specialist_pediatric.server:app|0"
    "trustedrisk-pa|apps.specialist_pa.server:app|0"
    "trustedrisk-scribe|apps.specialist_scribe.server:app|0"
    "trustedrisk-patient|apps.specialist_patient.server:app|0"
    "trustedrisk-coder|apps.specialist_coder.server:app|0"
    "trustedrisk-pgx|apps.specialist_pgx.server:app|0"
    "trustedrisk-preadmit|apps.specialist_preadmit.server:app|0"
    "trustedrisk-quality|apps.specialist_quality.server:app|0"
    "trustedrisk-pophealth|apps.specialist_pophealth.server:app|0"
    "trustedrisk-appeals|apps.specialist_appeals.server:app|0"
    "trustedrisk-multimodal|apps.specialist_multimodal.server:app|0"
    "trustedrisk-composer|apps.composer.server:app|0"
    "trustedrisk-cds-hooks|apps.cds_hooks.server:app|0"
)

# Filter targets via DEPLOY_TARGETS=foo,bar (comma-separated short ids
# matching the suffix after `trustedrisk-`)
filter_services() {
    local raw="${DEPLOY_TARGETS:-}"
    if [ -z "$raw" ]; then
        printf '%s\n' "${SERVICES[@]}"
        return 0
    fi
    IFS=',' read -ra wanted <<< "$raw"
    for entry in "${SERVICES[@]}"; do
        IFS='|' read -ra parts <<< "$entry"
        suffix="${parts[0]#trustedrisk-}"
        for w in "${wanted[@]}"; do
            if [ "$w" = "$suffix" ] || [ "$w" = "${parts[0]}" ]; then
                printf '%s\n' "$entry"
                continue 2
            fi
        done
    done
}

ACTIVE_SERVICES="$(filter_services)"
if [ -z "$ACTIVE_SERVICES" ]; then
    echo "ERROR: DEPLOY_TARGETS=${DEPLOY_TARGETS:-(empty)} matched no service."
    exit 1
fi

# ─────────────────────── Build + push ───────────────────────
echo ">>> Building image: ${IMAGE_URI}"
gcloud builds submit . \
    --tag "${IMAGE_URI}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}"

# ─────────────────────── Deploy ───────────────────────
echo "$ACTIVE_SERVICES" | while IFS='|' read -r SERVICE_NAME AGENT_MODULE MIN_INSTANCES; do
    echo ""
    echo ">>> Deploying ${SERVICE_NAME} (AGENT_MODULE=${AGENT_MODULE})"
    gcloud run deploy "${SERVICE_NAME}" \
        --image "${IMAGE_URI}" \
        --region "${REGION}" \
        --project "${PROJECT_ID}" \
        --platform managed \
        --allow-unauthenticated \
        --min-instances "${MIN_INSTANCES}" \
        --max-instances 10 \
        --memory 1Gi \
        --cpu 1 \
        --set-env-vars "AGENT_MODULE=${AGENT_MODULE}" \
        --set-env-vars "TRUSTEDRISK_OAUTH_ENABLED=0" \
        --set-env-vars "TRUSTEDRISK_RATE_LIMIT_PER_SEC=60" \
        --set-env-vars "TRUSTEDRISK_RATE_LIMIT_BURST=120" \
        --set-env-vars "OTEL_SERVICE_NAME=${SERVICE_NAME}" \
        --quiet

    URL="$(gcloud run services describe "${SERVICE_NAME}" \
        --region "${REGION}" --project "${PROJECT_ID}" \
        --format 'value(status.url)')"
    echo "    URL: ${URL}"
    echo "    Agent card: ${URL}/.well-known/agent-card.json"
    echo "    Healthz:    ${URL}/healthz"
    echo "    Metrics:    ${URL}/metrics"
done

echo ""
echo ">>> Deployment complete. Register the agent-card URLs at"
echo "    https://app.promptopinion.ai/ (External Agents)."
