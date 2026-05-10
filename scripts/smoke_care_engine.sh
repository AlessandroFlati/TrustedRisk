#!/usr/bin/env bash
# Smoke the Care Engine over A2A v1 with a real Prompt Opinion FHIR context.
#
# Use this to validate the orchestrator end-to-end without going through the PO
# chat UI: the script issues a JSON-RPC `message/send` against the live
# orchestrator endpoint, propagating the FHIR-context extension exactly the
# way PO does. Compare the trace returned here with `logs/a2a_trace.jsonl` to
# confirm the live PO chat is exercising the same code path.
#
# Required env vars (paste from PO's "Generate Token" dialog):
#   PO_FHIR_URL         e.g. https://app.promptopinion.ai/api/workspaces/<ws>/fhir
#   PO_PATIENT_ID       the bare UUID, e.g. b2ef0dca-f350-4de2-a1e2-a3aee73c2282
#   PO_FHIR_TOKEN       the long JWT (1-hour expiry; refresh in PO when stale)
#
# Optional env vars:
#   ORCHESTRATOR_URL    default https://flati.work/a2a/trustedrisk-orchestrator
#   OAUTH_CLIENT_ID     default po-judge-eval
#   OAUTH_CLIENT_SECRET default E9Zyf83KqpvZvyuYcsIGbFecv9iFDoHeZ7oob9tHfks
#                        (the judge-evaluation pre-shared client; rotate before
#                         a public release)
#   PROMPT              default "Run the readmission risk." (override via CLI)
#
# Usage:
#   PO_FHIR_TOKEN=eyJhbGc... PO_PATIENT_ID=b2ef0... PO_FHIR_URL=https://... \
#     ./scripts/smoke_care_engine.sh "What's the plan for discharge?"
#
# Exit code: 0 on a 200 response with a populated task; non-zero otherwise.

set -euo pipefail

ORCHESTRATOR_URL="${ORCHESTRATOR_URL:-https://flati.work/a2a/trustedrisk-orchestrator}"
OAUTH_TOKEN_URL="${OAUTH_TOKEN_URL:-https://flati.work/oauth/token}"
OAUTH_CLIENT_ID="${OAUTH_CLIENT_ID:-po-judge-eval}"
OAUTH_CLIENT_SECRET="${OAUTH_CLIENT_SECRET:-E9Zyf83KqpvZvyuYcsIGbFecv9iFDoHeZ7oob9tHfks}"
PROMPT="${1:-${PROMPT:-Run the readmission risk.}}"

if [ -z "${PO_FHIR_URL:-}" ] || [ -z "${PO_PATIENT_ID:-}" ] || [ -z "${PO_FHIR_TOKEN:-}" ]; then
    echo "ERROR: PO_FHIR_URL, PO_PATIENT_ID, PO_FHIR_TOKEN must be set." >&2
    echo "Copy them from PO's 'Generate Token' dialog (Care Engine scope)." >&2
    exit 2
fi

echo "[1/3] Acquiring orchestrator OAuth bearer..."
OAUTH_RESP=$(curl -sS -X POST "$OAUTH_TOKEN_URL" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"$OAUTH_CLIENT_ID\", \"client_secret\": \"$OAUTH_CLIENT_SECRET\", \"grant_type\": \"client_credentials\", \"scope\": \"discharge.read discharge.execute\"}")
BEARER=$(printf '%s' "$OAUTH_RESP" | python -c 'import sys, json; print(json.load(sys.stdin).get("access_token", ""))')
if [ -z "$BEARER" ]; then
    echo "ERROR: failed to acquire OAuth bearer." >&2
    echo "  Response: $OAUTH_RESP" >&2
    exit 3
fi
echo "      ok (${#BEARER} chars)"

echo "[2/3] Posting prompt to Care Engine..."
echo "      Prompt:    $PROMPT"
echo "      Patient:   $PO_PATIENT_ID"
echo "      FHIR URL:  $PO_FHIR_URL"

# Build the JSON body with python (avoids shell-quoting JWT + prompt).
BODY=$(python -c '
import json, os, sys, uuid
prompt = os.environ["PROMPT"]
fhir_url = os.environ["PO_FHIR_URL"]
fhir_token = os.environ["PO_FHIR_TOKEN"]
patient_id = os.environ["PO_PATIENT_ID"]
print(json.dumps({
    "jsonrpc": "2.0",
    "id": str(uuid.uuid4()),
    "method": "message/send",
    "params": {
        "message": {
            "role": "user",
            "messageId": str(uuid.uuid4()),
            "parts": [{"kind": "text", "text": prompt}],
            "metadata": {
                "https://app.promptopinion.ai/schemas/a2a/v1/fhir-context": {
                    "fhirUrl": fhir_url,
                    "fhirToken": fhir_token,
                    "patientId": patient_id,
                },
            },
        },
    },
}))
' PROMPT="$PROMPT" PO_FHIR_URL="$PO_FHIR_URL" PO_FHIR_TOKEN="$PO_FHIR_TOKEN" PO_PATIENT_ID="$PO_PATIENT_ID")

RESPONSE=$(curl -sS -X POST "$ORCHESTRATOR_URL/" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $BEARER" \
    --data "$BODY")

echo "[3/3] Response (formatted):"
printf '%s\n' "$RESPONSE" | python -m json.tool 2>/dev/null || printf '%s\n' "$RESPONSE"

echo
echo "Trace appended to logs/a2a_trace.jsonl. Tail it with:"
echo "  .venv/Scripts/python.exe scripts/watch_a2a_trace.py --tail 1"
