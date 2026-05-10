#!/usr/bin/env bash
# scripts/port_calibration_artifacts.sh — stage internal calibration workflow outputs into TrustedRisk.
#
# Finds the most recent successful run of each calibration workflow and
# copies its output artifacts into data/ + fixtures/. Then merges any
# trustedrisk_env_updates.env artifacts into .env.
#
# Usage:
#   ./scripts/port_calibration_artifacts.sh
#
# Dependencies:
#   - CALIBRATION_RUNS env var (defaults to ~/.trustedrisk/calibration_runs)
#   - Run from the trustedrisk/ repo root
set -euo pipefail

# Resolve repo root (this script's parent's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CALIBRATION_RUNS="${CALIBRATION_RUNS:-$HOME/.trustedrisk/calibration_runs}"
DATA_DIR="$REPO_ROOT/data"
FIXTURES_DIR="$REPO_ROOT/fixtures"
ENV_FILE="$REPO_ROOT/.env"

mkdir -p "$DATA_DIR" "$FIXTURES_DIR"

echo "Calibration runs source: $CALIBRATION_RUNS"
echo "TrustedRisk target:       $REPO_ROOT"
echo ""

find_latest_run() {
    local workflow="$1"
    if [ ! -d "$CALIBRATION_RUNS" ]; then
        return 1
    fi
    # Run directories use a name prefix in the basename (e.g., run-<ts>-<hash>).
    # Filter by workflow name in run-metadata.json.
    local latest=""
    for d in "$CALIBRATION_RUNS"/run-*/; do
        [ -d "$d" ] || continue
        local meta="$d/run-metadata.json"
        [ -f "$meta" ] || continue
        if grep -q "\"workflow_name\": \"$workflow\"" "$meta" 2>/dev/null; then
            latest="$d"
        fi
    done
    [ -n "$latest" ] && echo "$latest"
}

port_coefficients() {
    local run
    run=$(find_latest_run "fhir-readmission-calibration") || true
    if [ -z "${run:-}" ] || [ ! -d "$run" ]; then
        echo "[W1] No fhir-readmission-calibration run found; skipping coefficients."
        return
    fi
    local src="$run/output/coefficients.json"
    if [ ! -f "$src" ]; then
        echo "[W1] run found but no coefficients.json at $src (critic may have halted)"
        return
    fi
    cp "$src" "$DATA_DIR/coefficients.json"
    echo "[W1] Copied: $src -> $DATA_DIR/coefficients.json"
    _merge_env_updates "$run/output/trustedrisk_env_updates.env"
}

port_fixtures() {
    local run
    run=$(find_latest_run "demo-cohort-casting") || true
    if [ -z "${run:-}" ] || [ ! -d "$run" ]; then
        echo "[W2] No demo-cohort-casting run found; skipping fixtures."
        return
    fi
    local out="$run/output"
    for f in patient_01_clean.json patient_02_abstain.json patient_03_complex.json \
             expected_signals.json narrative_contracts.md patient_02_edit_log.md; do
        if [ -f "$out/$f" ]; then
            cp "$out/$f" "$FIXTURES_DIR/$f"
            echo "[W2] Copied: $out/$f -> $FIXTURES_DIR/$f"
        fi
    done
}

port_grounding() {
    local run
    run=$(find_latest_run "grounding-corpus-build") || true
    if [ -z "${run:-}" ] || [ ! -d "$run" ]; then
        echo "[W3] No grounding-corpus-build run found; skipping grounding index."
        return
    fi
    local out="$run/output"
    # Copy the winner embedder's pickle (per embedder_comparison_report)
    # For simplicity, copy all three and let TRUSTEDRISK_GROUNDING_INDEX_PATH env select
    for f in grounding_index_minilm.pkl grounding_index_mpnet.pkl grounding_index_pubmedbert.pkl \
             chunks_metadata.json benchmark_queries.json embedder_comparison_report.md; do
        if [ -f "$out/$f" ]; then
            cp "$out/$f" "$DATA_DIR/$f"
            echo "[W3] Copied: $out/$f -> $DATA_DIR/$f"
        fi
    done
}

port_abstain_policy() {
    local run
    run=$(find_latest_run "abstain-boundary-discovery") || true
    if [ -z "${run:-}" ] || [ ! -d "$run" ]; then
        echo "[W4] No abstain-boundary-discovery run found; skipping abstain policy."
        return
    fi
    local out="$run/output"
    for f in abstain_policy_topological.json abstain_policy_lace_percentile.json \
             cohort_embeddings.npz cluster_cards.md policy_annotations.json; do
        if [ -f "$out/$f" ]; then
            cp "$out/$f" "$DATA_DIR/$f"
            echo "[W4] Copied: $out/$f -> $DATA_DIR/$f"
        fi
    done
    _merge_env_updates "$out/trustedrisk_env_updates.env"
}

_merge_env_updates() {
    local env_updates="$1"
    [ -f "$env_updates" ] || return 0
    [ -f "$ENV_FILE" ] && cp "$ENV_FILE" "$ENV_FILE.bak" || true
    # Drop lines from .env that are about to be superseded
    if [ -f "$ENV_FILE" ]; then
        local keys_to_replace
        keys_to_replace=$(grep -E '^[A-Z_]+' "$env_updates" | cut -d= -f1 | tr '\n' '|' | sed 's/|$//')
        if [ -n "$keys_to_replace" ]; then
            grep -vE "^($keys_to_replace)=" "$ENV_FILE" > "$ENV_FILE.tmp" || true
            mv "$ENV_FILE.tmp" "$ENV_FILE"
        fi
    fi
    cat "$env_updates" >> "$ENV_FILE"
    echo "    merged env updates: $env_updates"
}

# ─── Porting sequence ──────────────────────────────────────────────────
echo "=== [1/4] Porting W1 coefficients ==="
port_coefficients
echo ""
echo "=== [2/4] Porting W2 demo fixtures ==="
port_fixtures
echo ""
echo "=== [3/4] Porting W3 grounding indices ==="
port_grounding
echo ""
echo "=== [4/4] Porting W4 abstain policy ==="
port_abstain_policy
echo ""

# ─── Log ──────────────────────────────────────────────────────────────
ts=$(date -u +%FT%TZ)
echo "$ts ported by scripts/port_calibration_artifacts.sh" >> "$DATA_DIR/PORTING_LOG.txt"

# ─── Verify ───────────────────────────────────────────────────────────
echo "=== Running verify_artifacts.py ==="
if command -v py &>/dev/null; then
    py "$REPO_ROOT/scripts/verify_artifacts.py"
elif command -v python3 &>/dev/null; then
    python3 "$REPO_ROOT/scripts/verify_artifacts.py"
elif command -v python &>/dev/null; then
    python "$REPO_ROOT/scripts/verify_artifacts.py"
else
    echo "No python interpreter found; skipping verify step."
fi
