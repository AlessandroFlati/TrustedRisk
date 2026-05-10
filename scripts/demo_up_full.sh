#!/usr/bin/env bash
# Phase 10.10 — bring up the full 14-specialist federation locally.
#
# Background-runs uvicorn for each apps/specialist_*/server.py with its
# canonical port. PIDs are written to /tmp/trustedrisk-pids/<id>.pid so
# `scripts/demo_down_full.sh` can stop them.
#
# Usage:
#   bash scripts/demo_up_full.sh [optional-list-of-ids]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_DIR="${TRUSTEDRISK_PID_DIR:-/tmp/trustedrisk-pids}"
LOG_DIR="${TRUSTEDRISK_LOG_DIR:-/tmp/trustedrisk-logs}"
mkdir -p "${PID_DIR}" "${LOG_DIR}"

PYTHON="${TRUSTEDRISK_PYTHON:-python}"
HOST="${TRUSTEDRISK_HOST:-0.0.0.0}"

declare -A SPECIALISTS=(
    [discharge]=8770
    [acute]=8771
    [evidence]=8772
    [population]=8773
    [pediatric]=8774
    [pa]=8775
    [scribe]=8776
    [patient]=8777
    [coder]=8778
    [pgx]=8779
    [composer]=8780
    [preadmit]=8781
    [quality]=8782
    [pophealth]=8783
    [appeals]=8784
    [multimodal]=8786
    [mental-health]=8785
)

declare -A APP_MODULE=(
    [discharge]=apps.specialist_discharge.server:app
    [acute]=apps.specialist_acute.server:app
    [evidence]=apps.specialist_evidence.server:app
    [population]=apps.specialist_population.server:app
    [pediatric]=apps.specialist_pediatric.server:app
    [pa]=apps.specialist_pa.server:app
    [scribe]=apps.specialist_scribe.server:app
    [patient]=apps.specialist_patient.server:app
    [coder]=apps.specialist_coder.server:app
    [pgx]=apps.specialist_pgx.server:app
    [composer]=apps.composer.server:app
    [preadmit]=apps.specialist_preadmit.server:app
    [quality]=apps.specialist_quality.server:app
    [pophealth]=apps.specialist_pophealth.server:app
    [appeals]=apps.specialist_appeals.server:app
    [multimodal]=apps.specialist_multimodal.server:app
    [mental-health]=apps.specialist_mental_health.server:app
)

# Optional positional filter: bash demo_up_full.sh quality pophealth appeals
TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("${!SPECIALISTS[@]}")
fi

echo "Starting ${#TARGETS[@]} federation member(s) under ${PID_DIR}"
echo "(logs under ${LOG_DIR})"
for id in "${TARGETS[@]}"; do
    port="${SPECIALISTS[$id]:-}"
    module="${APP_MODULE[$id]:-}"
    if [ -z "${port}" ] || [ -z "${module}" ]; then
        echo "  skip ${id} — unknown id"
        continue
    fi
    pidfile="${PID_DIR}/${id}.pid"
    logfile="${LOG_DIR}/${id}.log"
    if [ -f "${pidfile}" ] && kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
        echo "  ${id} already running (pid $(cat "${pidfile}"))"
        continue
    fi
    (
        cd "${ROOT}"
        PYTHONPATH=src nohup "${PYTHON}" -m uvicorn "${module}" \
            --host "${HOST}" --port "${port}" --log-level info \
            > "${logfile}" 2>&1 &
        echo $! > "${pidfile}"
    )
    echo "  ${id} on :${port} (pid $(cat "${pidfile}"))"
done

echo ""
echo "Use 'bash scripts/demo_status.sh' to inspect, "
echo "or 'make demo-down' to stop the docker-compose stack "
echo "(individual processes can be stopped with kill \$(cat ${PID_DIR}/<id>.pid))."
