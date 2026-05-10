#!/usr/bin/env bash
# Phase 10.10 — print health status of each running federation member.

set -euo pipefail

PID_DIR="${TRUSTEDRISK_PID_DIR:-/tmp/trustedrisk-pids}"
HOST="${TRUSTEDRISK_HOST:-127.0.0.1}"

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
)

printf "%-12s %-6s %-12s %-10s %s\n" "id" "port" "pid" "status" "tools/bundles"
printf -- "------------ ------ ------------ ---------- -----\n"
for id in "${!SPECIALISTS[@]}"; do
    port="${SPECIALISTS[$id]}"
    pidfile="${PID_DIR}/${id}.pid"
    pid=""
    status="stopped"
    if [ -f "${pidfile}" ]; then
        pid="$(cat "${pidfile}")"
        if kill -0 "${pid}" 2>/dev/null; then
            status="running"
        else
            status="dead"
        fi
    fi

    info="-"
    if [ "${status}" = "running" ]; then
        body="$(curl -s --max-time 2 "http://${HOST}:${port}/healthz" || true)"
        if [ -n "${body}" ]; then
            tools="$(echo "${body}" | grep -o '"tools_registered":[0-9]*' | cut -d: -f2 || true)"
            bundles="$(echo "${body}" | grep -o '"bundles_registered":[0-9]*' | cut -d: -f2 || true)"
            info="${tools:-?}/${bundles:-?}"
        else
            info="no /healthz"
        fi
    fi
    printf "%-12s %-6s %-12s %-10s %s\n" "${id}" "${port}" "${pid:--}" "${status}" "${info}"
done
