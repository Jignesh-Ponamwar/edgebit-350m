#!/bin/bash
set -euo pipefail

# Training monitor with alerts.
# Watches training logs and GPU state, sends alerts on anomalies.
#
# Usage:
#   bash scripts/monitor.sh --output_dir /mnt/ckpts/edgebit-350m [--webhook URL]

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ckpts/edgebit-350m}"
WEBHOOK_URL="${WEBHOOK_URL:-}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
LOSS_SPIKE_THRESHOLD="${LOSS_SPIKE_THRESHOLD:-5.0}"
GPU_TEMP_THRESHOLD="${GPU_TEMP_THRESHOLD:-85}"
GPU_UTIL_MIN="${GPU_UTIL_MIN:-50}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --webhook) WEBHOOK_URL="$2"; shift 2 ;;
        --interval) CHECK_INTERVAL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

send_alert() {
    local level="$1"
    local message="$2"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[${timestamp}] [${level}] ${message}"

    if [ -n "${WEBHOOK_URL}" ]; then
        curl -s -X POST "${WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"text\": \"[EdgeBit ${level}] ${message}\"}" \
            >/dev/null 2>&1 || true
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &>/dev/null; then
        return
    fi

    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null) || return

    while IFS=, read -r temp util mem_used mem_total; do
        temp=$(echo "${temp}" | tr -d ' ')
        util=$(echo "${util}" | tr -d ' ')

        if [ "${temp}" -gt "${GPU_TEMP_THRESHOLD}" ] 2>/dev/null; then
            send_alert "WARN" "GPU temp ${temp}C exceeds ${GPU_TEMP_THRESHOLD}C threshold"
        fi

        if [ "${util}" -lt "${GPU_UTIL_MIN}" ] 2>/dev/null; then
            send_alert "WARN" "GPU util ${util}% below ${GPU_UTIL_MIN}% — possible data loading bottleneck"
        fi
    done <<< "${gpu_info}"
}

check_training() {
    local latest_ckpt
    latest_ckpt=$(find "${OUTPUT_DIR}" -name "training_state.pt" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')

    if [ -z "${latest_ckpt}" ]; then
        return
    fi

    local ckpt_age
    ckpt_age=$(( $(date +%s) - $(stat -c %Y "${latest_ckpt}" 2>/dev/null || echo "0") ))

    if [ "${ckpt_age}" -gt 7200 ]; then
        send_alert "WARN" "Last checkpoint is $(( ckpt_age / 3600 ))h old — training may have stalled"
    fi
}

check_disk() {
    local usage
    usage=$(df "${OUTPUT_DIR}" --output=pcent 2>/dev/null | tail -1 | tr -d '% ')
    if [ -n "${usage}" ] && [ "${usage}" -gt 90 ] 2>/dev/null; then
        send_alert "CRITICAL" "Disk usage at ${usage}% — checkpoint writes may fail"
    fi
}

echo "=== EdgeBit Training Monitor ==="
echo "Watching: ${OUTPUT_DIR}"
echo "Interval: ${CHECK_INTERVAL}s"
echo "Alerts:   ${WEBHOOK_URL:-console only}"
echo ""

send_alert "INFO" "Monitor started for ${OUTPUT_DIR}"

while true; do
    check_gpu
    check_training
    check_disk
    sleep "${CHECK_INTERVAL}"
done
