#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

GPU_LIST="${GPU_LIST:-0,1}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 2 ]]; then
    echo "GPU_LIST must contain exactly two GPU ids, for example 0,1" >&2
    exit 1
fi

RSSCN7_ROOT="${RSSCN7_ROOT:-/root/autodl-tmp/TransENet_base/datasets/RSSCN7-dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
RESUME="${RESUME:-0}"

LOG_DIR="${LOG_DIR:-logs/paper_mt_cga_rsscn7_${RUN_TS}}"
RESULTS_DIR="${RESULTS_DIR:-batch_results/paper_mt_cga_rsscn7_${RUN_TS}}"
SUMMARY_FILE="${SUMMARY_FILE:-${RESULTS_DIR}/summary.txt}"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

for required_dir in "${RSSCN7_ROOT}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Missing required directory: ${required_dir}" >&2
        exit 1
    fi
done

sanitize_tag() {
    local value="$1"
    value="${value//-/_}"
    value="${value//\//_}"
    value="${value// /_}"
    printf '%s\n' "${value}"
}

model_tag() {
    case "$1" in
        mt_paper) printf 'mt\n' ;;
        cga_paper) printf 'cga\n' ;;
        *)
            echo "Unsupported model: $1" >&2
            exit 1
            ;;
    esac
}

TASKS=(
    "mt_paper:2"
    "mt_paper:3"
    "mt_paper:4"
    "cga_paper:2"
    "cga_paper:3"
    "cga_paper:4"
)

run_task() {
    local gpu_id="$1"
    local model_name="$2"
    local scale="$3"
    local short_tag
    local save_name
    local result_file
    local log_file

    short_tag="$(model_tag "${model_name}")"
    save_name="paper_${short_tag}_RSSCN7_x${scale}_${RUN_TS}"
    result_file="${RESULTS_DIR}/${short_tag}_rsscn7_x${scale}.txt"
    log_file="${LOG_DIR}/${short_tag}_rsscn7_x${scale}.log"

    echo "[GPU ${gpu_id}] ${model_name} x${scale} -> ${log_file}"

    env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        DATASET="RSSCN7" \
        TRAIN_DATASET="RSSCN7" \
        EVAL_DATASET="RSSCN7" \
        DATASET_TAG="RSSCN7" \
        DATA_ROOT="${RSSCN7_ROOT}" \
        TRAIN_ROOT="${RSSCN7_ROOT}" \
        VAL_ROOT="${RSSCN7_ROOT}" \
        TEST_ROOT="${RSSCN7_ROOT}" \
        MODEL_NAME="${model_name}" \
        SCALE="${scale}" \
        SAVE_NAME="${save_name}" \
        RUN_TS="${RUN_TS}" \
        RESUME="${RESUME}" \
        RESULTS_ROOT="${RESULTS_ROOT}" \
        RESULTS_FILE="${result_file}" \
        ./train_paper_mt_cga_dataset.sh 2>&1 | tee "${log_file}"
}

worker() {
    local worker_idx="$1"
    local gpu_id="$2"
    local task
    local model_name
    local scale

    for ((i = worker_idx; i < ${#TASKS[@]}; i += ${#GPUS[@]})); do
        task="${TASKS[i]}"
        model_name="${task%%:*}"
        scale="${task##*:}"
        run_task "${gpu_id}" "${model_name}" "${scale}"
    done
}

pids=()
for idx in "${!GPUS[@]}"; do
    worker "${idx}" "${GPUS[idx]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

{
    echo "Batch summary for mt/cga on RSSCN7"
    echo "Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ) UTC"
    echo "GPU_LIST=${GPU_LIST}"
    echo "RUN_TS=${RUN_TS}"
    echo
} > "${SUMMARY_FILE}"

for task in "${TASKS[@]}"; do
    model_name="${task%%:*}"
    scale="${task##*:}"
    short_tag="$(model_tag "${model_name}")"
    result_file="${RESULTS_DIR}/${short_tag}_rsscn7_x${scale}.txt"

    if [[ -f "${result_file}" ]]; then
        cat "${result_file}" >> "${SUMMARY_FILE}"
    else
        echo "Missing result file: ${result_file}" >> "${SUMMARY_FILE}"
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "Batch finished with failures. Check logs under ${LOG_DIR}" >&2
    exit "${status}"
fi

echo "Batch finished successfully. Summary: ${SUMMARY_FILE}"
