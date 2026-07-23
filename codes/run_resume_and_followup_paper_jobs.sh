#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

GPU_RESUME_CGA="${GPU_RESUME_CGA:-0}"
GPU_RESUME_MT="${GPU_RESUME_MT:-1}"
GPU_FOLLOWUP_CGA="${GPU_FOLLOWUP_CGA:-${GPU_RESUME_CGA}}"
GPU_AID_TO_WHU="${GPU_AID_TO_WHU:-${GPU_RESUME_MT}}"

RSSCN7_ROOT="${RSSCN7_ROOT:-/root/autodl-tmp/TransENet_base/datasets/RSSCN7-dataset}"
AID_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
WHU_RS19_TEST_ROOT="${WHU_RS19_TEST_ROOT:-/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-test}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"

FOLLOWUP_RUN_TS="${FOLLOWUP_RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-paper_resume_followup_${FOLLOWUP_RUN_TS}}"
LOG_DIR="${LOG_DIR:-logs/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-batch_results/${RUN_TAG}}"
SUMMARY_FILE="${SUMMARY_FILE:-${RESULTS_DIR}/summary.txt}"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

for required_dir in "${RSSCN7_ROOT}" "${AID_ROOT}" "${WHU_RS19_TEST_ROOT}"; do
    if [[ ! -d "${required_dir}" ]]; then
        echo "Missing required directory: ${required_dir}" >&2
        exit 1
    fi
done

run_train_job() {
    local gpu_id="$1"
    local model_name="$2"
    local scale="$3"
    local save_name="$4"
    local resume_flag="$5"
    local result_file="$6"
    local log_file="$7"

    echo "[train] gpu=${gpu_id} model=${model_name} scale=x${scale} save=${save_name}"

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
        EPOCHS=500 \
        RESUME="${resume_flag}" \
        SAVE_NAME="${save_name}" \
        RESULTS_ROOT="${RESULTS_ROOT}" \
        RESULTS_FILE="${result_file}" \
        ./train_paper_mt_cga_dataset.sh 2>&1 | tee "${log_file}"
}

run_test_job() {
    local gpu_id="$1"
    local scale="$2"
    local result_file="$3"
    local log_file="$4"

    echo "[test] gpu=${gpu_id} model=cga_paper scale=x${scale} AID->WHU-RS19"

    env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        MODEL_NAME="cga_paper" \
        SCALE="${scale}" \
        AID_ROOT="${AID_ROOT}" \
        WHU_RS19_TEST_ROOT="${WHU_RS19_TEST_ROOT}" \
        RESULTS_ROOT="${RESULTS_ROOT}" \
        RESULTS_FILE="${result_file}" \
        ./test_paper_mt_cga_aid_to_whu_rs19.sh 2>&1 | tee "${log_file}"
}

status=0

# Stage 1: resume the interrupted jobs from July 18, 2026.
run_train_job \
    "${GPU_RESUME_CGA}" \
    "cga_paper" \
    "2" \
    "paper_cga_RSSCN7_x2_20260718_162858" \
    "1" \
    "${RESULTS_DIR}/cga_rsscn7_x2_resume.txt" \
    "${LOG_DIR}/cga_rsscn7_x2_resume.log" &
pid_resume_cga=$!

run_train_job \
    "${GPU_RESUME_MT}" \
    "mt_paper" \
    "4" \
    "paper_mt_RSSCN7_x4_20260718_162858" \
    "1" \
    "${RESULTS_DIR}/mt_rsscn7_x4_resume.txt" \
    "${LOG_DIR}/mt_rsscn7_x4_resume.log" &
pid_resume_mt=$!

if ! wait "${pid_resume_cga}"; then
    status=1
fi
if ! wait "${pid_resume_mt}"; then
    status=1
fi

# Stage 2: continue with the remaining cga RSSCN7 training jobs.
if ! run_train_job \
    "${GPU_FOLLOWUP_CGA}" \
    "cga_paper" \
    "3" \
    "paper_cga_RSSCN7_x3_${FOLLOWUP_RUN_TS}" \
    "0" \
    "${RESULTS_DIR}/cga_rsscn7_x3.txt" \
    "${LOG_DIR}/cga_rsscn7_x3.log"; then
    status=1
fi

if ! run_train_job \
    "${GPU_FOLLOWUP_CGA}" \
    "cga_paper" \
    "4" \
    "paper_cga_RSSCN7_x4_${FOLLOWUP_RUN_TS}" \
    "0" \
    "${RESULTS_DIR}/cga_rsscn7_x4.txt" \
    "${LOG_DIR}/cga_rsscn7_x4.log"; then
    status=1
fi

# Stage 3: run cga AID -> WHU-RS19 test-only for x2/x3/x4.
if ! run_test_job \
    "${GPU_AID_TO_WHU}" \
    "2" \
    "${RESULTS_DIR}/cga_aid_to_whu_rs19_x2.txt" \
    "${LOG_DIR}/cga_aid_to_whu_rs19_x2.log"; then
    status=1
fi

if ! run_test_job \
    "${GPU_AID_TO_WHU}" \
    "3" \
    "${RESULTS_DIR}/cga_aid_to_whu_rs19_x3.txt" \
    "${LOG_DIR}/cga_aid_to_whu_rs19_x3.log"; then
    status=1
fi

if ! run_test_job \
    "${GPU_AID_TO_WHU}" \
    "4" \
    "${RESULTS_DIR}/cga_aid_to_whu_rs19_x4.txt" \
    "${LOG_DIR}/cga_aid_to_whu_rs19_x4.log"; then
    status=1
fi

{
    echo "Summary generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ) UTC"
    echo "RUN_TAG=${RUN_TAG}"
    echo "FOLLOWUP_RUN_TS=${FOLLOWUP_RUN_TS}"
    echo
    for result_file in \
        "${RESULTS_DIR}/cga_rsscn7_x2_resume.txt" \
        "${RESULTS_DIR}/mt_rsscn7_x4_resume.txt" \
        "${RESULTS_DIR}/cga_rsscn7_x3.txt" \
        "${RESULTS_DIR}/cga_rsscn7_x4.txt" \
        "${RESULTS_DIR}/cga_aid_to_whu_rs19_x2.txt" \
        "${RESULTS_DIR}/cga_aid_to_whu_rs19_x3.txt" \
        "${RESULTS_DIR}/cga_aid_to_whu_rs19_x4.txt"; do
        if [[ -f "${result_file}" ]]; then
            cat "${result_file}"
        else
            echo "Missing result file: ${result_file}"
        fi
        echo
    done
} > "${SUMMARY_FILE}"

if [[ "${status}" -ne 0 ]]; then
    echo "One or more jobs failed. Check ${LOG_DIR} and ${SUMMARY_FILE}" >&2
    exit "${status}"
fi

echo "All jobs completed. Summary: ${SUMMARY_FILE}"
