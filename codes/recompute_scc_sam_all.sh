#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

CONDA_ENV="${CONDA_ENV:-transenet-pren}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

SUMMARY="${SUMMARY:-codes/results_scc_sam_recomputed_20260717.txt}"
LOG_DIR="${LOG_DIR:-codes/scc_sam_recomputed_logs_20260717}"
N_THREADS="${N_THREADS:-$(nproc)}"
mkdir -p "${LOG_DIR}"

: > "${SUMMARY}"
{
    echo "Recomputed SCC/SAM with sewar-compatible implementation"
    echo "Generated at: $(date -u +%Y-%m-%dT%H:%M:%SZ) UTC"
    echo "Workers per metric run: ${N_THREADS}"
    echo
} | tee -a "${SUMMARY}"

run_metric() {
    local id="$1"
    local dataset="$2"
    local scale="$3"
    local gt="$4"
    local gen="$5"
    local log="${LOG_DIR}/${id}.log"

    for required_path in "${gt}" "${gen}"; do
        if [[ ! -d "${required_path}" ]]; then
            echo "Missing required directory for ${id}: ${required_path}" | tee -a "${SUMMARY}"
            exit 1
        fi
    done

    {
        echo "[${id}] dataset=${dataset} scale=x${scale}"
        echo "GT: ${gt}"
        echo "GEN: ${gen}"
    } | tee -a "${SUMMARY}"

    "${PYTHON_BIN}" codes/calculate_SCC_SAM.py \
        --dataset "${dataset}" \
        --scale "${scale}" \
        --n_threads "${N_THREADS}" \
        --folder_GT "${gt}" \
        --folder_Gen "${gen}" > "${log}"

    tail -n 1 "${log}" | tee -a "${SUMMARY}"
    {
        echo "Log: ${log}"
        echo
    } >> "${SUMMARY}"
}

run_metric aid_x2 AID 2 \
    datasets/AID-dataset/test/HR \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x2_w32_20260712_112935/x2
run_metric aid_x3 AID 3 \
    datasets/AID-dataset/test/HR \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x3_w32_20260712_112758/x3
run_metric aid_x4 AID 4 \
    datasets/AID-dataset/test/HR \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x4_w32_20260712_112858/x4

run_metric rsscn7_x2 RSSCN7 2 \
    datasets/RSSCN7-dataset/test/HR \
    experiment/results/COMSNET_RSSCN7_x2_w32_20260717_035255/x2
run_metric rsscn7_x3 RSSCN7 3 \
    datasets/RSSCN7-dataset/test/HR \
    experiment/results/COMSNET_RSSCN7_x3_w32_20260717_035718/x3
run_metric rsscn7_x4 RSSCN7 4 \
    datasets/RSSCN7-dataset/test/HR \
    experiment/results/COMSNET_RSSCN7_x4_w32_20260717_035733/x4

run_metric whu_rs19_x2 WHU-RS19 2 \
    datasets/WHU-RS19-test/test/HR \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x2_w32_20260712_112935_WHU_RS19/x2
run_metric whu_rs19_x3 WHU-RS19 3 \
    datasets/WHU-RS19-test/test/HR \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x3_w32_20260712_112758_WHU_RS19/x3
run_metric whu_rs19_x4 WHU-RS19 4 \
    datasets/WHU-RS19-test/test/HR \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x4_w32_20260712_112858_WHU_RS19/x4

run_metric ucmerced_x2 UCMerced 2 \
    datasets/UCMerced-dataset/test/HR_x2 \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260429_071942/x2
run_metric ucmerced_x3 UCMerced 3 \
    datasets/UCMerced-dataset/test/HR_x3 \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x3_w32_20260429_071942/x3
run_metric ucmerced_x4 UCMerced 4 \
    datasets/UCMerced-dataset/test/HR_x4 \
    experiment/results/s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x4_w32_20260520_190021/x4

echo "Done. Summary: ${SUMMARY}" | tee -a "${SUMMARY}"
