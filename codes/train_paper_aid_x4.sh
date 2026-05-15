#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

DATASET="AID"
SCALE=4
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
SCHEDULER="${SCHEDULER:-step}"
DECAY_TYPE="${DECAY_TYPE:-step}"
GAMMA="${GAMMA:-0.5}"
MAX_STEPS="${MAX_STEPS:-500000}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
EXT_MODE="${EXT_MODE:-sep}"

AID_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

# image_size keeps the dataset-level reference semantics in the original framework.
IMAGE_SIZE="${IMAGE_SIZE:-600}"
PATCH_SIZE="${PATCH_SIZE:-256}"
CGA_SPLIT_SIZE="${CGA_SPLIT_SIZE:-8,32}"

train_and_eval() {
    local model_name="$1"
    local save_name="$2"
    local use_amp="$3"
    local amp_args=()

    if [[ "${use_amp}" == "1" ]]; then
        amp_args=(--amp)
    fi

    python train_enhanced.py \
        --model "${model_name}" \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --epochs "${EPOCHS}" \
        --max_steps "${MAX_STEPS}" \
        --scheduler_unit epoch \
        --batch_size "${BATCH_SIZE}" \
        --n_threads "${N_THREADS}" \
        "${amp_args[@]}" \
        --ext "${EXT_MODE}" \
        --patch_size "${PATCH_SIZE}" \
        --resume 1 \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --decay_type "${DECAY_TYPE}" \
        --gamma "${GAMMA}" \
        --lr "${LR}" \
        --beta1 0.9 \
        --beta2 0.99 \
        --loss "1*L1" \
        --cga_paper_split_size "${CGA_SPLIT_SIZE}" \
        --data_train "${AID_ROOT}/train" \
        --data_val "${AID_ROOT}/val" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --save "${save_name}"

    local model_path="../experiment/${save_name}/model/model_best.pt"
    local out_dir="${RESULTS_ROOT}/${save_name}/x${SCALE}"
    local test_lr_dir="${AID_ROOT}/test/LR_x${SCALE}"
    local test_hr_dir="${AID_ROOT}/test/HR"

    python demo_deploy.py \
        --model "${model_name}" \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --cga_paper_split_size "${CGA_SPLIT_SIZE}" \
        --pre_train "${model_path}" \
        --dir_data "${test_lr_dir}" \
        --dir_out "${out_dir}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_PSNR_SSIM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${test_hr_dir}" \
        --folder_Gen "${out_dir}" | tail -n 1 >> "${RESULTS_FILE}"
}

SAVE_NAME_CGA="paper_cga_aid_x4_20260512_155448"
SAVE_NAME_MT="paper_mt_aid_x4_${RUN_TS}"
SAVE_NAME_HAM="paper_ham_aid_x4_${RUN_TS}"

train_and_eval "cga_paper" "${SAVE_NAME_CGA}" 0
# train_and_eval "mt_paper" "${SAVE_NAME_MT}" 0
# train_and_eval "ham_paper" "${SAVE_NAME_HAM}" 0
