#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME1="ham_paper"
MODEL_NAME2="mt_paper"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
SCHEDULER="${SCHEDULER:-step}"
DECAY_TYPE="${DECAY_TYPE:-step}"
GAMMA="${GAMMA:-0.5}"
MAX_STEPS="${MAX_STEPS:-500000}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-1}"
EXT_MODE="${EXT_MODE:-sep}"
DATASET="UCMerced"

UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

PATCH_SIZE_X3="${PATCH_SIZE_X3:-144}"
PATCH_SIZE_X2="${PATCH_SIZE_X2:-96}"

train_and_eval() {
    local model_name="$1"
    local save_name="$2"
    local scale="$3"
    local patch_size="$4"
    local use_amp="$5"
    local amp_args=()

    if [[ "${use_amp}" == "1" ]]; then
        amp_args=(--amp)
    fi

    python train_enhanced.py \
        --model "${model_name}" \
        --dataset "${DATASET}" \
        --scale "${scale}" \
        --epochs "${EPOCHS}" \
        --max_steps "${MAX_STEPS}" \
        --scheduler_unit epoch \
        --batch_size "${BATCH_SIZE}" \
        --n_threads "${N_THREADS}" \
        "${amp_args[@]}" \
        --ext "${EXT_MODE}" \
        --patch_size "${patch_size}" \
        --resume 0 \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --decay_type "${DECAY_TYPE}" \
        --gamma "${GAMMA}" \
        --lr "${LR}" \
        --beta1 0.9 \
        --beta2 0.99 \
        --loss "1*L1" \
        --data_train "${UCMERCED_ROOT}/train" \
        --data_val "${UCMERCED_ROOT}/val" \
        --val_every "${VAL_EVERY}" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --save "${save_name}"

    local model_path="../experiment/${save_name}/model/model_best.pt"
    local out_dir="${RESULTS_ROOT}/${save_name}/x${scale}"
    local test_lr_dir="${UCMERCED_ROOT}/test/LR_x${scale}"
    local test_hr_dir="${UCMERCED_ROOT}/test/HR_x${scale}"

    python demo_deploy.py \
        --model "${model_name}" \
        --dataset "${DATASET}" \
        --scale "${scale}" \
        --pre_train "${model_path}" \
        --dir_data "${test_lr_dir}" \
        --dir_out "${out_dir}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_PSNR_SSIM.py \
        --dataset "${DATASET}" \
        --scale "${scale}" \
        --folder_GT "${test_hr_dir}" \
        --folder_Gen "${out_dir}" | tail -n 1 >> "${RESULTS_FILE}"
}

SAVE_NAME_MT_X3="paper_mt_ucm_x3_${RUN_TS}"
SAVE_NAME_HAM_X3="paper_ham_ucm_x3_${RUN_TS}"
SAVE_NAME_MT_X2="paper_mt_ucm_x2_${RUN_TS}"
SAVE_NAME_HAM_X2="paper_ham_ucm_x2_${RUN_TS}"

train_and_eval "${MODEL_NAME2}" "${SAVE_NAME_MT_X3}" 3 "${PATCH_SIZE_X3}" 0
train_and_eval "${MODEL_NAME1}" "${SAVE_NAME_HAM_X3}" 3 "${PATCH_SIZE_X3}" 0
train_and_eval "${MODEL_NAME2}" "${SAVE_NAME_MT_X2}" 2 "${PATCH_SIZE_X2}" 0
train_and_eval "${MODEL_NAME1}" "${SAVE_NAME_HAM_X2}" 2 "${PATCH_SIZE_X2}" 0
