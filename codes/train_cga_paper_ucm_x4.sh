#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME0="cga_paper"
MODEL_NAME1="ham_paper"
MODEL_NAME2="mt_paper"
EPOCHS="${EPOCHS:-800}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
SCHEDULER="${SCHEDULER:-cosine}"
DECAY_TYPE0="${DECAY_TYPE:-step_250000_400000_450000_475000}"
DECAY_TYPE1="${DECAY_TYPE:-step_250000}"
GAMMA="${GAMMA:-0.5}"
MAX_STEPS="${MAX_STEPS:-500000}"
PATCH_SIZE="${PATCH_SIZE:-192}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-1}"
EXT_MODE="${EXT_MODE:-sep}"
SCALE=4
DATASET="UCMerced"

UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
TEST_LR_DIR="${TEST_LR_DIR:-${UCMERCED_ROOT}/test/LR_x${SCALE}}"
TEST_HR_DIR="${TEST_HR_DIR:-${UCMERCED_ROOT}/test/HR_x${SCALE}}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
SAVE_NAME0="paper_cga_ucm_x4_20260507_195957"
SAVE_NAME1="paper_ham_ucm_x4_${RUN_TS}"
SAVE_NAME2="paper_mt_ucm_x4_${RUN_TS}"

train_and_eval() {
    local model_name="$1"
    local save_name="$2"
    local decay_type="$3"
    local use_amp="$4"
    local amp_args=()

    if [[ "${use_amp}" == "1" ]]; then
        amp_args=(--amp)
    fi

    # python train_enhanced.py \
    #     --model "${model_name}" \
    #     --dataset "${DATASET}" \
    #     --scale "${SCALE}" \
    #     --epochs "${EPOCHS}" \
    #     --max_steps "${MAX_STEPS}" \
    #     --scheduler_unit epoch \
    #     --batch_size "${BATCH_SIZE}" \
    #     --n_threads "${N_THREADS}" \
    #     "${amp_args[@]}" \
    #     --ext "${EXT_MODE}" \
    #     --patch_size "${PATCH_SIZE}" \
    #     --resume 0 \
    #     --optimizer "${OPTIMIZER}" \
    #     --scheduler "${SCHEDULER}" \
    #     --decay_type "${decay_type}" \
    #     --gamma "${GAMMA}" \
    #     --lr "${LR}" \
    #     --beta1 0.9 \
    #     --beta2 0.99 \
    #     --loss "1*L1" \
    #     --data_train "${UCMERCED_ROOT}/train" \
    #     --data_val "${UCMERCED_ROOT}/val" \
    #     --val_every "${VAL_EVERY}" \
    #     --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
    #     --save "${save_name}"

    local model_path="../experiment/${save_name}/model/model_best.pt"
    local out_dir="${RESULTS_ROOT}/${save_name}/x${SCALE}"

    python demo_deploy.py \
        --model "${model_name}" \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --pre_train "${model_path}" \
        --dir_data "${TEST_LR_DIR}" \
        --dir_out "${out_dir}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_PSNR_SSIM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${TEST_HR_DIR}" \
        --folder_Gen "${out_dir}" | tail -n 1 >> "${RESULTS_FILE}"
}

train_and_eval "${MODEL_NAME0}" "${SAVE_NAME0}" "${DECAY_TYPE0}" 1
# train_and_eval "${MODEL_NAME2}" "${SAVE_NAME2}" "${DECAY_TYPE1}" 0
#train_and_eval "${MODEL_NAME1}" "${SAVE_NAME1}" "${DECAY_TYPE1}" 1
