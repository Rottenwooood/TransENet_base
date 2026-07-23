#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

DATASET="RSSCN7"
MODEL_NAME="ham_paper"
SCALE=3
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

RSSCN7_ROOT="${RSSCN7_ROOT:-/root/autodl-tmp/TransENet_base/datasets/RSSCN7-dataset}"
WHU_RS19_TEST_ROOT="${WHU_RS19_TEST_ROOT:-/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-test}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results_paper_ham_rsscn7_x3.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

PATCH_SIZE="${PATCH_SIZE:-192}"
SAVE_NAME="${SAVE_NAME:-paper_ham_rsscn7_x3_${RUN_TS}}"

train_and_eval_rsscn7() {
    python train_enhanced.py \
        --model "${MODEL_NAME}" \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --epochs "${EPOCHS}" \
        --max_steps "${MAX_STEPS}" \
        --scheduler_unit epoch \
        --batch_size "${BATCH_SIZE}" \
        --n_threads "${N_THREADS}" \
        --ext "${EXT_MODE}" \
        --patch_size "${PATCH_SIZE}" \
        --resume 0 \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --decay_type "${DECAY_TYPE}" \
        --gamma "${GAMMA}" \
        --lr "${LR}" \
        --beta1 0.9 \
        --beta2 0.99 \
        --loss "1*L1" \
        --data_train "${RSSCN7_ROOT}/train" \
        --data_val "${RSSCN7_ROOT}/val" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --save "${SAVE_NAME}"

    MODEL_PATH="../experiment/${SAVE_NAME}/model/model_best.pt"
    OUT_DIR="${RESULTS_ROOT}/${SAVE_NAME}/x${SCALE}"

    python demo_deploy.py \
        --model "${MODEL_NAME}" \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --pre_train "${MODEL_PATH}" \
        --dir_data "${RSSCN7_ROOT}/test/LR_x${SCALE}" \
        --dir_out "${OUT_DIR}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_PSNR_SSIM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${RSSCN7_ROOT}/test/HR" \
        --folder_Gen "${OUT_DIR}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_SCC_SAM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${RSSCN7_ROOT}/test/HR" \
        --folder_Gen "${OUT_DIR}" | tail -n 1 >> "${RESULTS_FILE}"
}

test_whu_rs19_with_aid_weight() {
    local aid_model_path="${AID_MODEL_PATH:-}"
    if [[ -z "${aid_model_path}" ]]; then
        if [[ -n "${AID_SAVE_NAME:-}" ]]; then
            aid_model_path="../experiment/${AID_SAVE_NAME}/model/model_best.pt"
        else
            echo "Set AID_MODEL_PATH or AID_SAVE_NAME for WHU-RS19 evaluation." >&2
            exit 1
        fi
    fi

    local whu_out_tag="${WHU_OUT_TAG:-${AID_SAVE_NAME:-$(basename "${aid_model_path%/model/model_best.pt}")}_WHU_RS19}"
    local whu_out_dir="${RESULTS_ROOT}/${whu_out_tag}/x${SCALE}"

    python demo_deploy.py \
        --model "${MODEL_NAME}" \
        --dataset "WHU-RS19" \
        --scale "${SCALE}" \
        --pre_train "${aid_model_path}" \
        --dir_data "${WHU_RS19_TEST_ROOT}/test/LR_x${SCALE}" \
        --dir_out "${whu_out_dir}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_PSNR_SSIM.py \
        --dataset "WHU-RS19" \
        --scale "${SCALE}" \
        --folder_GT "${WHU_RS19_TEST_ROOT}/test/HR" \
        --folder_Gen "${whu_out_dir}" | tail -n 1 >> "${RESULTS_FILE}"

    python calculate_SCC_SAM.py \
        --dataset "WHU-RS19" \
        --scale "${SCALE}" \
        --folder_GT "${WHU_RS19_TEST_ROOT}/test/HR" \
        --folder_Gen "${whu_out_dir}" | tail -n 1 >> "${RESULTS_FILE}"
}

train_and_eval_rsscn7
test_whu_rs19_with_aid_weight
