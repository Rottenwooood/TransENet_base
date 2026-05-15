#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

DATASET="AID"
MODEL_NAME="cga_paper"
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

AID_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

PATCH_SIZE="${PATCH_SIZE:-192}"
CGA_SPLIT_SIZE="${CGA_SPLIT_SIZE:-8,32}"
SAVE_NAME="paper_cga_aid_x3_20260514_032416"

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
    --save "${SAVE_NAME}"

MODEL_PATH="../experiment/${SAVE_NAME}/model/model_best.pt"
OUT_DIR="${RESULTS_ROOT}/${SAVE_NAME}/x${SCALE}"
TEST_LR_DIR="${AID_ROOT}/test/LR_x${SCALE}"
TEST_HR_DIR="${AID_ROOT}/test/HR"

python demo_deploy.py \
    --model "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --scale "${SCALE}" \
    --cga_paper_split_size "${CGA_SPLIT_SIZE}" \
    --pre_train "${MODEL_PATH}" \
    --dir_data "${TEST_LR_DIR}" \
    --dir_out "${OUT_DIR}" | tail -n 1 >> "${RESULTS_FILE}"

python calculate_PSNR_SSIM.py \
    --dataset "${DATASET}" \
    --scale "${SCALE}" \
    --folder_GT "${TEST_HR_DIR}" \
    --folder_Gen "${OUT_DIR}" | tail -n 1 >> "${RESULTS_FILE}"
