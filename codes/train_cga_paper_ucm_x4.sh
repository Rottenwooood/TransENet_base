#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="cga_paper"
EPOCHS="${EPOCHS:-999999}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
SCHEDULER="${SCHEDULER:-step}"
DECAY_TYPE="${DECAY_TYPE:-step_250000_400000_450000_475000}"
GAMMA="${GAMMA:-0.5}"
MAX_STEPS="${MAX_STEPS:-500000}"
PATCH_SIZE="${PATCH_SIZE:-192}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-2}"
EXT_MODE="${EXT_MODE:-sep}"

UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
SAVE_NAME="paper_cga_ucm_x4_${RUN_TS}"

python train_enhanced.py \
    --model "${MODEL_NAME}" \
    --dataset UCMerced \
    --scale 4 \
    --epochs "${EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --scheduler_unit step \
    --batch_size "${BATCH_SIZE}" \
    --n_threads "${N_THREADS}" \
    --amp \
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
    --data_train "${UCMERCED_ROOT}/train" \
    --data_val "${UCMERCED_ROOT}/val" \
    --val_every "${VAL_EVERY}" \
    --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
    --save "${SAVE_NAME}"
