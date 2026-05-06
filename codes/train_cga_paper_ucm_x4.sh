#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME0="cga_paper"
MODEL_NAME1="ham_paper"
MODEL_NAME2="mt_paper"
EPOCHS="${EPOCHS:-999999}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
SCHEDULER="${SCHEDULER:-step}"
DECAY_TYPE0="${DECAY_TYPE:-step_250000_400000_450000_475000}"
DECAY_TYPE1="${DECAY_TYPE:-step_250000}"
GAMMA="${GAMMA:-0.5}"
MAX_STEPS="${MAX_STEPS:-500000}"
PATCH_SIZE="${PATCH_SIZE:-192}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-1}"
EXT_MODE="${EXT_MODE:-sep}"

UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
SAVE_NAME0="paper_cga_ucm_x4_${RUN_TS}"
SAVE_NAME1="paper_ham_ucm_x4_${RUN_TS}"
SAVE_NAME2="paper_mt_ucm_x4_${RUN_TS}"

# python train_enhanced.py \
#     --model "${MODEL_NAME0}" \
#     --dataset UCMerced \
#     --scale 4 \
#     --epochs "${EPOCHS}" \
#     --max_steps "${MAX_STEPS}" \
#     --scheduler_unit step \
#     --batch_size "${BATCH_SIZE}" \
#     --n_threads "${N_THREADS}" \
#     --amp \
#     --ext "${EXT_MODE}" \
#     --patch_size "${PATCH_SIZE}" \
#     --resume 0 \
#     --optimizer "${OPTIMIZER}" \
#     --scheduler "${SCHEDULER}" \
#     --decay_type "${DECAY_TYPE0}" \
#     --gamma "${GAMMA}" \
#     --lr "${LR}" \
#     --beta1 0.9 \
#     --beta2 0.99 \
#     --loss "1*L1" \
#     --data_train "${UCMERCED_ROOT}/train" \
#     --data_val "${UCMERCED_ROOT}/val" \
#     --val_every "${VAL_EVERY}" \
#     --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
#     --save "${SAVE_NAME0}"

python train_enhanced.py \
    --model "${MODEL_NAME1}" \
    --dataset UCMerced \
    --scale 4 \
    --epochs "${EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --scheduler_unit step \
    --batch_size "${BATCH_SIZE}" \
    --n_threads "${N_THREADS}" \
    --ext "${EXT_MODE}" \
    --patch_size "${PATCH_SIZE}" \
    --resume 0 \
    --optimizer "${OPTIMIZER}" \
    --scheduler "${SCHEDULER}" \
    --decay_type "${DECAY_TYPE1}" \
    --gamma "${GAMMA}" \
    --lr "${LR}" \
    --beta1 0.9 \
    --beta2 0.99 \
    --loss "1*L1" \
    --data_train "${UCMERCED_ROOT}/train" \
    --data_val "${UCMERCED_ROOT}/val" \
    --val_every "${VAL_EVERY}" \
    --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
    --save "${SAVE_NAME1}"

python train_enhanced.py \
    --model "${MODEL_NAME2}" \
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
    --decay_type "${DECAY_TYPE1}" \
    --gamma "${GAMMA}" \
    --lr "${LR}" \
    --beta1 0.9 \
    --beta2 0.99 \
    --loss "1*L1" \
    --data_train "${UCMERCED_ROOT}/train" \
    --data_val "${UCMERCED_ROOT}/val" \
    --val_every "${VAL_EVERY}" \
    --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
    --save "${SAVE_NAME2}"
