#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET="AID"
MODEL_NAME="symunet_pretrain_strip_cam_parallel_add_dw5_pa_scale"
SCALE=3
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LOSS="${LOSS:-1*L1}"
OPTIMIZER="${OPTIMIZER:-ADAMW}"
SCHEDULER="${SCHEDULER:-step}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-2}"
EXT_MODE="${EXT_MODE:-sep}"

AID_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results_strip_cam_parallel_add_dw5_pa_scale_aid.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

PATCH_SIZE="${PATCH_SIZE:-144}"
MODEL_WIDTH="${MODEL_WIDTH:-32}"
MODEL_ENC="${MODEL_ENC:-4,6}"
MODEL_DEC="${MODEL_DEC:-6,4}"
STRIP_K2="${STRIP_K2:-27}"
MAB2_KERNEL_SIZES="${MAB2_KERNEL_SIZES:-7,11}"
MAB2_DILATIONS="${MAB2_DILATIONS:-5,3}"
SAVE_NAME="${SAVE_NAME:-s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x3_w${MODEL_WIDTH}_${RUN_TS}}"

python train_enhanced.py \
    --model "${MODEL_NAME}" \
    --dataset "${DATASET}" \
    --scale "${SCALE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --n_threads "${N_THREADS}" \
    --amp \
    --ext "${EXT_MODE}" \
    --patch_size "${PATCH_SIZE}" \
    --resume 0 \
    --optimizer "${OPTIMIZER}" \
    --scheduler "${SCHEDULER}" \
    --lr "${LR}" \
    --loss "${LOSS}" \
    --symunet_pretrain_width "${MODEL_WIDTH}" \
    --symunet_pretrain_enc_blk_nums "${MODEL_ENC}" \
    --symunet_pretrain_dec_blk_nums "${MODEL_DEC}" \
    --symunet_pretrain_strip_k2 "${STRIP_K2}" \
    --symunet_pretrain_mab2_kernel_sizes "${MAB2_KERNEL_SIZES}" \
    --symunet_pretrain_mab2_dilations "${MAB2_DILATIONS}" \
    --data_train "${AID_ROOT}/train" \
    --data_val "${AID_ROOT}/val" \
    --val_every "${VAL_EVERY}" \
    --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
    --save "${SAVE_NAME}"

MODEL_PATH="../experiment/${SAVE_NAME}/model/model_best.pt"
OUT_DIR="${RESULTS_ROOT}/${SAVE_NAME}/x${SCALE}"
TEST_LR_DIR="${AID_ROOT}/test/LR_x${SCALE}"
TEST_HR_DIR="${AID_ROOT}/test/HR"

{
    echo "=== ${DATASET} x${SCALE} | save=${SAVE_NAME} ==="
    python demo_deploy.py \
        --model "${MODEL_NAME}" \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --symunet_pretrain_width "${MODEL_WIDTH}" \
        --symunet_pretrain_enc_blk_nums "${MODEL_ENC}" \
        --symunet_pretrain_dec_blk_nums "${MODEL_DEC}" \
        --symunet_pretrain_strip_k2 "${STRIP_K2}" \
        --symunet_pretrain_mab2_kernel_sizes "${MAB2_KERNEL_SIZES}" \
        --symunet_pretrain_mab2_dilations "${MAB2_DILATIONS}" \
        --pre_train "${MODEL_PATH}" \
        --dir_data "${TEST_LR_DIR}" \
        --dir_out "${OUT_DIR}" | tail -n 1

    python calculate_PSNR_SSIM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${TEST_HR_DIR}" \
        --folder_Gen "${OUT_DIR}" | tail -n 1

    python calculate_SCC_SAM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${TEST_HR_DIR}" \
        --folder_Gen "${OUT_DIR}" | tail -n 1
    echo
} >> "${RESULTS_FILE}"
