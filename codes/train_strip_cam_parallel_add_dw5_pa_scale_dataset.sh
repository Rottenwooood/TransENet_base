#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

: "${DATASET:?DATASET is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"


CONDA_ENV="${CONDA_ENV:-transenet-pren}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

DATASET_TAG="${DATASET_TAG:-${DATASET}}"
TRAIN_DATASET="${TRAIN_DATASET:-${DATASET}}"
EVAL_DATASET="${EVAL_DATASET:-${DATASET}}"
TRAIN_ROOT="${TRAIN_ROOT:-${DATA_ROOT}}"
VAL_ROOT="${VAL_ROOT:-${TRAIN_ROOT}}"
TEST_ROOT="${TEST_ROOT:-${DATA_ROOT}}"
MODEL_NAME="${MODEL_NAME:-symunet_pretrain_strip_cam_parallel_add_dw5_pa_scale}"
SCALE="${SCALE:-4}"
EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LOSS="${LOSS:-1*L1}"
OPTIMIZER="${OPTIMIZER:-ADAMW}"
SCHEDULER="${SCHEDULER:-step}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-2}"
EXT_MODE="${EXT_MODE:-sep}"
RESUME="${RESUME:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results_strip_cam_parallel_add_dw5_pa_scale_${DATASET_TAG}.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

case "${SCALE}" in
    2)
        PATCH_SIZE="${PATCH_SIZE:-96}"
        STRIP_K2="${STRIP_K2:-19}"
        ;;
    3)
        PATCH_SIZE="${PATCH_SIZE:-144}"
        STRIP_K2="${STRIP_K2:-27}"
        ;;
    4)
        PATCH_SIZE="${PATCH_SIZE:-192}"
        STRIP_K2="${STRIP_K2:-47}"
        ;;
    *)
        echo "Unsupported SCALE=${SCALE}; expected 2, 3, or 4" >&2
        exit 1
        ;;
esac

MODEL_WIDTH="${MODEL_WIDTH:-32}"
MODEL_ENC="${MODEL_ENC:-4,6}"
MODEL_DEC="${MODEL_DEC:-6,4}"
MAB2_KERNEL_SIZES="${MAB2_KERNEL_SIZES:-7,11}"
MAB2_DILATIONS="${MAB2_DILATIONS:-5,3}"
SAVE_NAME="${SAVE_NAME:-COMSNET_${DATASET_TAG}_x${SCALE}_w${MODEL_WIDTH}_${RUN_TS}}"

if [[ "${RESUME}" == "1" ]]; then
    for resume_file in \
        "../experiment/${SAVE_NAME}/model/model_latest.pt" \
        "../experiment/${SAVE_NAME}/optimizer.pt" \
        "../experiment/${SAVE_NAME}/psnr_log.pt"; do
        if [[ ! -f "${resume_file}" ]]; then
            echo "Missing resume file: ${resume_file}" >&2
            exit 1
        fi
    done
fi

"${PYTHON_BIN}" train_enhanced.py \
    --model "${MODEL_NAME}" \
    --dataset "${TRAIN_DATASET}" \
    --scale "${SCALE}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --n_threads "${N_THREADS}" \
    --amp \
    --ext "${EXT_MODE}" \
    --patch_size "${PATCH_SIZE}" \
    --resume "${RESUME}" \
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
    --data_train "${TRAIN_ROOT}/train" \
    --data_val "${VAL_ROOT}/val" \
    --val_every "${VAL_EVERY}" \
    --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
    --save "${SAVE_NAME}"

MODEL_PATH="../experiment/${SAVE_NAME}/model/model_best.pt"
OUT_DIR="${RESULTS_ROOT}/${SAVE_NAME}/x${SCALE}"
TEST_LR_DIR="${TEST_ROOT}/test/LR_x${SCALE}"
TEST_HR_DIR="${TEST_ROOT}/test/HR"

{
    echo "=== train=${TRAIN_DATASET} eval=${EVAL_DATASET} x${SCALE} | save=${SAVE_NAME} ==="
    "${PYTHON_BIN}" demo_deploy.py \
        --model "${MODEL_NAME}" \
        --dataset "${EVAL_DATASET}" \
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

    "${PYTHON_BIN}" calculate_PSNR_SSIM.py \
        --dataset "${EVAL_DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${TEST_HR_DIR}" \
        --folder_Gen "${OUT_DIR}" | tail -n 1

    "${PYTHON_BIN}" calculate_SCC_SAM.py \
        --dataset "${EVAL_DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${TEST_HR_DIR}" \
        --folder_Gen "${OUT_DIR}" | tail -n 1
    echo
} >> "${RESULTS_FILE}"
