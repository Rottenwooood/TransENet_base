#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

: "${SAVE_NAME:?SAVE_NAME is required}"

CONDA_ENV="${CONDA_ENV:-transenet-pren}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

MODEL_NAME="${MODEL_NAME:-symunet_pretrain_strip_cam_parallel_add_dw5_pa_scale}"
TRAIN_DATASET="${TRAIN_DATASET:-AID}"
EVAL_DATASET="${EVAL_DATASET:-WHU-RS19}"
SCALE="${SCALE:-4}"

case "${SCALE}" in
    2)
        STRIP_K2="${STRIP_K2:-19}"
        ;;
    3)
        STRIP_K2="${STRIP_K2:-27}"
        ;;
    4)
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

TEST_ROOT="${WHU_RS19_TEST_ROOT:-/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-test}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results_strip_cam_parallel_add_dw5_pa_scale_aid_to_whu_rs19_test_only.txt}"

MODEL_PATH="${MODEL_PATH:-../experiment/${SAVE_NAME}/model/model_best.pt}"
OUT_TAG="${OUT_TAG:-${SAVE_NAME}_WHU_RS19}"
OUT_DIR="${OUT_DIR:-${RESULTS_ROOT}/${OUT_TAG}/x${SCALE}}"
TEST_LR_DIR="${TEST_ROOT}/test/LR_x${SCALE}"
TEST_HR_DIR="${TEST_ROOT}/test/HR"

for required_path in "${MODEL_PATH}" "${TEST_LR_DIR}" "${TEST_HR_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required path: ${required_path}" >&2
        exit 1
    fi
done

{
    echo "=== train=${TRAIN_DATASET} eval=${EVAL_DATASET} x${SCALE} | save=${SAVE_NAME} | test_only ==="
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
