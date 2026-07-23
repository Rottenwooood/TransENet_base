#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

: "${DATASET:?DATASET is required}"
: "${DATA_ROOT:?DATA_ROOT is required}"
: "${MODEL_NAME:?MODEL_NAME is required}"
: "${SCALE:?SCALE is required}"

CONDA_ENV="${CONDA_ENV:-transenet-pren}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/${CONDA_ENV}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="python"
fi

TRAIN_DATASET="${TRAIN_DATASET:-${DATASET}}"
EVAL_DATASET="${EVAL_DATASET:-${DATASET}}"
DATASET_TAG="${DATASET_TAG:-${DATASET}}"
TRAIN_ROOT="${TRAIN_ROOT:-${DATA_ROOT}}"
VAL_ROOT="${VAL_ROOT:-${TRAIN_ROOT}}"
TEST_ROOT="${TEST_ROOT:-${DATA_ROOT}}"

EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-0}"
LR="${LR:-2e-4}"
LOSS="${LOSS:-1*L1}"
OPTIMIZER="${OPTIMIZER:-ADAM}"
SCHEDULER="${SCHEDULER:-step}"
DECAY_TYPE="${DECAY_TYPE:-step}"
GAMMA="${GAMMA:-0.5}"
MAX_STEPS="${MAX_STEPS:-500000}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
EXT_MODE="${EXT_MODE:-sep}"
RESUME="${RESUME:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
SKIP_PRIMARY_EVAL="${SKIP_PRIMARY_EVAL:-0}"
SKIP_EXTRA_EVAL="${SKIP_EXTRA_EVAL:-0}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"

sanitize_tag() {
    local value="$1"
    value="${value//-/_}"
    value="${value//\//_}"
    value="${value// /_}"
    printf '%s\n' "${value}"
}

case "${MODEL_NAME}" in
    mt_paper)
        MODEL_TAG="mt"
        EXTRA_MODEL_ARGS=()
        ;;
    cga_paper)
        MODEL_TAG="cga"
        CGA_SPLIT_SIZE="${CGA_SPLIT_SIZE:-8,32}"
        EXTRA_MODEL_ARGS=(--cga_paper_split_size "${CGA_SPLIT_SIZE}")
        ;;
    *)
        echo "Unsupported MODEL_NAME=${MODEL_NAME}; expected mt_paper or cga_paper" >&2
        exit 1
        ;;
esac

case "${SCALE}" in
    2)
        PATCH_SIZE="${PATCH_SIZE:-128}"
        ;;
    3)
        PATCH_SIZE="${PATCH_SIZE:-192}"
        ;;
    4)
        PATCH_SIZE="${PATCH_SIZE:-256}"
        ;;
    *)
        echo "Unsupported SCALE=${SCALE}; expected 2, 3, or 4" >&2
        exit 1
        ;;
esac

SAVE_NAME="${SAVE_NAME:-paper_${MODEL_TAG}_$(sanitize_tag "${DATASET_TAG}")_x${SCALE}_${RUN_TS}}"
RESULTS_FILE="${RESULTS_FILE:-results_paper_${MODEL_TAG}_$(sanitize_tag "${DATASET_TAG}").txt}"
PRIMARY_OUT_TAG="${PRIMARY_OUT_TAG:-${SAVE_NAME}}"
MODEL_PATH="${MODEL_PATH:-../experiment/${SAVE_NAME}/model/model_best.pt}"

require_eval_root() {
    local eval_root="$1"
    for required_path in \
        "${eval_root}/test/LR_x${SCALE}" \
        "${eval_root}/test/HR"; do
        if [[ ! -e "${required_path}" ]]; then
            echo "Missing required path: ${required_path}" >&2
            exit 1
        fi
    done
}

if [[ "${EVAL_ONLY}" != "1" ]]; then
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

    for required_dir in \
        "${TRAIN_ROOT}/train" \
        "${VAL_ROOT}/val"; do
        if [[ ! -e "${required_dir}" ]]; then
            echo "Missing required path: ${required_dir}" >&2
            exit 1
        fi
    done

    "${PYTHON_BIN}" train_enhanced.py \
        --model "${MODEL_NAME}" \
        --dataset "${TRAIN_DATASET}" \
        --scale "${SCALE}" \
        --epochs "${EPOCHS}" \
        --max_steps "${MAX_STEPS}" \
        --scheduler_unit epoch \
        --batch_size "${BATCH_SIZE}" \
        --val_batch_size "${VAL_BATCH_SIZE}" \
        --n_threads "${N_THREADS}" \
        --ext "${EXT_MODE}" \
        --patch_size "${PATCH_SIZE}" \
        --resume "${RESUME}" \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --decay_type "${DECAY_TYPE}" \
        --gamma "${GAMMA}" \
        --lr "${LR}" \
        --beta1 0.9 \
        --beta2 0.99 \
        --loss "${LOSS}" \
        "${EXTRA_MODEL_ARGS[@]}" \
        --data_train "${TRAIN_ROOT}/train" \
        --data_val "${VAL_ROOT}/val" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --save "${SAVE_NAME}"
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Missing model checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi

run_eval() {
    local stage_tag="$1"
    local eval_dataset="$2"
    local eval_root="$3"
    local out_tag="$4"
    local result_file="$5"

    local out_dir="${RESULTS_ROOT}/${out_tag}/x${SCALE}"
    local test_lr_dir="${eval_root}/test/LR_x${SCALE}"
    local test_hr_dir="${eval_root}/test/HR"

    for required_path in "${MODEL_PATH}" "${test_lr_dir}" "${test_hr_dir}"; do
        if [[ ! -e "${required_path}" ]]; then
            echo "Missing required path: ${required_path}" >&2
            exit 1
        fi
    done

    {
        echo "=== stage=${stage_tag} train=${TRAIN_DATASET} eval=${eval_dataset} model=${MODEL_NAME} x${SCALE} | save=${SAVE_NAME} ==="
        "${PYTHON_BIN}" demo_deploy.py \
            --model "${MODEL_NAME}" \
            --dataset "${eval_dataset}" \
            --scale "${SCALE}" \
            "${EXTRA_MODEL_ARGS[@]}" \
            --pre_train "${MODEL_PATH}" \
            --dir_data "${test_lr_dir}" \
            --dir_out "${out_dir}" | tail -n 1

        "${PYTHON_BIN}" calculate_PSNR_SSIM.py \
            --dataset "${eval_dataset}" \
            --scale "${SCALE}" \
            --folder_GT "${test_hr_dir}" \
            --folder_Gen "${out_dir}" | tail -n 1

        "${PYTHON_BIN}" calculate_SCC_SAM.py \
            --dataset "${eval_dataset}" \
            --scale "${SCALE}" \
            --folder_GT "${test_hr_dir}" \
            --folder_Gen "${out_dir}" | tail -n 1
        echo
    } >> "${result_file}"
}

if [[ "${SKIP_PRIMARY_EVAL}" != "1" ]]; then
    require_eval_root "${TEST_ROOT}"
    run_eval "primary" "${EVAL_DATASET}" "${TEST_ROOT}" "${PRIMARY_OUT_TAG}" "${RESULTS_FILE}"
fi

if [[ "${SKIP_EXTRA_EVAL}" != "1" && ( -n "${EXTRA_EVAL_DATASET:-}" || -n "${EXTRA_TEST_ROOT:-}" ) ]]; then
    : "${EXTRA_EVAL_DATASET:?EXTRA_EVAL_DATASET is required when EXTRA_TEST_ROOT is set}"
    : "${EXTRA_TEST_ROOT:?EXTRA_TEST_ROOT is required when EXTRA_EVAL_DATASET is set}"
    EXTRA_RESULTS_FILE="${EXTRA_RESULTS_FILE:-${RESULTS_FILE}}"
    EXTRA_OUT_TAG="${EXTRA_OUT_TAG:-${SAVE_NAME}_$(sanitize_tag "${EXTRA_EVAL_DATASET}")}"
    require_eval_root "${EXTRA_TEST_ROOT}"
    run_eval "extra_test_only" "${EXTRA_EVAL_DATASET}" "${EXTRA_TEST_ROOT}" "${EXTRA_OUT_TAG}" "${EXTRA_RESULTS_FILE}"
fi
