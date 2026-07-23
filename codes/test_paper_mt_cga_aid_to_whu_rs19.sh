#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

: "${MODEL_NAME:?MODEL_NAME is required}"
: "${SCALE:?SCALE is required}"

AID_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
WHU_RS19_TEST_ROOT="${WHU_RS19_TEST_ROOT:-/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-test}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results_paper_aid_to_whu_rs19.txt}"

default_save_name() {
    case "${MODEL_NAME}:${SCALE}" in
        cga_paper:2) printf 'paper_cga_aid_x2_20260514_043252\n' ;;
        cga_paper:3) printf 'paper_cga_aid_x3_20260514_032416\n' ;;
        cga_paper:4) printf 'paper_cga_aid_x4_20260512_155448\n' ;;
        mt_paper:3) printf 'paper_mt_aid_x3_20260514_164526\n' ;;
        mt_paper:4) printf 'paper_mt_aid_x4_20260514_043403\n' ;;
        *)
            return 1
            ;;
    esac
}

if [[ -z "${SAVE_NAME:-}" ]]; then
    if ! SAVE_NAME="$(default_save_name)"; then
        echo "No default AID checkpoint for MODEL_NAME=${MODEL_NAME}, SCALE=${SCALE}. Set SAVE_NAME or MODEL_PATH manually." >&2
        exit 1
    fi
fi

PRIMARY_OUT_TAG="${PRIMARY_OUT_TAG:-${SAVE_NAME}_WHU_RS19}"

env \
    DATASET="AID" \
    TRAIN_DATASET="AID" \
    EVAL_DATASET="WHU-RS19" \
    DATASET_TAG="AID" \
    DATA_ROOT="${AID_ROOT}" \
    TRAIN_ROOT="${AID_ROOT}" \
    VAL_ROOT="${AID_ROOT}" \
    TEST_ROOT="${WHU_RS19_TEST_ROOT}" \
    MODEL_NAME="${MODEL_NAME}" \
    SCALE="${SCALE}" \
    SAVE_NAME="${SAVE_NAME}" \
    PRIMARY_OUT_TAG="${PRIMARY_OUT_TAG}" \
    EVAL_ONLY=1 \
    RESULTS_ROOT="${RESULTS_ROOT}" \
    RESULTS_FILE="${RESULTS_FILE}" \
    ./train_paper_mt_cga_dataset.sh
