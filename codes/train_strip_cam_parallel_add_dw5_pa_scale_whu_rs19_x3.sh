#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export DATASET="AID"
export TRAIN_DATASET="AID"
export EVAL_DATASET="WHU-RS19"
export DATASET_TAG="AID_to_WHU_RS19"
export DATA_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
export TRAIN_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
export VAL_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"
export TEST_ROOT="${WHU_RS19_TEST_ROOT:-/root/autodl-tmp/TransENet_base/datasets/WHU-RS19-test}"
export SCALE=3
export RESULTS_FILE="${RESULTS_FILE:-results_strip_cam_parallel_add_dw5_pa_scale_aid_to_whu_rs19.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec ./train_strip_cam_parallel_add_dw5_pa_scale_dataset.sh
