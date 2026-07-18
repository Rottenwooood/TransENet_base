#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export DATASET="RSSCN7"
export DATASET_TAG="RSSCN7"
export DATA_ROOT="${RSSCN7_ROOT:-/root/autodl-tmp/TransENet_base/datasets/RSSCN7-dataset}"
export SCALE=4
export RESULTS_FILE="${RESULTS_FILE:-results_strip_cam_parallel_add_dw5_pa_scale_rsscn7.txt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export SAVE_NAME="${SAVE_NAME:-COMSNET_RSSCN7_x4_w32_20260717_035733}"
export RESUME="${RESUME:-1}"

exec ./train_strip_cam_parallel_add_dw5_pa_scale_dataset.sh
