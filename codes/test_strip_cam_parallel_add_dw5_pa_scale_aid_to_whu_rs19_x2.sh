#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SCALE=2
export SAVE_NAME="${SAVE_NAME:-s1_trans_strip_cam_parallel_add_dw5_pa_scale_AID_x2_w32_20260712_112935}"

exec ./test_strip_cam_parallel_add_dw5_pa_scale_aid_to_whu_rs19_dataset.sh
