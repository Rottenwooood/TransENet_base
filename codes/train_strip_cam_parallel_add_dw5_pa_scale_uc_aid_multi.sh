#!/bin/bash
# Train, deploy and evaluate strip-cam parallel add dw5+pa_scale on:
# UCMerced x2/x3 and AID x2/x3/x4

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="symunet_pretrain_strip_cam_parallel_add_dw5_pa_scale"
MODEL_WIDTH=32
MODEL_ENC="4,6"
MODEL_DEC="6,4"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LOSS="${LOSS:-1*L1}"
OPTIMIZER="${OPTIMIZER:-ADAMW}"
SCHEDULER="${SCHEDULER:-step}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-100}"
N_THREADS="${N_THREADS:-8}"
VAL_EVERY="${VAL_EVERY:-5}"
EXT_MODE="${EXT_MODE:-sep}"

UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
AID_ROOT="${AID_ROOT:-/root/autodl-tmp/TransENet_base/datasets/AID-dataset}"

RESULTS_SUMMARY="${RESULTS_SUMMARY:-results_strip_cam_parallel_add_dw5_pa_scale_multi.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

PATCH_SIZE_X2=96
PATCH_SIZE_X3=144
PATCH_SIZE_X4=192

touch "${RESULTS_SUMMARY}"

get_strip_k2() {
    local scale="$1"
    case "${scale}" in
        2) echo 19 ;;
        3) echo 27 ;;
        *) echo 47 ;;
    esac
}

run_case() {
    local dataset="$1"
    local scale="$2"
    local patch_size="$3"
    local dataset_root="$4"
    local strip_k2
    strip_k2="$(get_strip_k2 "${scale}")"

    local exp_prefix="s1_trans_strip_cam_parallel_add_dw5_pa_scale_${dataset}_x${scale}"
    local save_name="${exp_prefix}_w${MODEL_WIDTH}_${RUN_TS}"
    local model_path="../experiment/${save_name}/model/model_best.pt"
    local out_dir="../experiment/results/${save_name}/x${scale}"
    local test_lr_dir="${dataset_root}/test/LR_x${scale}"
    local test_hr_dir="${dataset_root}/test/HR_x${scale}"

    echo
    echo "============================================================"
    echo "Running ${dataset} x${scale}"
    echo "============================================================"
    echo "Train root: ${dataset_root}"
    echo "Test LR dir: ${test_lr_dir}"
    echo "Test HR dir: ${test_hr_dir}"
    echo "Strip k2: ${strip_k2}"
    echo "Save name: ${save_name}"
    echo "Output dir: ${out_dir}"
    echo "Ext mode: ${EXT_MODE}"

    python train_enhanced.py \
        --model "${MODEL_NAME}" \
        --dataset "${dataset}" \
        --scale "${scale}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --n_threads "${N_THREADS}" \
        --amp \
        --ext "${EXT_MODE}" \
        --patch_size "${patch_size}" \
        --resume 0 \
        --optimizer "${OPTIMIZER}" \
        --scheduler "${SCHEDULER}" \
        --lr "${LR}" \
        --loss "${LOSS}" \
        --symunet_pretrain_width "${MODEL_WIDTH}" \
        --symunet_pretrain_enc_blk_nums "${MODEL_ENC}" \
        --symunet_pretrain_dec_blk_nums "${MODEL_DEC}" \
        --symunet_pretrain_strip_k2 "${strip_k2}" \
        --data_train "${dataset_root}/train" \
        --data_val "${dataset_root}/val" \
        --val_every "${VAL_EVERY}" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --save "${save_name}"

    python demo_deploy.py \
        --model "${MODEL_NAME}" \
        --dataset "${dataset}" \
        --scale "${scale}" \
        --symunet_pretrain_width "${MODEL_WIDTH}" \
        --symunet_pretrain_enc_blk_nums "${MODEL_ENC}" \
        --symunet_pretrain_dec_blk_nums "${MODEL_DEC}" \
        --symunet_pretrain_strip_k2 "${strip_k2}" \
        --pre_train "${model_path}" \
        --dir_data "${test_lr_dir}" \
        --dir_out "${out_dir}" | tail -n 1 >> results.txt

    python calculate_PSNR_SSIM.py \
        --dataset "${dataset}" \
        --scale "${scale}" \
        --folder_GT "${test_hr_dir}" \
        --folder_Gen "${out_dir}" | tail -n 1 >> results.txt
}

run_case "UCMerced" 2 "${PATCH_SIZE_X2}" "${UCMERCED_ROOT}"
run_case "UCMerced" 3 "${PATCH_SIZE_X3}" "${UCMERCED_ROOT}"
run_case "AID" 2 "${PATCH_SIZE_X2}" "${AID_ROOT}"
run_case "AID" 3 "${PATCH_SIZE_X3}" "${AID_ROOT}"
run_case "AID" 4 "${PATCH_SIZE_X4}" "${AID_ROOT}"
