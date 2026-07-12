#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="symunet_pretrain_strip_cam_parallel_add_dw5_pa_scale"
DATASET="UCMerced"
MODEL_WIDTH=32
MODEL_ENC="4,6"
MODEL_DEC="6,4"

# The two resume runs come from the original no-suffix experiments, which used
# the model defaults recorded in their config files.
STRIP_K2="${STRIP_K2:-47}"
MAB2_KERNEL_SIZES="${MAB2_KERNEL_SIZES:-7,11}"
MAB2_DILATIONS="${MAB2_DILATIONS:-5,3}"

EPOCHS="${EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
LOSS="${LOSS:-1*L1}"
OPTIMIZER="${OPTIMIZER:-ADAMW}"
SCHEDULER="${SCHEDULER:-step}"
SAVE_EVERY_N_STEPS="${SAVE_EVERY_N_STEPS:-50}"
N_THREADS="${N_THREADS:-4}"
EXT_MODE="${EXT_MODE:-img}"

PATCH_SIZE_X2=96
PATCH_SIZE_X3=144
PATCH_SIZE_X4=192

GPU_X2="${GPU_X2:-0}"
GPU_X3="${GPU_X3:-1}"
GPU_X4="${GPU_X4:-2}"

UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results}"
RESULTS_FILE="${RESULTS_FILE:-results_strip_cam_parallel_add_dw5_pa_scale_uc_gpu3_resume500.txt}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"

SAVE_X2="s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32"
SAVE_X3="s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x3_w32"
SAVE_X4="${SAVE_X4:-s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x4_w32_${RUN_TS}}"

touch "${RESULTS_FILE}"

ensure_resume_case() {
    local save_name="$1"

    if [[ ! -f "../experiment/${save_name}/model/model_latest.pt" ]]; then
        echo "Missing resume model: ../experiment/${save_name}/model/model_latest.pt" >&2
        exit 1
    fi

    if [[ ! -f "../experiment/${save_name}/optimizer.pt" ]]; then
        echo "Missing resume optimizer: ../experiment/${save_name}/optimizer.pt" >&2
        exit 1
    fi

    if [[ ! -f "../experiment/${save_name}/psnr_log.pt" ]]; then
        echo "Missing resume metric log: ../experiment/${save_name}/psnr_log.pt" >&2
        exit 1
    fi
}

run_case() {
    local gpu_id="$1"
    local scale="$2"
    local patch_size="$3"
    local resume_flag="$4"
    local save_name="$5"

    local model_path="../experiment/${save_name}/model/model_best.pt"
    local out_dir="${RESULTS_ROOT}/${save_name}/x${scale}"
    local test_lr_dir="${UCMERCED_ROOT}/test/LR_x${scale}"
    local test_hr_dir="${UCMERCED_ROOT}/test/HR_x${scale}"
    local tmp_result

    mkdir -p "${out_dir}"
    tmp_result="$(mktemp)"

    export CUDA_VISIBLE_DEVICES="${gpu_id}"

    echo
    echo "============================================================"
    echo "Running ${DATASET} x${scale} on GPU ${gpu_id}"
    echo "============================================================"
    echo "Resume: ${resume_flag}"
    echo "Save name: ${save_name}"
    echo "Patch size: ${patch_size}"
    echo "Results dir: ${out_dir}"

    python train_enhanced.py \
        --model "${MODEL_NAME}" \
        --dataset "${DATASET}" \
        --scale "${scale}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --n_threads "${N_THREADS}" \
        --ext "${EXT_MODE}" \
        --patch_size "${patch_size}" \
        --resume "${resume_flag}" \
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
        --data_train "${UCMERCED_ROOT}/train" \
        --data_val "${UCMERCED_ROOT}/val" \
        --save_every_n_steps "${SAVE_EVERY_N_STEPS}" \
        --save "${save_name}"

    {
        echo "=== ${DATASET} x${scale} | save=${save_name} | gpu=${gpu_id} ==="
        python demo_deploy.py \
            --model "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --scale "${scale}" \
            --symunet_pretrain_width "${MODEL_WIDTH}" \
            --symunet_pretrain_enc_blk_nums "${MODEL_ENC}" \
            --symunet_pretrain_dec_blk_nums "${MODEL_DEC}" \
            --symunet_pretrain_strip_k2 "${STRIP_K2}" \
            --symunet_pretrain_mab2_kernel_sizes "${MAB2_KERNEL_SIZES}" \
            --symunet_pretrain_mab2_dilations "${MAB2_DILATIONS}" \
            --pre_train "${model_path}" \
            --dir_data "${test_lr_dir}" \
            --dir_out "${out_dir}" | tail -n 1

        python calculate_PSNR_SSIM.py \
            --dataset "${DATASET}" \
            --scale "${scale}" \
            --folder_GT "${test_hr_dir}" \
            --folder_Gen "${out_dir}" | tail -n 1
        echo
    } >> "${tmp_result}"

    cat "${tmp_result}" >> "${RESULTS_FILE}"
    rm -f "${tmp_result}"
}

ensure_resume_case "${SAVE_X2}"
ensure_resume_case "${SAVE_X3}"

pids=()
labels=()

run_case "${GPU_X2}" 2 "${PATCH_SIZE_X2}" 1 "${SAVE_X2}" &
pids+=("$!")
labels+=("UCMerced x2 resume")

run_case "${GPU_X3}" 3 "${PATCH_SIZE_X3}" 1 "${SAVE_X3}" &
pids+=("$!")
labels+=("UCMerced x3 resume")

run_case "${GPU_X4}" 4 "${PATCH_SIZE_X4}" 0 "${SAVE_X4}" &
pids+=("$!")
labels+=("UCMerced x4 from scratch")

status=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "Failed: ${labels[$idx]}" >&2
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    exit "${status}"
fi

echo "All jobs finished."
echo "Results summary: ${RESULTS_FILE}"
