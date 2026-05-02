#!/bin/bash
# UC x2 ablation script for:
# 1. non-(high batch + high lr) optimizations
# 2. all optimizations
# 3. cosine scheduler
# 4. all optimizations + cosine
# 5. low-RF MAB
#
# Existing reference runs reused by default:
# - baseline + high-RF MAB:
#   s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260429_071942
# - all optimizations + high-RF MAB:
#   s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260430_120407
# - non-(high batch + high lr) optimizations + low-RF MAB:
#   s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260430_133350

set -euo pipefail

cd "$(dirname "$0")"

MODEL_NAME="symunet_pretrain_strip_cam_parallel_add_dw5_pa_scale"
DATASET="UCMerced"
SCALE=2
MODEL_WIDTH=32
MODEL_ENC="4,6"
MODEL_DEC="6,4"
PATCH_SIZE=96
EPOCHS="${EPOCHS:-300}"
LOSS="${LOSS:-1*L1}"
OPTIMIZER="${OPTIMIZER:-ADAMW}"
SAVE_EVERY_N_EPOCHS="${SAVE_EVERY_N_EPOCHS:-100}"
UCMERCED_ROOT="${UCMERCED_ROOT:-/root/autodl-tmp/TransENet_base/datasets/UCMerced-dataset}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d_%H%M%S)}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
FORCE_DEPLOY="${FORCE_DEPLOY:-0}"

BASELINE_SAVE="s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260429_071942"
ALL_OPT_STEP_SAVE="s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260430_120407"
OPT_NO_HBLR_LOW_RF_SAVE="s1_trans_strip_cam_parallel_add_dw5_pa_scale_UCMerced_x2_w32_20260430_133350"

OPT_NO_HBLR_HIGH_RF_SAVE="${MODEL_NAME}_${DATASET}_x${SCALE}_w${MODEL_WIDTH}_opt_no_hblr_highrf_${RUN_TAG}"
ALL_OPT_COSINE_SAVE="${MODEL_NAME}_${DATASET}_x${SCALE}_w${MODEL_WIDTH}_all_opt_cosine_${RUN_TAG}"

SUMMARY_FILE="${SUMMARY_FILE:-results_uc_x2_ablation_${RUN_TAG}.tsv}"
COMPARISON_FILE="${COMPARISON_FILE:-results_uc_x2_ablation_compare_${RUN_TAG}.tsv}"
RESULTS_ROOT="${RESULTS_ROOT:-../experiment/results_ablation}"

TEST_LR_DIR="${UCMERCED_ROOT}/test/LR_x${SCALE}"
TEST_HR_DIR="${UCMERCED_ROOT}/test/HR_x${SCALE}"
TRAIN_DIR="${UCMERCED_ROOT}/train"
VAL_DIR="${UCMERCED_ROOT}/val"

declare -A CASE_SAVE
declare -A CASE_PSNR
declare -A CASE_SSIM

cat > "${SUMMARY_FILE}" <<'EOF'
case_id	save_name	psnr	ssim	note
EOF

cat > "${COMPARISON_FILE}" <<'EOF'
comparison	control	target	delta_psnr	delta_ssim
EOF

ensure_model_exists() {
    local save_name="$1"
    local model_path="../experiment/${save_name}/model/model_best.pt"
    if [[ ! -f "${model_path}" ]]; then
        echo "Missing model: ${model_path}" >&2
        return 1
    fi
}

train_case_if_needed() {
    local save_name="$1"
    local ext_mode="$2"
    local batch_size="$3"
    local lr="$4"
    local n_threads="$5"
    local scheduler="$6"
    local amp_flag="$7"
    local mab2_kernel_sizes="$8"
    local mab2_dilations="$9"

    local model_path="../experiment/${save_name}/model/model_best.pt"
    if [[ "${FORCE_RETRAIN}" != "1" && -f "${model_path}" ]]; then
        echo "Skip training, found existing model: ${save_name}"
        return 0
    fi

    local -a cmd=(
        python train_enhanced.py
        --model "${MODEL_NAME}"
        --dataset "${DATASET}"
        --scale "${SCALE}"
        --epochs "${EPOCHS}"
        --batch_size "${batch_size}"
        --n_threads "${n_threads}"
        --ext "${ext_mode}"
        --patch_size "${PATCH_SIZE}"
        --resume 0
        --optimizer "${OPTIMIZER}"
        --scheduler "${scheduler}"
        --lr "${lr}"
        --loss "${LOSS}"
        --symunet_pretrain_width "${MODEL_WIDTH}"
        --symunet_pretrain_enc_blk_nums "${MODEL_ENC}"
        --symunet_pretrain_dec_blk_nums "${MODEL_DEC}"
        --symunet_pretrain_strip_k2 19
        --symunet_pretrain_mab2_kernel_sizes "${mab2_kernel_sizes}"
        --symunet_pretrain_mab2_dilations "${mab2_dilations}"
        --data_train "${TRAIN_DIR}"
        --data_val "${VAL_DIR}"
        --save_every_n_epochs "${SAVE_EVERY_N_EPOCHS}"
        --save "${save_name}"
    )

    if [[ "${scheduler}" == "cosine" ]]; then
        cmd+=(--cosine_t_max "${EPOCHS}" --cosine_eta_min 1e-5)
    fi

    if [[ "${amp_flag}" == "1" ]]; then
        cmd+=(--amp)
    fi

    "${cmd[@]}"
}

deploy_and_eval_case() {
    local case_id="$1"
    local save_name="$2"
    local note="$3"
    local mab2_kernel_sizes="$4"
    local mab2_dilations="$5"

    local model_path="../experiment/${save_name}/model/model_best.pt"
    local out_dir="${RESULTS_ROOT}/${save_name}/x${SCALE}"
    local metric_output
    local metric_line
    local psnr
    local ssim

    mkdir -p "${out_dir}"

    if [[ "${FORCE_DEPLOY}" == "1" || ! -f "${out_dir}/airplane50.tif" ]]; then
        python demo_deploy.py \
            --model "${MODEL_NAME}" \
            --dataset "${DATASET}" \
            --scale "${SCALE}" \
            --symunet_pretrain_width "${MODEL_WIDTH}" \
            --symunet_pretrain_enc_blk_nums "${MODEL_ENC}" \
            --symunet_pretrain_dec_blk_nums "${MODEL_DEC}" \
            --symunet_pretrain_strip_k2 19 \
            --symunet_pretrain_mab2_kernel_sizes "${mab2_kernel_sizes}" \
            --symunet_pretrain_mab2_dilations "${mab2_dilations}" \
            --pre_train "${model_path}" \
            --dir_data "${TEST_LR_DIR}" \
            --dir_out "${out_dir}" >/tmp/"${case_id}"_deploy.log
    fi

    metric_output="$(python calculate_PSNR_SSIM.py \
        --dataset "${DATASET}" \
        --scale "${SCALE}" \
        --folder_GT "${TEST_HR_DIR}" \
        --folder_Gen "${out_dir}")"
    metric_line="$(printf '%s\n' "${metric_output}" | tail -n 1)"
    read -r psnr ssim < <(printf '%s\n' "${metric_line}" | sed -E 's/.*PSNR: ([0-9.]+) dB, SSIM: ([0-9.]+).*/\1 \2/')

    CASE_SAVE["${case_id}"]="${save_name}"
    CASE_PSNR["${case_id}"]="${psnr}"
    CASE_SSIM["${case_id}"]="${ssim}"

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${case_id}" "${save_name}" "${psnr}" "${ssim}" "${note}" >> "${SUMMARY_FILE}"
}

append_comparison() {
    local comparison_name="$1"
    local control_case="$2"
    local target_case="$3"
    local control_psnr="${CASE_PSNR[$control_case]}"
    local target_psnr="${CASE_PSNR[$target_case]}"
    local control_ssim="${CASE_SSIM[$control_case]}"
    local target_ssim="${CASE_SSIM[$target_case]}"
    local delta_psnr
    local delta_ssim

    delta_psnr="$(python - <<PY
control = float("${control_psnr}")
target = float("${target_psnr}")
print(f"{target - control:.6f}")
PY
)"
    delta_ssim="$(python - <<PY
control = float("${control_ssim}")
target = float("${target_ssim}")
print(f"{target - control:.6f}")
PY
)"

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${comparison_name}" \
        "${control_case}" \
        "${target_case}" \
        "${delta_psnr}" \
        "${delta_ssim}" >> "${COMPARISON_FILE}"
}

echo "Running UC x2 ablation suite"
echo "Summary: ${SUMMARY_FILE}"
echo "Comparison: ${COMPARISON_FILE}"

# Existing references
ensure_model_exists "${BASELINE_SAVE}"
ensure_model_exists "${ALL_OPT_STEP_SAVE}"
ensure_model_exists "${OPT_NO_HBLR_LOW_RF_SAVE}"

# New case 1: non-(high batch + high lr) optimizations, high-RF MAB
train_case_if_needed \
    "${OPT_NO_HBLR_HIGH_RF_SAVE}" \
    "sep_reset" \
    "4" \
    "2e-4" \
    "8" \
    "step" \
    "1" \
    "7,11" \
    "5,3"

# New case 2: all optimizations + cosine, high-RF MAB
train_case_if_needed \
    "${ALL_OPT_COSINE_SAVE}" \
    "sep" \
    "16" \
    "4e-4" \
    "16" \
    "cosine" \
    "1" \
    "7,11" \
    "5,3"

ensure_model_exists "${OPT_NO_HBLR_HIGH_RF_SAVE}"
ensure_model_exists "${ALL_OPT_COSINE_SAVE}"

deploy_and_eval_case \
    "baseline_highrf_step" \
    "${BASELINE_SAVE}" \
    "baseline, no extra optimization, high-RF MAB" \
    "7,11" \
    "5,3"

deploy_and_eval_case \
    "opt_no_hblr_highrf_step" \
    "${OPT_NO_HBLR_HIGH_RF_SAVE}" \
    "non-(high batch + high lr) optimizations, high-RF MAB, step" \
    "7,11" \
    "5,3"

deploy_and_eval_case \
    "all_opt_highrf_step" \
    "${ALL_OPT_STEP_SAVE}" \
    "all optimizations, high-RF MAB, step" \
    "7,11" \
    "5,3"

deploy_and_eval_case \
    "all_opt_highrf_cosine" \
    "${ALL_OPT_COSINE_SAVE}" \
    "all optimizations, high-RF MAB, cosine" \
    "7,11" \
    "5,3"

deploy_and_eval_case \
    "opt_no_hblr_lowrf_step" \
    "${OPT_NO_HBLR_LOW_RF_SAVE}" \
    "non-(high batch + high lr) optimizations, low-RF MAB, step" \
    "5,9" \
    "4,2"

append_comparison "non_hblr_optim_impact" "baseline_highrf_step" "opt_no_hblr_highrf_step"
append_comparison "all_optim_impact" "baseline_highrf_step" "all_opt_highrf_step"
append_comparison "cosine_impact" "all_opt_highrf_step" "all_opt_highrf_cosine"
append_comparison "all_optim_plus_cosine_impact" "baseline_highrf_step" "all_opt_highrf_cosine"
append_comparison "low_rf_mab_impact" "opt_no_hblr_highrf_step" "opt_no_hblr_lowrf_step"

echo
echo "Ablation summary saved to: ${SUMMARY_FILE}"
cat "${SUMMARY_FILE}"
echo
echo "Comparison summary saved to: ${COMPARISON_FILE}"
cat "${COMPARISON_FILE}"
