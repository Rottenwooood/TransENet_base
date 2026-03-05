#!/bin/bash
# Training script for SymUNet_Pretrain_CAM (Channel Attention Middle)
# Parameters hardcoded in the script

# Base settings
MODEL="SYMUNET_PRETRAIN_CAM"
DATASET="UCMerced"
SCALE=4
PATCH_SIZE=192

# Training settings
EPOCHS=500
BATCH_SIZE=4
LR=0.0001

# SymUNet-Pretrain specific settings (hardcoded)
WIDTH=48
ENC_BLK_NUMS="4,6,6"
DEC_BLK_NUMS="6,6,4"
RESTORMER_HEADS="1,2,4"
MIDDLE_BLK_NUM=1
FFN_EXPANSION=2.66

# Experiment name
SAVE="SYMUNET_PRETRAIN_CAM_x${SCALE}"

echo "Training ${MODEL} on ${DATASET} with scale ${SCALE}"
echo "Save name: ${SAVE}"

cd /home/c6h4o2/dev/TransENet_base/codes

python demo_train.py \
    --model=${MODEL} \
    --dataset=${DATASET} \
    --scale=${SCALE} \
    --patch_size=${PATCH_SIZE} \
    --ext=img \
    --epochs=${EPOCHS} \
    --batch_size=${BATCH_SIZE} \
    --lr=${LR} \
    --symunet_pretrain_width=${WIDTH} \
    --symunet_pretrain_enc_blk_nums=${ENC_BLK_NUMS} \
    --symunet_pretrain_dec_blk_nums=${DEC_BLK_NUMS} \
    --symunet_pretrain_restormer_heads=${RESTORMER_HEADS} \
    --symunet_pretrain_middle_blk_num=${MIDDLE_BLK_NUM} \
    --symunet_pretrain_ffn_expansion_factor=${FFN_EXPANSION} \
    --save=${SAVE}
