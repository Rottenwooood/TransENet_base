#!/bin/bash

# 参考 symunet.yml 配置修改的训练命令
python demo_train.py \
    --model SYMUNET_PRETRAIN \
    --dataset=UCMerced \
    --scale 4 \
    --ext=img \
    --symunet_pretrain_width 48 \
    --symunet_pretrain_enc_blk_nums 4,6,6 \
    --symunet_pretrain_dec_blk_nums 6,6,4 \
    --symunet_pretrain_restormer_heads 1,2,4 \
    --symunet_pretrain_restormer_middle_heads 8 \
    --symunet_pretrain_ffn_expansion_factor 2.66 \
    --symunet_pretrain_bias False \
    --symunet_pretrain_layer_norm_type WithBias \
    --epochs 500 \
    --batch_size 4 \
    --lr 1e-3 \
    --loss "1*L1+0.1*FFT" \
    --data_train=/root/autodl-tmp/TransENet/datasets/UCMerced-train/UCMerced-dataset/train \
    --data_val=/root/autodl-tmp/TransENet/datasets/UCMerced-train/UCMerced-dataset/val \
    --save symunet_pretrain_x4_w48
