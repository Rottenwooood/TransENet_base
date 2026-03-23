#!/bin/bash
# Training script for symunet_pretrain_strip_mhla (MHLA替换NeighborhoodAttention)

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# Training
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_strip_mhla.json

# Inference and Evaluation
python demo_deploy.py --model symunet_pretrain_strip_mhla \
    --symunet_pretrain_width 32 \
    --symunet_pretrain_enc_blk_nums 4,6 \
    --symunet_pretrain_dec_blk_nums 6,4 \
    --dataset UCMerced \
    --scale 4 \
    --pre_train ../experiment/s1_trans_strip_mhla_001_s1_trans_strip_mhla_w32/model/model_best.pt \
    --dir_out ../experiment/results/s1_trans_strip_mhla_46W32/x4

# Calculate PSNR/SSIM
python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_strip_mhla_46W32/x4 | tail -n 1 >> results.txt
