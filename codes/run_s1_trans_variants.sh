#!/bin/bash
# Test script for S1_Trans variants (NoMAB1, NoMAB2, NoMAB2_Conv)
# Usage: bash run_s1_trans_variants.sh
# Note: Use uv to manage environment

# ===== Training (commented out - uncomment to train) =====
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_nomab2.json
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_nomab1.json
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_nomab2_conv.json
python user/batch_train.py --config user/experiments_config5_02_s1_trans_nomab1_conv.json

# ===== Testing (with pre-trained models) =====

# Variant 1: NoMAB2 (去掉mab2)
# python demo_deploy.py --model CSYMUNET_PRETRAIN_S1_Trans_NoMAB2     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet_base/experiment/s1_trans_nomab2_001_s1_trans_nomab2_w32/model/model_best.pt --dir_out ../experiment/results/s1_trans_nomab2_46W32/x4
# python calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet_base/experiment/results/s1_trans_nomab2_46W32/x4 | tail -n 1 >> results.txt

# # Variant 2: NoMAB1 (去掉mab1)
# python demo_deploy.py --model CSYMUNET_PRETRAIN_S1_Trans_NoMAB1     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet_base/experiment/s1_trans_nomab1_001_s1_trans_nomab1_w32/model/model_best.pt --dir_out ../experiment/results/s1_trans_nomab1_46W32/x4
# python calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet_base/experiment/results/s1_trans_nomab1_46W32/x4 | tail -n 1 >> results.txt

# # Variant 3: NoMAB2_Conv (去掉mab2且用卷积替换MA)
# python demo_deploy.py --model CSYMUNET_PRETRAIN_S1_Trans_NoMAB2_Conv     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet_base/experiment/s1_trans_nomab2_conv_001_s1_trans_nomab2_conv_w32/model/model_best.pt --dir_out ../experiment/results/s1_trans_nomab2_conv_46W32/x4
# python calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet_base/experiment/results/s1_trans_nomab2_conv_46W32/x4 | tail -n 1 >> results.txt

# Variant 4: NoMAB1_Conv (去掉mab1且用卷积替换MA)
python demo_deploy.py --model CSYMUNET_PRETRAIN_S1_Trans_NoMAB1_Conv     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet_base/experiment/s1_trans_nomab1_conv_001_s1_trans_nomab1_conv_w32/model/model_best.pt --dir_out ../experiment/results/s1_trans_nomab1_conv_46W32/x4
python calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet_base/experiment/results/s1_trans_nomab1_conv_46W32/x4 | tail -n 1 >> results.txt

# ===== S1_Trans_Strip Variant (新方案: 使用StripModule) =====

# Training
python user/batch_train.py --config user/experiments_config5_02_s1_trans_strip.json

# Testing (with pre-trained model)
python demo_deploy.py --model CSYMUNET_PRETRAIN_S1_Trans_Strip     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet_base/experiment/s1_trans_strip_001_s1_trans_strip_w32/model/model_best.pt --dir_out ../experiment/results/s1_trans_strip_46W32/x4
python calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet_base/experiment/results/s1_trans_strip_46W32/x4 | tail -n 1 >> results.txt