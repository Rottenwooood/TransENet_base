#!/bin/bash
# Training script for symunet_pretrain_pki (InceptionBottleneck)

cd /home/c6h4o2/dev/TransENet_base/codes

# Training
python user/batch_train.py --config user/experiments_config5_02_s1_trans_pki.json

# Inference and Evaluation
python demo_deploy.py --model symunet_pretrain_pki \
    --symunet_pretrain_width 32 \
    --symunet_pretrain_enc_blk_nums 4,6 \
    --symunet_pretrain_dec_blk_nums 6,4 \
    --dataset UCMerced \
    --scale 4 \
    --pre_train ../experiment/s1_trans_pki_001_s1_trans_pki_w32/model/model_best.pt \
    --dir_out ../experiment/results/s1_trans_pki_46W32/x4

python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_pki_46W32/x4 | tail -n 1 >> results.txt
