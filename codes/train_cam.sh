#!/bin/bash
# Training script for symunet_pretrain_cam (Channel Attention Middle)

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
# cd /home/c6h4o2/dev/TransENet_base/codes

# # Training
# # python user/batch_train.py --config user/experiments_config5_02_s1_trans_strip_c.json
# # python demo_deploy.py --model CSYMUNET_PRETRAIN_S1_Trans_Strip     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet_base/experiment/s1_trans_strip_c_001_s1_trans_strip_c_w32/model/model_best.pt --dir_out ../experiment/results/s1_trans_strip_c_46W32/x4
# # python calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet_base/experiment/results/s1_trans_strip_c_46W32/x4 | tail -n 1 >> results.txt

# python user/batch_train.py --config user/experiments_config5_02_s1_trans_cam.json
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_lsconv.json
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_pki.json
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_strip.json
# python user/batch_train.py --config user/experiments_config5_02_s1_trans_lsconv_strip.json
python user/batch_train.py --config user/experiments_config5_02_s1_trans_strip_cam.json
python user/batch_train.py --config user/experiments_config5_02_s1_trans_nomab1_middle.json

# Inference and Evaluation
# python demo_deploy.py --model symunet_pretrain_cam \
#     --symunet_pretrain_width 32 \
#     --symunet_pretrain_enc_blk_nums 4,6 \
#     --symunet_pretrain_dec_blk_nums 6,4 \
#     --dataset UCMerced \
#     --scale 4 \
#     --pre_train ../experiment/s1_trans_cam_001_s1_trans_cam_w32/model/model_best.pt \
#     --dir_out ../experiment/results/s1_trans_cam_v2_46W32/x4
# python demo_deploy.py --model symunet_pretrain_lsconv \
#     --symunet_pretrain_width 32 \
#     --symunet_pretrain_enc_blk_nums 4,6 \
#     --symunet_pretrain_dec_blk_nums 6,4 \
#     --dataset UCMerced \
#     --scale 4 \
#     --pre_train ../experiment/s1_trans_lsconv_001_s1_trans_lsconv_w32/model/model_best.pt \
#     --dir_out ../experiment/results/s1_trans_lsconv_v2_46W32/x4
# python demo_deploy.py --model symunet_pretrain_pki \
#     --symunet_pretrain_width 32 \
#     --symunet_pretrain_enc_blk_nums 4,6 \
#     --symunet_pretrain_dec_blk_nums 6,4 \
#     --dataset UCMerced \
#     --scale 4 \
#     --pre_train ../experiment/s1_trans_pki_001_s1_trans_pki_w32/model/model_best.pt \
#     --dir_out ../experiment/results/s1_trans_pki_v2_46W32/x4
# python demo_deploy.py --model symunet_pretrain_strip \
#     --symunet_pretrain_width 32 \
#     --symunet_pretrain_enc_blk_nums 4,6 \
#     --symunet_pretrain_dec_blk_nums 6,4 \
#     --dataset UCMerced \
#     --scale 3 \
#     --pre_train ../experiment/s1_trans_strip_001_s1_trans_strip_w32/model/model_best.pt \
#     --dir_out ../experiment/results/s1_trans_strip_v2_46W32/x4

python demo_deploy.py --model symunet_pretrain_strip_cam \
    --symunet_pretrain_width 32 \
    --symunet_pretrain_enc_blk_nums 4,6 \
    --symunet_pretrain_dec_blk_nums 6,4 \
    --dataset UCMerced \
    --scale 4 \
    --pre_train ../experiment/s1_trans_strip_cam_001_s1_trans_strip_cam_w32/model/model_best.pt \
    --dir_out ../experiment/results/s1_trans_strip_cam_46W32/x4

python demo_deploy.py --model csymunet_pretrain_s1_trans_nomab1_middle \
    --symunet_pretrain_width 32 \
    --symunet_pretrain_enc_blk_nums 4,6 \
    --symunet_pretrain_dec_blk_nums 6,4 \
    --dataset UCMerced \
    --scale 4 \
    --pre_train ../experiment/s1_trans_nomab1_middle_001_s1_trans_nomab1_middle_w32/model/model_best.pt \
    --dir_out ../experiment/results/s1_trans_nomab1_middle_46W32/x4

# python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_cam_v2_46W32/x4 | tail -n 1 >> results.txt
# python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_lsconv_v2_46W32/x4 | tail -n 1 >> results.txt
# python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_pki_v2_46W32/x4 | tail -n 1 >> results.txt
# python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_strip_v2_46W32/x4 | tail -n 1 >> results.txt

python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_strip_cam_46W32/x4 | tail -n 1 >> results.txt
python calculate_PSNR_SSIM.py --folder_Gen ../experiment/results/s1_trans_nomab1_middle_46W32/x4 | tail -n 1 >> results.txt
