#!/bin/bash
# Training script for SymUNet Pretrain variants
# Usage: bash run_pretrain_variants.sh

# python user/batch_train.py --config user/experiments_config1_cg_f1.json
# python user/batch_train.py --config user/experiments_config1_cg_f2.json
# python user/batch_train.py --config user/experiments_config1_cg_f3.json
# python user/batch_train.py --config user/experiments_config1_cg_f4.json

# python demo_deploy.py --model CSYMUNET_PRETRAIN_CG_F1     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/cg_pretrain_f1_001_cg_pretrain_f1_gce2_w32/model/model_best.pt --dir_out ../experiment/results/cgfix146W32/x4
# python demo_deploy.py --model CSYMUNET_PRETRAIN_CG_F2     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/cg_pretrain_f2_001_cg_pretrain_f2_gce2_w32/model/model_best.pt --dir_out ../experiment/results/cgfix246W32/x4
# python demo_deploy.py --model CSYMUNET_PRETRAIN_CG_F3     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/cg_pretrain_f3_001_cg_pretrain_f3_gce2_w32/model/model_best.pt --dir_out ../experiment/results/cgfix346W32/x4
# python demo_deploy.py --model CSYMUNET_PRETRAIN_CG_F4     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/cg_pretrain_f4_001_cg_pretrain_f4_gce2_w32/model/model_best.pt --dir_out ../experiment/results/cgfix446W32/x4

# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/cgfix146W32/x4 | tail -n 1 >> results.txt
# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/cgfix246W32/x4 | tail -n 1 >> results.txt
# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/cgfix346W32/x4 | tail -n 1 >> results.txt
# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/cgfix446W32/x4 | tail -n 1 >> results.txt
