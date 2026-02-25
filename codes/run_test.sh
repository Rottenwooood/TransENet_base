#!/bin/bash
# python demo_deploy.py --model CSYMUNET_PRETRAIN_CG     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/cg_pretrain_001_cg_pretrain_gce2_w32/model/model_best.pt --dir_out ../experiment/results/cg46W32/x4

# python demo_deploy.py --model CSYMUNET_PRETRAIN_LPU     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/lpu_pretrain_001_lpu_pretrain_w32/model/model_best.pt --dir_out ../experiment/results/lpu46W32/x4

# python demo_deploy.py --model CSYMUNET_PRETRAIN_DWCONV     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/dwconv_pretrain_001_dwconv_pretrain_w32/model/model_best.pt --dir_out ../experiment/results/dwconv46W32/x4

python demo_deploy.py --model CSYMUNET_PRETRAIN_LK     --symunet_pretrain_width 32     --symunet_pretrain_enc_blk_nums 4,6     --symunet_pretrain_dec_blk_nums 6,4     --dataset UCMerced     --scale 4 --pre_train /root/autodl-tmp/TransENet/experiment/lk_pretrain_001_lk_pretrain_k7_w32/model/model_best.pt --dir_out ../experiment/results/lk46W32/x4

# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/cg46W32/x4 | tail -n 1 >> results.txt
# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/lpu46W32/x4 | tail -n 1 >> results.txt
# python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/dwconv46W32/x4 | tail -n 1 >> results.txt
python metric_scripts/calculate_PSNR_SSIM.py --folder_Gen /root/autodl-tmp/TransENet/experiment/results/lk46W32/x4 | tail -n 1 >> results.txt