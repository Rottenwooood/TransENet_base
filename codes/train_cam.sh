#!/bin/bash
# Training script for symunet_pretrain_cam (Channel Attention Middle)

cd /home/c6h4o2/dev/TransENet_base/codes
python user/batch_train.py --config user/experiments_config5_02_s1_trans_cam.json
