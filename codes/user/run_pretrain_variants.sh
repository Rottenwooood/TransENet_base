#!/bin/bash
# Training script for SymUNet Pretrain variants
# Usage: bash run_pretrain_variants.sh

python user/batch_train.py --config user/experiments_config1_cg.json
python user/batch_train.py --config user/experiments_config2_lpu.json
python user/batch_train.py --config user/experiments_config3_lk.json
python user/batch_train.py --config user/experiments_config4_dwconv.json
