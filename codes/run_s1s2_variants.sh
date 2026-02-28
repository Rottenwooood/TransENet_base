#!/bin/bash
# Training script for SymUNet S1/S2 variants
# Usage: bash run_s1s2_variants.sh

# Run each experiment individually
python user/quick_batch.py --config user/experiments_config5_01_base_mscs.json
python user/quick_batch.py --config user/experiments_config5_02_s1_trans.json
python user/quick_batch.py --config user/experiments_config5_03_s1_singledilated.json
python user/quick_batch.py --config user/experiments_config5_04_s1_mrdilated.json
python user/quick_batch.py --config user/experiments_config5_05_s1_singledense.json
python user/quick_batch.py --config user/experiments_config5_06_s1_mrdense.json
python user/quick_batch.py --config user/experiments_config5_07_s2_trans.json
python user/quick_batch.py --config user/experiments_config5_08_s2_singledilated.json
python user/quick_batch.py --config user/experiments_config5_09_s2_mrdilated.json
python user/quick_batch.py --config user/experiments_config5_10_s2_singledense.json
python user/quick_batch.py --config user/experiments_config5_11_s2_mrdense.json
