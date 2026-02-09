import sys
import pytest
import os
import shutil
import cv2
import yaml
import tempfile
import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add codes to path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))
PROJECT_ROOT = CODE_DIR.parent

# Import main function from train.py
# We can import it as a module
import tools.train as train_tool

@pytest.fixture
def temp_workspace():
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    
    # Create valid dataset structure: UCMerced-train/UCMerced-dataset/{train,val}/...
    # Actually, default dataset is UCMerced, data_train path is hardcoded in base.yaml default
    # but we will override it in our config.
    
    # Structure:
    # temp_dir/train/class1/img1.tif
    # temp_dir/val/class1/img1.tif
    
    # UCMercedDataset expects HR_x4 and LR_x4 subdirectories
    for mode in ['train', 'val']:
        base_dir = os.path.join(temp_dir, mode)
        hr_dir = os.path.join(base_dir, 'HR_x4')
        lr_dir = os.path.join(base_dir, 'LR_x4')
        os.makedirs(hr_dir, exist_ok=True)
        os.makedirs(lr_dir, exist_ok=True)
        
        # Create dummy images
        for i in range(2):
            img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
            # HR image (128x128)
            cv2.imwrite(os.path.join(hr_dir, f'img_{i}.tif'), img)
            # LR image (32x32 for scale 4)
            lr_img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(lr_dir, f'img_{i}.tif'), lr_img)
            
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

def test_e2e_training_loop(temp_workspace):
    """
    Run a minimal training loop (1 epoch, minimal batches) to ensure no crashes.
    """
    output_dir = os.path.join(temp_workspace, 'output')
    config_path = os.path.join(temp_workspace, 'config.yaml')
    
    # Minimal Config
    config = {
        'model': 'SymUNet_Pretrain',
        'dataset': 'UCMerced', # Dataset class name
        'data_train': os.path.join(temp_workspace, 'train'),
        'data_val': os.path.join(temp_workspace, 'val'),
        'dir_data': temp_workspace, # Some datasets look here
        'save': 'e2e_test', # Explicit save name
        'dir_out': output_dir,
        'ext': 'img', # Load images directly - crucial for finding .tif files without .npy conversion
        'scale': 4,
        'epochs': 1,
        'batch_size': 2,
        'patch_size': 32, # Small patch for speed
        'n_threads': 1,
        'use_wandb': False,
        'print_every': 1,
        'test_every': 1,
        'save_every_n_steps': 100,
        'cpu': True, # Use CPU for test environment
        'symunet_pretrain_width': 16, # Small model
        'symunet_pretrain_enc_blk_nums': [1,1],
        'symunet_pretrain_dec_blk_nums': [1,1],
        'symunet_pretrain_restormer_heads': [1,2],
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run in subprocess to ensure clean environment (sys.argv pollution etc)
    import subprocess
    cmd = [sys.executable, 'codes/tools/train.py', '--config', config_path]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    
    print(result.stdout)
    print(result.stderr)
    
    assert result.returncode == 0
    
    # Check if outputs generated
    # checkpoint creates subdirectory based on 'save' arg
    exp_dir = os.path.join(output_dir, 'e2e_test')
    assert os.path.exists(exp_dir)
    # Check logs
    assert os.path.exists(os.path.join(exp_dir, 'log.txt'))

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
