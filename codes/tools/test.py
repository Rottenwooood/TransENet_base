#!/usr/bin/env python3
import argparse
import os
import sys
import glob
import cv2
import torch
import numpy as np
import yaml
from pathlib import Path

# Add codes to path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))
PROJECT_ROOT = CODE_DIR.parent

from utils.registry import ARCH_REGISTRY
from utils.metrics import calculate_psnr, calculate_ssim
from utils.config import ConfigToArgsAdapter
import utils.common as utils
import template

def evaluate(args):
    device = torch.device('cpu' if args.cpu else 'cuda')
    
    print(f"Loading model: {args.model}")
    try:
        model = ARCH_REGISTRY.build(args.model, args=args).to(device)
    except Exception as e:
        print(f"Failed to build model {args.model} from registry: {e}")
        return

    # Load Checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        try:
            state_dict = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return
    else:
        print("Warning: No checkpoint provided. Evaluating random initialization.")
            
    model.eval()
    
    # Prepare Data
    if not hasattr(args, 'input_dir') or not args.input_dir:
        print("Error: Input directory not specified.")
        return
        
    ext = getattr(args, 'ext', '.png')
    img_list = sorted(glob.glob(os.path.join(args.input_dir, '*' + ext)))
    if not img_list:
        print(f"No images found in {args.input_dir} with extension {ext}")
        return
        
    print(f"Found {len(img_list)} images.")
    
    if hasattr(args, 'output_dir') and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    # Get scale for border cropping
    scale = args.scale[0] if isinstance(args.scale, list) else args.scale
    
    for img_path in img_list:
        img_name = os.path.basename(img_path)
        
        # Read LR
        img_lr = cv2.imread(img_path)
        if img_lr is None:
            continue
        img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        
        # Read GT if available
        gt_dir = getattr(args, 'gt_dir', None)
        gt_path = os.path.join(gt_dir, img_name) if gt_dir else None
        img_gt = None
        if gt_path and os.path.exists(gt_path):
            img_gt = cv2.imread(gt_path)
            img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
            
        # Pre-process
        rgb_range = getattr(args, 'rgb_range', 1)
        img_lr_tensor = utils.np2Tensor([img_lr], rgb_range)[0].unsqueeze(0).to(device)
        
        # Inference
        try:
            with torch.no_grad():
                # Handling tiled inference if needed
                if getattr(args, 'test_tile', False):
                    # Placeholder for tiling logic
                    output = model(img_lr_tensor)
                else:
                    output = model(img_lr_tensor)
        except Exception as e:
            print(f"Error checking {img_name}: {e}")
            continue
                 
        # Post-process
        output_tensor = utils.quantize(output, rgb_range)
        output_np = utils.torch_to_np(output_tensor) # [H, W, C]
        
        # Save
        if hasattr(args, 'output_dir') and args.output_dir:
            save_path = os.path.join(args.output_dir, img_name)
            cv2.imwrite(save_path, cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        # Calculate Metrics
        if img_gt is not None:
             out_bgr_uint8 = cv2.cvtColor((output_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
             gt_bgr_uint8 = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
             
             psnr = calculate_psnr(out_bgr_uint8, gt_bgr_uint8, crop_border=scale)
             ssim = calculate_ssim(out_bgr_uint8, gt_bgr_uint8, crop_border=scale)
             
             print(f"{img_name}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
             total_psnr += psnr
             total_ssim += ssim
             count += 1
             
    if count > 0:
        print(f"Average PSNR: {total_psnr/count:.2f}")
        print(f"Average SSIM: {total_ssim/count:.4f}")

def main():
    parser = argparse.ArgumentParser(description='SymUNet Evaluation Tool')
    parser.add_argument('--config', type=str, help='Path to experiment YAML config')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--input_dir', type=str, help='Input LR images')
    parser.add_argument('--output_dir', type=str, help='Output SR images')
    parser.add_argument('--gt_dir', type=str, help='Ground Truth High-Res images')
    parser.add_argument('--model', type=str, help='Model name (override config)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU')
    
    cli_args, unknown = parser.parse_known_args()
    
    # 1. Load Base Config
    base_config_path = PROJECT_ROOT / 'configs' / 'base.yaml'
    if base_config_path.exists():
        config = yaml.safe_load(open(base_config_path, 'r'))
    else:
        config = {}

    # 2. Load User Config
    if cli_args.config:
         with open(cli_args.config, 'r') as f:
             user_config = yaml.safe_load(f)
             config.update(user_config)
             
    # 3. Apply CLI Overrides
    if cli_args.model: config['model'] = cli_args.model
    if cli_args.cpu: config['cpu'] = True
    
    # 4. Convert
    args = ConfigToArgsAdapter.dict_to_namespace(config)
    
    # 5. Apply Runtime args (paths that might not be in config)
    if cli_args.checkpoint: args.checkpoint = cli_args.checkpoint
    if cli_args.input_dir: args.input_dir = cli_args.input_dir
    if cli_args.output_dir: args.output_dir = cli_args.output_dir
    if cli_args.gt_dir: args.gt_dir = cli_args.gt_dir
    
    # 6. Normalize
    if hasattr(args, 'scale') and not isinstance(args.scale, list):
        args.scale = [int(args.scale)]
        
    template.set_template(args)
    
    evaluate(args)

if __name__ == "__main__":
    main()
