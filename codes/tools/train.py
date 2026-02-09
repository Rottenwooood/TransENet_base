#!/usr/bin/env python3
import sys
import argparse
import yaml
import os
from pathlib import Path

# Add codes to path (to find model, data, utils, template)
# train.py is in codes/tools/
# we want to add codes/ (parent) to sys.path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))
PROJECT_ROOT = CODE_DIR.parent

import data
import model
import utils
import loss
import trainer
import template
from utils.config import ConfigToArgsAdapter

try:
    import wandb
except ImportError:
    wandb = None

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def update_args_with_wandb(args):
    """
    Update args with values from wandb.config (if available).
    This allows hyperparameter sweeps to override command line arguments.
    """
    if getattr(args, 'use_wandb', False) and wandb is not None:
        # Check if we are in a sweep
        if wandb.run is None:
            # If not initialized, try to init (might be part of a sweep)
            wandb.init(project=args.wandb_project, config=args, reinit=True)
        
        config = wandb.config
        print(f"🔄 WandB Config Sync: Overriding args from WandB...")
        
        updated_count = 0
        for key in config.keys():
            if hasattr(args, key) and key != 'wandb_name': 
                old_val = getattr(args, key)
                new_val = config[key]
                if old_val != new_val:
                    setattr(args, key, new_val)
                    print(f"   Parameter '{key}': {old_val} -> {new_val}")
                    updated_count += 1
        
        print(f"✅ WandB Sync Complete. {updated_count} parameters updated.\n")
    return args

def main():
    parser = argparse.ArgumentParser(description='SymUNet Training Tool')
    parser.add_argument('--config', type=str, default=None, help='Path to experiment YAML config')
    
    # We allow overriding specific args via CLI for convenience (optional, but good for quick tests)
    # But strictly speaking, we rely on YAML. 
    # Let's support --config only for now to enforce the pattern.
    
    cli_args, unknown = parser.parse_known_args()
    
    # 1. Load Base Config (Defaults)
    base_config_path = PROJECT_ROOT / 'configs' / 'base.yaml'
    if not base_config_path.exists():
        print(f"Warning: Base config not found at {base_config_path}")
        config = {}
    else:
        config = load_config(base_config_path)

    # 2. Load User Config (Override)
    if cli_args.config:
        user_config_path = Path(cli_args.config)
        if not user_config_path.exists():
            print(f"Error: Config file not found at {user_config_path}")
            sys.exit(1)
        user_config = load_config(user_config_path)
        config.update(user_config)
    elif not base_config_path.exists():
        print("Error: No config provided and no base config found.")
        sys.exit(1)
        
    # 3. Convert to Namespace (Simulating option.py output)
    args = ConfigToArgsAdapter.dict_to_namespace(config)
    
    # 4. Post-process / Normalization (Compatibility Layer)
    # Ensure scale is a list [4] not int 4
    if hasattr(args, 'scale') and not isinstance(args.scale, list):
        args.scale = [int(args.scale)]
        
    # Ensure other lists are actual lists (YAML handles this generally)
    
    # 5. Apply Template (Legacy model-specific defaults)
    template.set_template(args)
    
    # 6. WandB Override (Sweep Support)
    args = update_args_with_wandb(args)
    
    # Print Configuration
    print("🚀 Training Configuration:")
    print(f"Model: {args.model}")
    print(f"Output Dir: {args.dir_out}")
    print(f"Scale: {args.scale}")
    print(f"Seed: {getattr(args, 'seed', 1)}")

    # Set random seed
    if hasattr(args, 'seed'):
        utils.set_random_seed(args.seed)
    else:
        print("⚠️ Warning: No seed found in args, using default seed 1")
        utils.set_random_seed(1)
    
    # Initialize checkpoint
    # We might need to override args.save if we want dynamic naming in v2? 
    # For now, stick to config.
    checkpoint = utils.checkpoint(args)
    
    if checkpoint.ok:
        # Create dataloaders
        dataloaders = data.create_dataloaders(args)

        # Create model
        # Uses ARCH_REGISTRY internally now via model/__init__.py refactor
        sr_model = model.Model(args, checkpoint)

        # Create loss function
        # Uses LOSS_REGISTRY/LossBuilder via loss/__init__.py refactor
        sr_loss = loss.Loss(args, checkpoint) if not args.test_only else None

        # Create trainer
        t = trainer.Trainer(args, dataloaders, sr_model, sr_loss, checkpoint)

        # Training loop
        print("\n🏃 Starting training...")
        
        patience = getattr(args, 'patience', 20)
        no_improve_epochs = 0
        best_psnr = 0.0
        
        while not t.terminate():
            t.train()
            t.test()
            
            # Early Stopping Check
            if hasattr(t.ckp, 'log') and len(t.ckp.log) > 0:
                current_psnr = t.ckp.log[-1, 0].item()
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    
                if no_improve_epochs >= patience:
                    print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
                    break

        print("✅ Training completed!")
        checkpoint.done()
    else:
        print("❌ Checkpoint initialization failed!")

if __name__ == '__main__':
    main()
