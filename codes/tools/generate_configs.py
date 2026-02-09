#!/usr/bin/env python3
"""
Config Generator for SymUNet Experiments
Generates an explicit JSON manifest for batch training.
Resolves "Li's Problem" by handling all complex parameter logic offline.
"""

import json
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple

def create_base_config() -> Dict[str, Any]:
    return {
        "model": "SYMUNET_PRETRAIN",
        "dataset": "UCMerced",
        "scale": 4,
        "epochs": 300,
        "batch_size": 4,
        "optimizer": "ADAMW",
        "scheduler": "cosine",
        "lr": 2e-4,
        "loss": "1*L1+0.005*StableFFT",
        "save_every_n_steps": 50,
        "use_wandb": True,
        "wandb_project": "SymUNet-Batch"
    }

def generate_experiments() -> List[Dict[str, Any]]:
    experiments = []
    base_config = create_base_config()
    
    # --- Definition of Search Space ---
    
    # 1. Standard Hyperparameters
    widths = [32, 48, 64]
    
    # 2. Paired Architecture Parameters (Encoder - Decoder)
    # Explicitly define valid pairs here. No more runtime guessing.
    arch_pairs = [
        # Small
        {"enc": "2,2,2", "dec": "2,2,2", "heads": "1,2,4", "mid": 8},
        # Medium
        {"enc": "4,4,4", "dec": "4,4,4", "heads": "1,2,4", "mid": 8},
        # Large
        {"enc": "4,6,6", "dec": "6,6,4", "heads": "1,2,8", "mid": 16},
    ]

    exp_id = 1
    
    # Generate Cartesian Product
    for width, arch in itertools.product(widths, arch_pairs):
        config = base_config.copy()
        
        # Apply specific params
        config["symunet_pretrain_width"] = width
        config["symunet_pretrain_enc_blk_nums"] = arch["enc"]
        config["symunet_pretrain_dec_blk_nums"] = arch["dec"]
        config["symunet_pretrain_restormer_heads"] = arch["heads"]
        config["symunet_pretrain_restormer_middle_heads"] = arch["mid"]
        
        # Generate Name
        config["save"] = f"symunet_w{width}_enc{arch['enc'].replace(',','')}"
        if config["use_wandb"]:
            config["wandb_name"] = config["save"]

        experiments.append({
            "id": f"exp_{exp_id:03d}",
            "args": config
        })
        exp_id += 1

    return experiments

def save_manifest(experiments: List[Dict[str, Any]], filename: str = "experiments_manifest.json"):
    output_path = Path(filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated {len(experiments)} experiments in {output_path.absolute()}")

if __name__ == "__main__":
    experiments = generate_experiments()
    save_manifest(experiments)
