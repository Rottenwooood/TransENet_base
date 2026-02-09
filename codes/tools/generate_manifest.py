import yaml
import json
import itertools
import argparse
import copy
from pathlib import Path
from datetime import datetime

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_combinations(search_space):
    """
    Generates all combinations of parameters from a search space dictionary.
    Keys with list values are treated as axes for grid search.
    """
    if not search_space:
        return [{}]
        
    keys = list(search_space.keys())
    values = []
    
    for k in keys:
        v = search_space[k]
        if isinstance(v, list):
            values.append(v)
        else:
            values.append([v])
            
    combinations = []
    for p in itertools.product(*values):
        combinations.append(dict(zip(keys, p)))
        
    return combinations

def compile_manifest(config_path, output_path=None):
    """
    Compiles a YAML config (potentially with search spaces) into a JSON manifest.
    """
    config_def = load_yaml(config_path)
    
    # 1. Load Base Config
    base_config_path = config_def.get('base_config', 'configs/base.yaml')
    # Resolve relative to the config file provided
    base_path_resolved = Path(config_path).parent / base_config_path
    if not base_path_resolved.exists():
        # Try relative to project root
        base_path_resolved = Path(base_config_path)
        
    base_config = load_yaml(base_path_resolved)
    
    # 2. Get Group ID and Overrides
    group_id = config_def.get('group_id', 'experiment')
    fixed_overrides = config_def.get('fixed_overrides', {})
    search_space = config_def.get('search_space', {})
    
    # 3. Generate Combinations
    combinations = generate_combinations(search_space)
    
    experiments = []
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, combo in enumerate(combinations):
        # Start with base
        exp_config = copy.deepcopy(base_config)
        
        # Apply fixed overrides
        exp_config.update(fixed_overrides)
        
        # Apply search combo
        exp_config.update(combo)
        
        # Generate ID
        exp_id = f"{group_id}_{i+1:03d}"
        
        # Special handling for list parameters that need to be strings for argparse 
        # (Legacy compatibility for lists like [2,2,2])
        for k, v in exp_config.items():
            if isinstance(v, list) and k in ['symunet_enc_blk_nums', 'symunet_dec_blk_nums', 
                                           'symunet_restormer_heads', 'symunet_pretrain_enc_blk_nums',
                                           'symunet_pretrain_dec_blk_nums', 'symunet_pretrain_restormer_heads']:
                 exp_config[k] = ",".join(map(str, v))
            elif isinstance(v, list):
                 # Other lists might need specific handling or just JSON dumping if the runner supports it
                 pass

        experiments.append({
            "id": exp_id,
            "args": exp_config
        })
        
    manifest = {
        "timestamp": current_time,
        "group_id": group_id,
        "experiments": experiments
    }
    
    # 4. Save Manifest
    if output_path:
        out_file = Path(output_path)
    else:
        out_file = Path("experiments_manifest.json")
        
    with open(out_file, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Generated manifest with {len(experiments)} experiments at {out_file}")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument("--output", help="Path to output JSON manifest", default="experiments_manifest.json")
    args = parser.parse_args()
    
    compile_manifest(args.config, args.output)
