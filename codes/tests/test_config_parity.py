import sys
import unittest
import yaml
import argparse
from unittest.mock import patch
from pathlib import Path

# Add codes to path
CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CODE_DIR))
PROJECT_ROOT = CODE_DIR.parent

from utils.config import ConfigToArgsAdapter

class TestConfigParity(unittest.TestCase):
    def test_default_values_match(self):
        # 1. Get Legacy Defaults from option.py
        # We assume option.py uses argparse.ArgumentParser
        # We need to import it ensuring sys.argv is empty so it uses defaults
        with patch.object(sys, 'argv', ['option.py']):
            # We might need to reload if it was already imported
            if 'option' in sys.modules:
                del sys.modules['option']
            import option
            legacy_args = option.args

        # 2. Get New Defaults from base.yaml
        base_config_path = PROJECT_ROOT / 'configs' / 'base.yaml'
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        new_args = ConfigToArgsAdapter.dict_to_namespace(base_config)
        
        # 3. Compare Critical Hyperparameters
        # List of keys to check. We verify that new_args has the same value as legacy_args
        # We only check keys present in base.yaml (as we might omit some legacy unused ones)
        
        # Critical training params
        params_to_check = [
            'epochs', 'batch_size', 'split_batch',
            'lr', 'lr_decay', 'decay_type', 'gamma',
            'optimizer', 'momentum', 'beta1', 'beta2', 'epsilon', 'weight_decay',
            'loss', 'skip_threshold',
            'n_threads', 'patch_size', 'rgb_range', 'n_colors', 'scale'
        ]
        
        mismatches = []
        for key in params_to_check:
            if not hasattr(legacy_args, key):
                print(f"Warning: Legacy args missing key {key}")
                continue
                
            legacy_val = getattr(legacy_args, key)
            
            # YAML loading might be strict on types, argparse often gives strings or lists
            # We handle known necessary conversions
            
            if hasattr(new_args, key):
                new_val = getattr(new_args, key)
                

                
                # Coerce legacy value to new value type if possible
                # This handles cases where argparse default is '1e6' (str) but YAML gives 1e6 (float)
                if isinstance(new_val, (int, float)) and isinstance(legacy_val, str):
                    try:
                        legacy_val = float(legacy_val) # legacy defaults like '1e6'
                        if isinstance(new_val, int):
                           legacy_val = int(legacy_val)
                    except ValueError:
                        pass

                # Normalization for float comparison
                if isinstance(legacy_val, float) and isinstance(new_val, float):
                    if abs(legacy_val - new_val) > 1e-9:
                        mismatches.append(f"{key}: Legacy({legacy_val}) != New({new_val})")
                
                # Normalization for lists (option.py might produce lists via explicit parsing or split)
                elif key == 'scale':
                    # option.py produces a list of ints
                    legacy_scale = legacy_val
                    new_scale = new_val if isinstance(new_val, list) else [new_val]
                    if legacy_scale != new_scale:
                        mismatches.append(f"{key}: Legacy({legacy_scale}) != New({new_scale})")
                
                elif legacy_val != new_val:
                    mismatches.append(f"{key}: Legacy({legacy_val}) != New({new_val})")
            else:
                 mismatches.append(f"{key}: Missing in New Config")
                 
        if mismatches:
            self.fail("Configuration Parity Check Failed:\n" + "\n".join(mismatches))

if __name__ == '__main__':
    unittest.main()
