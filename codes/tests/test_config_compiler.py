import sys
import json
import yaml
import pytest
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.generate_manifest import compile_manifest

def test_compile_manifest(tmp_path):
    # Create a dummy base config
    base_config = {"epochs": 100, "lr": 0.001}
    base_yaml = tmp_path / "base.yaml"
    with open(base_yaml, 'w') as f:
        yaml.dump(base_config, f)
        
    # Create a dummy experiment config
    exp_config = {
        "base_config": str(base_yaml),
        "group_id": "test_group",
        "fixed_overrides": {"epochs": 50},
        "search_space": {
            "lr": [0.001, 0.002],
            "batch_size": [16, 32]
        }
    }
    exp_yaml = tmp_path / "exp.yaml"
    with open(exp_yaml, 'w') as f:
        yaml.dump(exp_config, f)
        
    # Compile
    manifest_path = tmp_path / "manifest.json"
    compile_manifest(str(exp_yaml), str(manifest_path))
    
    # Verify
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    assert manifest['group_id'] == "test_group"
    assert len(manifest['experiments']) == 4 # 2x2 grid
    
    # Check override
    assert manifest['experiments'][0]['args']['epochs'] == 50
    
    # Check search space coverage
    lrs = set(e['args']['lr'] for e in manifest['experiments'])
    assert lrs == {0.001, 0.002}
    
    bs = set(e['args']['batch_size'] for e in manifest['experiments'])
    assert bs == {16, 32}

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    sys.exit(pytest.main(["-v", __file__]))
