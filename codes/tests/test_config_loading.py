import sys
import os
import json
import pytest
from argparse import Namespace
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

# Patch sys.argv BEFORE importing modules that use argparse at module level
# This prevents option.py from failing when parsing pytest arguments
_original_argv = sys.argv
sys.argv = [sys.argv[0]]

try:
    from tools.generate_configs import generate_experiments, save_manifest
    from user.batch_train import ExperimentRunner
    from train_enhanced import update_args_with_wandb
finally:
    # Restore argv just in case, though for tests it might not matter much
    # provided we don't rely on it later in a way that conflicts
    sys.argv = _original_argv

# Mock WandB
class MockWandB:
    def __init__(self, config):
        self.config = config
        self.run = None
    
    def init(self, **kwargs):
        pass

def test_manifest_generation(tmp_path):
    """Test that experiments are generated correctly"""
    experiments = generate_experiments()
    assert len(experiments) > 0
    assert "args" in experiments[0]
    assert "id" in experiments[0]
    
    # Check specific keys
    config = experiments[0]["args"]
    assert "symunet_pretrain_width" in config
    assert "symunet_pretrain_enc_blk_nums" in config

    # Test saving
    manifest_path = tmp_path / "test_manifest.json"
    save_manifest(experiments, str(manifest_path))
    assert manifest_path.exists()

def test_experiment_runner(tmp_path):
    """Test loading manifest in Runner"""
    # Create dummy manifest
    manifest = [{"id": "test_01", "args": {"lr": 0.001}}]
    manifest_path = tmp_path / "dummy.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f)
        
    runner = ExperimentRunner(str(manifest_path))
    assert len(runner.experiments) == 1
    assert runner.experiments[0]["id"] == "test_01"
    
    # Test command building
    cmd = runner.build_command({"lr": 0.001, "list_arg": [1,2]})
    assert "python" in cmd
    assert "--lr 0.001" in cmd
    assert "--list_arg 1,2" in cmd

def test_wandb_override(monkeypatch):
    """Test that wandb config overrides args"""
    args = Namespace(lr=0.001, use_wandb=True, wandb_project="test")
    
    # Mock wandb module
    mock_config = {"lr": 0.01} # Override lr
    mock_wandb = MockWandB(mock_config)
    
    monkeypatch.setattr("train_enhanced.wandb", mock_wandb)
    
    updated_args = update_args_with_wandb(args)
    assert updated_args.lr == 0.01

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
