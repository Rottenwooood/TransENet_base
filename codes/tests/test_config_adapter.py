import sys
import pytest
import argparse
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import ConfigToArgsAdapter

def test_dict_to_namespace_flat():
    config = {"epochs": 100, "lr": 0.001, "model": "TestModel"}
    args = ConfigToArgsAdapter.dict_to_namespace(config)
    
    assert isinstance(args, argparse.Namespace)
    assert args.epochs == 100
    assert args.lr == 0.001
    assert args.model == "TestModel"

def test_dict_to_namespace_nested():
    config = {
        "training": {
            "epochs": 100
        },
        "model": "TestModel"
    }
    args = ConfigToArgsAdapter.dict_to_namespace(config)
    
    assert args.model == "TestModel"
    assert isinstance(args.training, argparse.Namespace)
    assert args.training.epochs == 100

def test_apply_defaults():
    config = {"epochs": 50}
    defaults = {"epochs": 100, "lr": 0.001}
    
    args = ConfigToArgsAdapter.dict_to_namespace(config)
    args = ConfigToArgsAdapter.apply_defaults(args, defaults)
    
    assert args.epochs == 50 # Kept provided value
    assert args.lr == 0.001 # Applied default

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    sys.exit(pytest.main(["-v", __file__]))
