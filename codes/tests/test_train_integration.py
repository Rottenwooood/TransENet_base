import sys
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.registry import ARCH_REGISTRY, LOSS_REGISTRY

# Mock a model
@ARCH_REGISTRY.register("MockModel")
class MockModel(nn.Module):
    def __init__(self, args):
        super(MockModel, self).__init__()
        self.args = args
    def forward(self, x):
        return x

# Mock a loss
@LOSS_REGISTRY.register("MockLoss")
class MockLoss(nn.Module):
    def forward(self, x, y):
        return torch.tensor(0.0)

def test_model_init_registry():
    """Test that Model class uses ARCH_REGISTRY"""
    from model import Model
    
    args = MagicMock()
    args.model = "MockModel"
    args.scale = [4]
    args.self_ensemble = False
    args.chop = False
    args.precision = "single"
    args.cpu = True
    args.n_GPUs = 1
    args.save_models = False
    args.pre_train = "."
    args.resume = 0
    args.print_model = False
    
    ckp = MagicMock()
    ckp.dir = "."
    
    model = Model(args, ckp)
    # Check if the underlying model is our MockModel
    assert isinstance(model.model, MockModel)

def test_loss_init_registry():
    """Test that Loss class uses LOSS_REGISTRY"""
    from loss import Loss
    
    args = MagicMock()
    args.loss = "1*MockLoss" # Should use registry
    args.n_GPUs = 1
    args.cpu = True
    args.precision = "single"
    args.resume = 0
    args.rgb_range = 1
    
    ckp = MagicMock()
    
    loss = Loss(args, ckp)
    # Check if MockLoss is in the loss module list
    assert isinstance(loss.loss_module[0], MockLoss)

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    sys.exit(pytest.main(["-v", __file__]))
