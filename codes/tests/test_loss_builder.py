import sys
import pytest
import torch.nn as nn
from pathlib import Path

# Add codes to path
sys.path.append(str(Path(__file__).parent.parent))

from loss.builder import LossBuilder, CompositeLoss
from utils.registry import LOSS_REGISTRY

# Register a mock loss for testing
@LOSS_REGISTRY.register("MockLoss")
class MockLoss(nn.Module):
    def forward(self, x, y):
        return 0

def test_build_l1():
    loss = LossBuilder.build("1*L1")
    assert isinstance(loss, CompositeLoss)
    assert len(loss.losses) == 1
    assert isinstance(loss.losses[0], nn.L1Loss)
    assert loss.weights == [1.0]

def test_build_mse():
    loss = LossBuilder.build("1*MSE")
    assert isinstance(loss, CompositeLoss)
    assert isinstance(loss.losses[0], nn.MSELoss)

def test_build_composite():
    # Mock lookup
    loss = LossBuilder.build("1*L1+0.05*MockLoss")
    assert len(loss.losses) == 2
    assert isinstance(loss.losses[0], nn.L1Loss)
    assert isinstance(loss.losses[1], MockLoss)
    assert loss.weights == [1.0, 0.05]

def test_build_default():
    loss = LossBuilder.build("")
    assert isinstance(loss.losses[0], nn.L1Loss)

if __name__ == "__main__":
    sys.argv = [sys.argv[0]]
    sys.exit(pytest.main(["-v", __file__]))
