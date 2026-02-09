import re
import torch
import torch.nn as nn
from utils.registry import LOSS_REGISTRY

class CompositeLoss(nn.Module):
    def __init__(self, losses, weights):
        super(CompositeLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        
    def forward(self, *args, **kwargs):
        total_loss = 0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(*args, **kwargs)
        return total_loss

class LossBuilder:
    @staticmethod
    def build(config_str):
        """
        Builds a CompositeLoss from a string like "1*L1+0.05*StableFFT".
        
        Args:
            config_str (str): Loss configuration string.
            
        Returns:
            CompositeLoss: The combined loss function.
        """
        # Default to 1*L1 if not specified or empty
        if not config_str:
            config_str = "1*L1"
            
        components = config_str.split('+')
        losses = []
        weights = []
        
        for component in components:
            component = component.strip()
            if not component:
                continue
                
            if '*' in component:
                try:
                    weight_str, loss_name = component.split('*')
                    weight = float(weight_str)
                except ValueError:
                     # Handle case where * might be part of name or multiple *?
                     # For now assume format is STRICTLY weight*Name
                     raise ValueError(f"Invalid loss component format: {component}. Expected 'weight*LossName'")
            else:
                weight = 1.0
                loss_name = component
                
            # Handle Built-in PyTorch Losses mapped to registry
            # We can register them lazily or just handle specific ones
            if loss_name == "L1":
                loss_fn = nn.L1Loss()
            elif loss_name == "MSE":
                loss_fn = nn.MSELoss()
            else:
                # Retrieve from Registry
                loss_cls = LOSS_REGISTRY.get(loss_name)
                if loss_cls is None:
                    raise KeyError(f"Loss '{loss_name}' not found in registry or standard list.")
                loss_fn = loss_cls() # Instantiate with default args for now
            
            losses.append(loss_fn)
            weights.append(weight)
            
        return CompositeLoss(losses, weights)
