import torch
import random
import numpy as np

def setup_device():
    """Setup computing device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"#  Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("💻 Using CPU")
    return device

def set_random_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"🎲 Random seed set to {seed}")

def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

