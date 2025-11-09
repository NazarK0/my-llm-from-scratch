import torch

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: left {left.shape}, right {right.shape}")
    
    return torch.nn.Parameter(torch.tensor(right))