import torch
import numpy as np

# Utility function for CUDA
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

def to_tensor(x, device):
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32, device=device)
    return x.to(device=device, dtype=torch.float32) if device else x
