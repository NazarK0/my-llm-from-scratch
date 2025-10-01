import os
import torch

def get_cpu_cores() -> int:
    cpu_count = os.cpu_count()

    if cpu_count is not None:
        return cpu_count
    else:
        return 1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
