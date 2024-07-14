import torch
from typing import Literal


def device_selector(device: Literal["auto", "cpu", "mps", "cuda"] = "auto", label=""):
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    print(f"{label}: Using device {device}")
    return device
