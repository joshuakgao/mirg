import torch
from typing import Literal
from utils.logger import logger


def device_selector(device: Literal["auto", "cpu", "mps", "cuda"] = "auto", label=""):
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"

    logger.log(f"Using device {device} for {label}")
    return device
