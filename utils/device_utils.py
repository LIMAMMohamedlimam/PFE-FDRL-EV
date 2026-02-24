"""
device_utils.py
===============
Shared device selection utility.
Every PyTorch-based agent imports `get_device()` at construction time,
ensuring they automatically use a GPU when available.
"""
import torch


def get_device() -> torch.device:
    """
    Return the best available compute device.

    Priority:
      1. CUDA GPU  (nvidia)
      2. MPS       (Apple Silicon — optional)
      3. CPU       (fallback)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def device_info() -> str:
    """Human-readable description of the selected device."""
    device = get_device()
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        return f"CUDA ({name}, {mem} MB)"
    if device.type == "mps":
        return "MPS (Apple Silicon)"
    return "CPU"
