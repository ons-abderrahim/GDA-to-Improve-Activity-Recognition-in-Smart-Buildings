"""
General Utilities
==================
Reproducibility seeds, device helpers, checkpoint management.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for full reproducibility.
    Covers Python, NumPy, PyTorch (CPU + CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (may slow training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """
    Auto-select the best available device.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"[device] Using CUDA: {name}")
    elif prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[device] Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("[device] Using CPU")
    return device


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model: torch.nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: Optional[torch.device] = None,
    key: str = "model_state",
) -> dict:
    """
    Load a model checkpoint safely.

    Parameters
    ----------
    model  : nn.Module to load weights into
    path   : path to .pt checkpoint file
    device : target device (auto-detected if None)
    key    : key in the checkpoint dict for model state

    Returns
    -------
    The full checkpoint dict (for access to metadata).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = device or get_device()
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt[key])
    print(f"[checkpoint] Loaded from {path}")
    return ckpt


def save_results_json(results: dict, path: str | Path) -> None:
    """Save evaluation results dict to JSON."""
    import json
    path = Path(path)
    ensure_dir(path.parent)
    # Convert numpy arrays to lists for JSON serialisation
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    clean = {k: _convert(v) for k, v in results.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[results] Saved to {path}")
