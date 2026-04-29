"""
SensorDataset & SensorWindow
==============================
PyTorch Dataset wrappers for smart-building IoT sensor data.

Expected data layout: X of shape (N, T, C)
  N – number of windows
  T – timesteps per window (e.g. 50)
  C – sensor channels (e.g. 9)

The model receives tensors of shape (B, C, T) – channels-first,
following PyTorch Conv1d convention. The Dataset handles transposing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SensorWindow:
    """Single labelled sensor window."""
    x: torch.Tensor       # (C, T)
    y: torch.Tensor       # scalar int64
    domain: torch.Tensor  # scalar int64


class SensorDataset(Dataset):
    """
    PyTorch Dataset for IoT sensor windows.

    Parameters
    ----------
    X : np.ndarray, shape (N, T, C)
        Sensor windows.
    y : np.ndarray, shape (N,)
        Integer activity labels.
    domains : np.ndarray | None, shape (N,)
        Building / domain IDs. Zeros if None.
    transform : callable | None
        Optional transform applied to the (T, C) numpy window
        *before* transposing to (C, T).
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        domains: Optional[np.ndarray] = None,
        transform: Optional[Callable] = None,
    ):
        assert X.ndim == 3, f"X must be (N, T, C), got {X.shape}"
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.domains = (
            domains.astype(np.int64)
            if domains is not None
            else np.zeros(len(y), dtype=np.int64)
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.X[idx]  # (T, C)
        if self.transform is not None:
            window = self.transform(window)
        # Transpose to channels-first (C, T) for Conv1d
        x = torch.from_numpy(window.T.copy())          # (C, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        d = torch.tensor(self.domains[idx], dtype=torch.long)
        return x, y, d

    @property
    def n_channels(self) -> int:
        return self.X.shape[2]

    @property
    def window_size(self) -> int:
        return self.X.shape[1]

    @property
    def n_classes(self) -> int:
        return int(self.y.max()) + 1

    @property
    def n_domains(self) -> int:
        return int(self.domains.max()) + 1

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for weighted CE loss."""
        counts = np.bincount(self.y, minlength=self.n_classes).astype(float)
        counts = np.maximum(counts, 1)
        weights = 1.0 / counts
        weights /= weights.sum()
        return torch.tensor(weights * self.n_classes, dtype=torch.float32)

    def __repr__(self) -> str:
        return (
            f"SensorDataset(N={len(self)}, T={self.window_size}, "
            f"C={self.n_channels}, classes={self.n_classes}, "
            f"domains={self.n_domains})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: SensorDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Wrap a SensorDataset in a DataLoader with sensible defaults.

    Returns
    -------
    DataLoader yielding (x, y, domain) tuples.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
