"""
Data Loaders
=============
Functions for loading .npz datasets and producing
Leave-One-Building-Out (LOBO) domain splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader

from .dataset import SensorDataset, make_dataloader


# ---------------------------------------------------------------------------
# NPZ Loader
# ---------------------------------------------------------------------------

def load_npz(
    path: str | Path,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a compressed .npz smart-building dataset.

    Expected arrays
    ---------------
      X       : (N, T, C)  float32
      y       : (N,)       int64
      domains : (N,)       int64

    Returns
    -------
    X, y, domains
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    domains = data["domains"].astype(np.int64) if "domains" in data else np.zeros(len(y), dtype=np.int64)

    if verbose:
        print(f"[load_npz] Loaded {path.name}: X={X.shape}, y={y.shape}, domains={domains.shape}")
        if "activity_names" in data:
            print(f"  Activities : {list(data['activity_names'])}")
        if "sensor_names" in data:
            print(f"  Sensors    : {list(data['sensor_names'])}")

    return X, y, domains


# ---------------------------------------------------------------------------
# CASAS-style loader
# ---------------------------------------------------------------------------

def load_casas_style(
    x_path: str | Path,
    y_path: str | Path,
    domain_path: Optional[str | Path] = None,
    window_size: int = 50,
    stride: int = 25,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw CASAS-format CSV data and apply sliding-window segmentation.

    Parameters
    ----------
    x_path : path to CSV of shape (timesteps, n_channels)
    y_path : path to CSV of shape (timesteps,) – integer activity labels
    domain_path : optional CSV of shape (timesteps,) – domain IDs
    window_size : sliding window length
    stride : window hop size

    Returns
    -------
    X : (N, T, C)
    y : (N,)  – majority vote per window
    domains : (N,)
    """
    import csv

    def _read_csv_array(p):
        p = Path(p)
        arr = []
        with open(p) as f:
            reader = csv.reader(f)
            for row in reader:
                arr.append([float(v) for v in row])
        return np.array(arr, dtype=np.float32)

    raw_X = _read_csv_array(x_path)
    raw_y = _read_csv_array(y_path).flatten().astype(np.int64)
    raw_d = (
        _read_csv_array(domain_path).flatten().astype(np.int64)
        if domain_path is not None
        else np.zeros(len(raw_y), dtype=np.int64)
    )

    if raw_X.ndim == 1:
        raw_X = raw_X[:, None]

    X_wins, y_wins, d_wins = [], [], []
    T = len(raw_y)
    for start in range(0, T - window_size + 1, stride):
        end = start + window_size
        X_wins.append(raw_X[start:end])           # (T, C)
        # majority vote for activity label
        labels_win = raw_y[start:end]
        y_wins.append(int(np.bincount(labels_win).argmax()))
        # domain: first value in window
        d_wins.append(int(raw_d[start]))

    X = np.stack(X_wins, axis=0)
    y = np.array(y_wins, dtype=np.int64)
    domains = np.array(d_wins, dtype=np.int64)

    if verbose:
        print(f"[load_casas_style] {len(X)} windows (size={window_size}, stride={stride})")

    return X, y, domains


# ---------------------------------------------------------------------------
# Leave-One-Building-Out (LOBO) split
# ---------------------------------------------------------------------------

def make_lobo_splits(
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    test_domain: int,
    val_fraction: float = 0.15,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Produce train / val / test DataLoaders using a Leave-One-Building-Out
    (LOBO) protocol.

    The building with ID == test_domain is held out as test set.
    val_fraction of the remaining training data is used for validation.

    Parameters
    ----------
    test_domain : int
        Building domain ID to hold out as unseen test domain.
    val_fraction : float
        Fraction of non-test data used for validation (0 < f < 1).

    Returns
    -------
    dict with keys 'train', 'val', 'test' → DataLoader
    """
    rng = np.random.default_rng(seed)

    test_mask = domains == test_domain
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    rng.shuffle(train_idx)
    n_val = max(1, int(len(train_idx) * val_fraction))
    val_idx = train_idx[:n_val]
    train_idx = train_idx[n_val:]
    test_idx = np.where(test_mask)[0]

    def _ds(idx):
        return SensorDataset(X[idx], y[idx], domains[idx])

    loaders = {
        "train": make_dataloader(_ds(train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val":   make_dataloader(_ds(val_idx),   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  make_dataloader(_ds(test_idx),  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }

    print(
        f"[LOBO] test_domain={test_domain} | "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )
    return loaders


# ---------------------------------------------------------------------------
# All-domain split (for single-dataset experiments)
# ---------------------------------------------------------------------------

def make_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    domains: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Random train / val / test split (stratified by class).
    Useful for quick experiments when LOBO is not needed.
    """
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(X))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=1 - train_frac - val_frac, stratify=y, random_state=seed
    )
    val_ratio = val_frac / (train_frac + val_frac)
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_ratio,
        stratify=y[idx_trainval],
        random_state=seed + 1,
    )

    def _ds(ix):
        return SensorDataset(X[ix], y[ix], domains[ix])

    return {
        "train": make_dataloader(_ds(idx_train), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val":   make_dataloader(_ds(idx_val),   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test":  make_dataloader(_ds(idx_test),  batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
