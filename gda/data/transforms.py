"""
Sensor Data Transforms / Augmentations
=======================================
Lightweight augmentations designed for smart-building IoT sensor windows.
Each transform operates on a numpy array of shape (T, C).

These can be composed with torchvision.transforms.Compose or a simple
list + sequential application.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


class NormalizeSensor:
    """
    Z-score normalization per channel.

    Parameters
    ----------
    mean : np.ndarray | None, shape (C,)
        Pre-computed channel means. Computed from input if None.
    std : np.ndarray | None, shape (C,)
        Pre-computed channel stds.
    eps : float
        Small constant to avoid division by zero.
    """

    def __init__(
        self,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        eps: float = 1e-6,
    ):
        self.mean = mean
        self.std = std
        self.eps = eps

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-6) -> "NormalizeSensor":
        """
        Fit normalization statistics from an (N, T, C) dataset.
        Computes statistics across N and T axes.
        """
        mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
        std  = X.reshape(-1, X.shape[-1]).std(axis=0)
        return cls(mean=mean, std=std, eps=eps)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        """Apply normalization to a single (T, C) window."""
        mean = self.mean if self.mean is not None else window.mean(axis=0)
        std  = self.std  if self.std  is not None else window.std(axis=0)
        return (window - mean) / (std + self.eps)

    def transform_dataset(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization to an (N, T, C) dataset."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Call NormalizeSensor.fit() first, or provide mean/std.")
        return (X - self.mean) / (self.std + self.eps)


class AddSensorNoise:
    """
    Add Gaussian noise to continuous sensor channels (indices 4–8).
    Binary channels (PIR, door contact) are left unchanged.

    Parameters
    ----------
    std : float
        Standard deviation of the additive Gaussian noise.
    continuous_channels : list[int]
        Indices of continuous sensor channels to perturb.
    """

    CONTINUOUS_DEFAULTS = [4, 5, 6, 7, 8]  # Temp, Humid, CO2, Light, Power

    def __init__(
        self,
        std: float = 0.05,
        continuous_channels: Optional[list] = None,
        seed: Optional[int] = None,
    ):
        self.std = std
        self.channels = continuous_channels or self.CONTINUOUS_DEFAULTS
        self.rng = np.random.default_rng(seed)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        window = window.copy()
        noise = self.rng.normal(0, self.std, size=(window.shape[0], len(self.channels)))
        window[:, self.channels] += noise
        return window


class RandomWindowDrop:
    """
    Randomly zero out entire sensor channels to simulate sensor failure.

    Parameters
    ----------
    drop_prob : float
        Probability that any given channel is zeroed out.
    """

    def __init__(self, drop_prob: float = 0.1, seed: Optional[int] = None):
        self.drop_prob = drop_prob
        self.rng = np.random.default_rng(seed)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        window = window.copy()
        C = window.shape[1]
        mask = self.rng.random(C) < self.drop_prob
        window[:, mask] = 0.0
        return window


class RandomTimeShift:
    """
    Randomly shift the temporal window by a few timesteps (circular shift).
    Simulates asynchronous sensor sampling.

    Parameters
    ----------
    max_shift : int
        Maximum number of timesteps to shift left or right.
    """

    def __init__(self, max_shift: int = 5, seed: Optional[int] = None):
        self.max_shift = max_shift
        self.rng = np.random.default_rng(seed)

    def __call__(self, window: np.ndarray) -> np.ndarray:
        shift = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
        return np.roll(window, shift, axis=0)


class Compose:
    """Compose multiple transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, window: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            window = t(window)
        return window


# ---------------------------------------------------------------------------
# Factory: default augmentation pipeline for training
# ---------------------------------------------------------------------------

def default_train_transforms(
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    noise_std: float = 0.05,
    drop_prob: float = 0.08,
    seed: Optional[int] = None,
) -> Compose:
    """
    Standard augmentation pipeline for training:
      1. Z-score normalization
      2. Additive Gaussian noise on continuous channels
      3. Random channel dropout (sensor failure simulation)
    """
    transforms = []
    transforms.append(NormalizeSensor(mean=mean, std=std))
    if noise_std > 0:
        transforms.append(AddSensorNoise(std=noise_std, seed=seed))
    if drop_prob > 0:
        transforms.append(RandomWindowDrop(drop_prob=drop_prob, seed=seed))
    return Compose(transforms)


def default_eval_transforms(
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Compose:
    """
    Evaluation pipeline (normalization only, no stochastic augmentation).
    """
    return Compose([NormalizeSensor(mean=mean, std=std)])
