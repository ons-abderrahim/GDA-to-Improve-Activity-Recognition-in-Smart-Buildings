"""
SmartBuildingDataGenerator
============================
Generates realistic synthetic IoT sensor data for smart-building
activity recognition experiments.

Sensor types modelled
---------------------
  - PIR motion sensors       (binary, room-level presence)
  - Door/window contacts      (binary, open/close events)
  - Temperature sensors       (continuous, °C)
  - Humidity sensors          (continuous, %)
  - CO₂ / air-quality sensors (continuous, ppm)
  - Light intensity sensors   (continuous, lux)
  - Power consumption meters  (continuous, W)

Activity classes (default 6)
-----------------------------
  0 – Idle / No Activity
  1 – Cooking
  2 – Working at Desk
  3 – Exercising
  4 – Sleeping
  5 – Leaving / Entering

Domain shifts simulated
-----------------------
  Each "building" domain has:
    • Different sensor placement → shifted mean activations
    • Different occupant routines → activity distribution shifts
    • Different sensor hardware → noise level / drift offsets

Usage
-----
    >>> from gda.data.generator import SmartBuildingDataGenerator
    >>> gen = SmartBuildingDataGenerator(n_buildings=5, seed=42)
    >>> X, y, domains = gen.generate(n_samples_per_building=500)
    >>> # X: (N, T, C)  y: (N,)  domains: (N,)
    >>> gen.save("data/synthetic_smart_building.npz")
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# ---------------------------------------------------------------------------
# Activity-specific sensor signatures
# Shape: [n_sensors] mean activation per activity
# Sensors: [PIR1, PIR2, PIR3, DoorContact, Temp, Humidity, CO2, Light, Power]
# ---------------------------------------------------------------------------
_ACTIVITY_PROFILES: Dict[int, Dict] = {
    0: {  # Idle
        "name": "Idle",
        "pir_mean": [0.05, 0.02, 0.03],
        "door_mean": 0.02,
        "temp_base": 21.0,
        "humid_base": 45.0,
        "co2_base": 450.0,
        "light_base": 200.0,
        "power_base": 120.0,
    },
    1: {  # Cooking
        "name": "Cooking",
        "pir_mean": [0.8, 0.1, 0.05],
        "door_mean": 0.3,
        "temp_base": 25.0,
        "humid_base": 60.0,
        "co2_base": 600.0,
        "light_base": 400.0,
        "power_base": 1800.0,
    },
    2: {  # Working at Desk
        "name": "Working",
        "pir_mean": [0.05, 0.9, 0.02],
        "door_mean": 0.05,
        "temp_base": 22.0,
        "humid_base": 42.0,
        "co2_base": 700.0,
        "light_base": 500.0,
        "power_base": 350.0,
    },
    3: {  # Exercising
        "name": "Exercising",
        "pir_mean": [0.7, 0.4, 0.6],
        "door_mean": 0.1,
        "temp_base": 24.0,
        "humid_base": 65.0,
        "co2_base": 900.0,
        "light_base": 600.0,
        "power_base": 200.0,
    },
    4: {  # Sleeping
        "name": "Sleeping",
        "pir_mean": [0.0, 0.0, 0.02],
        "door_mean": 0.0,
        "temp_base": 20.0,
        "humid_base": 50.0,
        "co2_base": 800.0,
        "light_base": 5.0,
        "power_base": 80.0,
    },
    5: {  # Leaving/Entering
        "name": "Leaving_Entering",
        "pir_mean": [0.5, 0.1, 0.8],
        "door_mean": 0.9,
        "temp_base": 21.5,
        "humid_base": 46.0,
        "co2_base": 420.0,
        "light_base": 300.0,
        "power_base": 150.0,
    },
}

ACTIVITY_NAMES = {k: v["name"] for k, v in _ACTIVITY_PROFILES.items()}
SENSOR_NAMES = [
    "PIR_Kitchen", "PIR_Office", "PIR_Hallway",
    "DoorContact",
    "Temperature_C", "Humidity_pct", "CO2_ppm", "Light_lux", "Power_W",
]
N_SENSORS = len(SENSOR_NAMES)  # 9


class SmartBuildingDataGenerator:
    """
    Generate multi-domain smart-building IoT sensor datasets.

    Parameters
    ----------
    n_buildings : int
        Number of building domains to simulate.
    n_activities : int
        Number of activity classes (2-6 supported; defaults use 6).
    window_size : int
        Temporal window length T (timesteps per sample).
    seed : int
        Global random seed for reproducibility.
    noise_scale : float
        Base noise multiplier (building-specific offsets added on top).
    """

    def __init__(
        self,
        n_buildings: int = 5,
        n_activities: int = 6,
        window_size: int = 50,
        seed: int = 42,
        noise_scale: float = 1.0,
    ):
        if not 2 <= n_activities <= 6:
            raise ValueError("n_activities must be between 2 and 6.")
        self.n_buildings = n_buildings
        self.n_activities = n_activities
        self.window_size = window_size
        self.seed = seed
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

        # Per-building domain shift parameters (sampled once at init)
        self._building_offsets = self._sample_building_offsets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        n_samples_per_building: int = 500,
        imbalance: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sensor window data.

        Returns
        -------
        X : np.ndarray, shape (N, T, C)
            Sensor windows. T = window_size, C = N_SENSORS.
        y : np.ndarray, shape (N,)
            Integer activity labels.
        domains : np.ndarray, shape (N,)
            Integer building/domain IDs.
        """
        X_list, y_list, d_list = [], [], []

        for bld_idx in range(self.n_buildings):
            n = n_samples_per_building
            y_bld = self._sample_activity_labels(n, bld_idx, imbalance)
            X_bld = np.stack(
                [self._generate_window(act, bld_idx) for act in y_bld], axis=0
            )  # (n, T, C)
            X_list.append(X_bld)
            y_list.append(y_bld)
            d_list.append(np.full(n, bld_idx, dtype=np.int64))

        X = np.concatenate(X_list, axis=0).astype(np.float32)
        y = np.concatenate(y_list, axis=0).astype(np.int64)
        domains = np.concatenate(d_list, axis=0).astype(np.int64)

        self._X, self._y, self._domains = X, y, domains
        return X, y, domains

    def save(self, path: str | Path) -> None:
        """Save generated dataset to a compressed .npz file."""
        if not hasattr(self, "_X"):
            raise RuntimeError("Call .generate() before .save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self._X,
            y=self._y,
            domains=self._domains,
            sensor_names=np.array(SENSOR_NAMES),
            activity_names=np.array([ACTIVITY_NAMES[i] for i in range(self.n_activities)]),
            window_size=np.array(self.window_size),
            n_buildings=np.array(self.n_buildings),
        )
        print(f"[DataGenerator] Saved {len(self._X)} samples → {path}")

    def get_activity_names(self) -> List[str]:
        return [ACTIVITY_NAMES[i] for i in range(self.n_activities)]

    def get_sensor_names(self) -> List[str]:
        return SENSOR_NAMES

    def summary(self) -> str:
        if not hasattr(self, "_X"):
            return "No data generated yet. Call .generate() first."
        lines = [
            "=" * 55,
            "  SmartBuildingDataGenerator – Dataset Summary",
            "=" * 55,
            f"  Total samples  : {len(self._X)}",
            f"  Window size    : {self.window_size} timesteps",
            f"  Channels       : {N_SENSORS} sensors",
            f"  Buildings      : {self.n_buildings}",
            f"  Activities     : {self.n_activities}",
            "",
            "  Class distribution:",
        ]
        for i in range(self.n_activities):
            cnt = (self._y == i).sum()
            bar = "█" * (cnt // max(1, len(self._X) // 40))
            lines.append(f"    [{i}] {ACTIVITY_NAMES[i]:<20s} {cnt:5d}  {bar}")
        lines.append(
            "\n  Sensors: " + ", ".join(SENSOR_NAMES)
        )
        lines.append("=" * 55)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_building_offsets(self) -> List[Dict]:
        offsets = []
        for _ in range(self.n_buildings):
            offsets.append({
                "temp_drift": self.rng.uniform(-3.0, 3.0),
                "humid_drift": self.rng.uniform(-10.0, 10.0),
                "co2_drift": self.rng.uniform(-100.0, 150.0),
                "light_drift": self.rng.uniform(-50.0, 100.0),
                "power_drift": self.rng.uniform(-50.0, 100.0),
                "noise_mult": self.rng.uniform(0.5, 2.0) * self.noise_scale,
                "pir_sensitivity": self.rng.uniform(0.7, 1.3),
            })
        return offsets

    def _sample_activity_labels(
        self, n: int, bld_idx: int, imbalance: bool
    ) -> np.ndarray:
        if imbalance:
            # Simulate building-specific routine → class frequency varies
            base_probs = self.rng.dirichlet(
                alpha=np.ones(self.n_activities) * (1.5 + bld_idx * 0.3)
            )
        else:
            base_probs = np.ones(self.n_activities) / self.n_activities
        return self.rng.choice(self.n_activities, size=n, p=base_probs).astype(np.int64)

    def _generate_window(self, activity: int, bld_idx: int) -> np.ndarray:
        """
        Produce a single sensor window of shape (T, C) for a given
        activity label and building domain.
        """
        T = self.window_size
        prof = _ACTIVITY_PROFILES[activity]
        off = self._building_offsets[bld_idx]
        nm = off["noise_mult"]
        ps = off["pir_sensitivity"]

        window = np.zeros((T, N_SENSORS), dtype=np.float32)

        # ---- Binary sensors (PIR & door) --------------------------------
        for ch, pir_mean in enumerate(prof["pir_mean"]):
            p = np.clip(pir_mean * ps, 0.0, 1.0)
            # Bursty patterns: use Markov-like transitions
            seq = self._markov_binary(T, p_on=p, p_off=1 - p * 0.6)
            window[:, ch] = seq.astype(np.float32)

        door_p = np.clip(prof["door_mean"] * ps, 0.0, 1.0)
        window[:, 3] = self._markov_binary(T, p_on=door_p, p_off=0.9).astype(np.float32)

        # ---- Continuous sensors -----------------------------------------
        # Temperature
        temp = prof["temp_base"] + off["temp_drift"]
        window[:, 4] = self._smooth_signal(T, mean=temp, std=0.3 * nm, smooth=8)

        # Humidity
        hum = prof["humid_base"] + off["humid_drift"]
        window[:, 5] = self._smooth_signal(T, mean=hum, std=1.5 * nm, smooth=10)

        # CO₂
        co2 = prof["co2_base"] + off["co2_drift"]
        window[:, 6] = self._smooth_signal(T, mean=co2, std=20.0 * nm, smooth=6)

        # Light
        light = max(0.0, prof["light_base"] + off["light_drift"])
        window[:, 7] = np.clip(
            self._smooth_signal(T, mean=light, std=30.0 * nm, smooth=5), 0, None
        )

        # Power
        power = max(0.0, prof["power_base"] + off["power_drift"])
        # Add random spikes for appliances
        power_sig = self._smooth_signal(T, mean=power, std=50.0 * nm, smooth=4)
        n_spikes = self.rng.integers(0, 4)
        spike_pos = self.rng.integers(0, T, size=n_spikes)
        power_sig[spike_pos] += self.rng.uniform(100, 500, size=n_spikes)
        window[:, 8] = np.clip(power_sig, 0, None)

        return window  # (T, C)

    def _markov_binary(self, T: int, p_on: float, p_off: float) -> np.ndarray:
        """Simple 2-state Markov chain for bursty binary sensor signals."""
        seq = np.zeros(T, dtype=np.uint8)
        state = int(self.rng.random() < p_on)
        for t in range(T):
            seq[t] = state
            if state == 1:
                state = 0 if self.rng.random() < p_off else 1
            else:
                state = 1 if self.rng.random() < p_on else 0
        return seq

    def _smooth_signal(
        self, T: int, mean: float, std: float, smooth: int = 5
    ) -> np.ndarray:
        """Generate a smoothed Gaussian signal with temporal correlations."""
        raw = self.rng.normal(mean, std, size=T + smooth)
        kernel = np.ones(smooth) / smooth
        smoothed = np.convolve(raw, kernel, mode="valid")[:T]
        return smoothed.astype(np.float32)
