"""
GDA Smart Building – Data Module
=================================
Handles:
  • Synthetic smart-building IoT sensor data generation
  • Real dataset loaders (CASAS, OPPORTUNITY, UCI-HAR friendly format)
  • Leave-One-Building-Out (LOBO) split generation
  • PyTorch Dataset / DataLoader wrappers
"""

from .generator import SmartBuildingDataGenerator
from .dataset import SensorDataset, SensorWindow
from .loaders import load_npz, load_casas_style, make_lobo_splits
from .transforms import NormalizeSensor, RandomWindowDrop, AddSensorNoise

__all__ = [
    "SmartBuildingDataGenerator",
    "SensorDataset",
    "SensorWindow",
    "load_npz",
    "load_casas_style",
    "make_lobo_splits",
    "NormalizeSensor",
    "RandomWindowDrop",
    "AddSensorNoise",
]
