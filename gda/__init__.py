"""
GDA Smart Building – Generalized Domain Adaptation for IoT Activity Recognition
================================================================================
Official implementation of:
  "Scalable Activity Recognition in Smart Buildings via
   Generalized Domain Adaptation of IoT Sensor Data"

Three GDA methods adapted for 1-D IoT sensor time-series:
  • ERM  – Empirical Risk Minimization (baseline)
  • SWAD – Stochastic Weight Averaging Densely
  • DFDG – Distribution-Free Domain Generalization
  • TERM – Tilted Empirical Risk Minimization
"""

__version__ = "1.0.0"
__author__  = "Smart Building GDA Research"
__license__ = "MIT"

from gda.data      import SmartBuildingDataGenerator, SensorDataset, make_lobo_splits, load_npz
from gda.models    import SensorActivityModel, build_model
from gda.methods   import ERMTrainer, SWADTrainer, DFDGTrainer, TERMTrainer
from gda.utils     import set_seed, get_device, compute_metrics

__all__ = [
    "SmartBuildingDataGenerator", "SensorDataset", "make_lobo_splits", "load_npz",
    "SensorActivityModel", "build_model",
    "ERMTrainer", "SWADTrainer", "DFDGTrainer", "TERMTrainer",
    "set_seed", "get_device", "compute_metrics",
]
