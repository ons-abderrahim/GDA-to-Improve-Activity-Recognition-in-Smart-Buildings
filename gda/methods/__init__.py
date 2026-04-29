"""
GDA Methods for Smart Building Activity Recognition
=====================================================
Three state-of-the-art Generalized Domain Adaptation methods,
each adapted for 1-D IoT sensor time-series data.

Methods
-------
ERM   – Empirical Risk Minimization (baseline)
SWAD  – Stochastic Weight Averaging Densely
DFDG  – Distribution-Free Domain Generalization
TERM  – Tilted Empirical Risk Minimization
"""

from .erm  import ERMTrainer,  ERMConfig
from .swad import SWADTrainer, SWADConfig
from .dfdg import DFDGTrainer, DFDGConfig
from .term import TERMTrainer, TERMConfig

__all__ = [
    "ERMTrainer",  "ERMConfig",
    "SWADTrainer", "SWADConfig",
    "DFDGTrainer", "DFDGConfig",
    "TERMTrainer", "TERMConfig",
]
