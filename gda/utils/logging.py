"""
Training Logger
================
Lightweight logging for GDA training runs.
Provides consistent console output across all methods.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Optional

from gda.models.backbone import SensorActivityModel


def get_logger(name: str = "gda") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class TrainingLogger:
    """
    Structured per-method training logger.

    Parameters
    ----------
    method : str  Method name (e.g. 'ERM', 'SWAD', 'DFDG', 'TERM')
    log_every : int  Print epoch metrics every N epochs
    """

    def __init__(self, method: str = "GDA", log_every: int = 1):
        self.method = method
        self.log_every = log_every
        self._logger = get_logger(f"gda.{method.lower()}")

    def info(self, msg: str) -> None:
        self._logger.info(f"[{self.method}] {msg}")

    def start(self, n_epochs: int, model: Optional[SensorActivityModel] = None) -> None:
        self.info(f"Training started | {n_epochs} epochs")
        if model is not None:
            self.info(f"Model: {model}")

    def log_epoch(self, record: Dict, extra: str = "") -> None:
        epoch = record.get("epoch", "?")
        if epoch % self.log_every != 0:
            return
        line = (
            f"Epoch {epoch:3d} | "
            f"loss={record.get('train_loss', 0):.4f} "
            f"acc={record.get('train_acc', 0):.4f} "
            f"f1={record.get('train_f1', 0):.4f} | "
            f"val_loss={record.get('val_loss', 0):.4f} "
            f"val_acc={record.get('val_acc', 0):.4f} "
            f"val_f1={record.get('val_f1', 0):.4f}"
        )
        if extra:
            line += f"  {extra}"
        self.info(line)

    def finish(self, best_f1: float, best_epoch: int) -> None:
        self.info(
            f"Training complete. Best val F1={best_f1:.4f} @ epoch {best_epoch}."
        )
