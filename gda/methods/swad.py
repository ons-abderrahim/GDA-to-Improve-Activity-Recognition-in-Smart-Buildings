"""
Method 2 – SWAD: Stochastic Weight Averaging Densely
======================================================
Adapted from: "In Search of Lost Domain Generalization" (Gulrajani & Lopez-Paz, 2020)
              and SWAD: "Domain Generalization by Seeking Flat Minima" (Cha et al., 2021)

Key idea
--------
Standard ERM finds a sharp local minimum that overfits to source buildings.
SWAD maintains a *running average* of model weights during late training,
steering toward flatter, broader minima that generalize better to unseen
building environments.

Dense averaging: unlike classic SWA (one average per epoch), SWAD updates
the running average **at every training step** once averaging starts.
This finer-grained averaging has been shown to yield better generalization.

Smart-building adaptation
--------------------------
Building-to-building domain shifts arise from:
  - Different sensor placements (PIR, door contacts in different rooms)
  - Different occupant routines (activity distributions vary per building)
  - Different hardware (calibration drifts, sampling rates)

SWAD's flat-minima bias makes it robust to these distribution shifts
without requiring explicit domain labels at test time.

Usage
-----
    from gda.methods.swad import SWADTrainer, SWADConfig

    cfg = SWADConfig(n_epochs=60, swa_start_epoch=30)
    trainer = SWADTrainer(model, cfg, device="cuda")
    history = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader

from gda.models.backbone import SensorActivityModel
from gda.utils.metrics import compute_metrics
from gda.utils.logging import TrainingLogger


@dataclass
class SWADConfig:
    """SWAD hyper-parameters."""
    n_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    dropout: float = 0.3
    model_size: str = "medium"
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    # ---- SWAD-specific ----
    swa_start_epoch: int = 30        # epoch at which dense averaging begins
    swa_lr: float = 5e-4             # LR used during SWA phase
    dense: bool = True               # True → update average every step; False → every epoch
    patience: int = 15
    save_best: bool = True
    checkpoint_path: str = "checkpoints/swad_best.pt"


class SWADTrainer:
    """
    SWAD trainer for smart-building IoT activity recognition.

    The trainer maintains two models:
      • self.model      – the live ERM model being SGD-optimized
      • self.swa_model  – the averaged weight model (AveragedModel)

    During inference (evaluate), the swa_model is used with updated
    batch-norm statistics.

    Parameters
    ----------
    model : SensorActivityModel
    config : SWADConfig
    device : str | torch.device
    class_weights : torch.Tensor | None
    """

    METHOD_NAME = "SWAD"

    def __init__(
        self,
        model: SensorActivityModel,
        config: SWADConfig,
        device: str | torch.device = "cpu",
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.logger = TrainingLogger(method=self.METHOD_NAME)

        cw = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=cw, label_smoothing=config.label_smoothing
        )

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Pre-SWA cosine schedule
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.swa_start_epoch, eta_min=config.swa_lr
        )

        # SWA averaged model & SWA LR scheduler
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            self.optimizer, swa_lr=config.swa_lr, anneal_epochs=5
        )

        self._in_swa_phase = False
        self._swa_step_count = 0
        self.history: List[Dict] = []
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self._no_improve = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict]:
        """Full SWAD training loop."""
        self.logger.start(self.config.n_epochs, self.model)
        self.logger.info(
            f"SWA averaging starts at epoch {self.config.swa_start_epoch} "
            f"({'dense step-level' if self.config.dense else 'epoch-level'} updates)"
        )

        for epoch in range(1, self.config.n_epochs + 1):
            t0 = time.time()

            if epoch >= self.config.swa_start_epoch and not self._in_swa_phase:
                self._in_swa_phase = True
                self.logger.info(f"[Epoch {epoch}] SWA phase started.")

            train_metrics = self._train_epoch(train_loader, epoch)

            # After each epoch in SWA phase: update BN, evaluate SWA model
            if self._in_swa_phase:
                if not self.config.dense:
                    # Epoch-level update (classic SWA behaviour)
                    self.swa_model.update_parameters(self.model)
                # Update BN stats on train data
                update_bn(train_loader, self.swa_model, device=self.device)
                val_metrics = self._eval_epoch(val_loader, use_swa=True)
                self.swa_scheduler.step()
            else:
                val_metrics = self._eval_epoch(val_loader, use_swa=False)
                self.cosine_scheduler.step()

            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "phase": "SWA" if self._in_swa_phase else "ERM",
                "train_loss": train_metrics["loss"],
                "train_acc":  train_metrics["acc"],
                "train_f1":   train_metrics["f1"],
                "val_loss":   val_metrics["loss"],
                "val_acc":    val_metrics["acc"],
                "val_f1":     val_metrics["f1"],
                "swa_steps":  self._swa_step_count,
                "time_s":     elapsed,
            }
            self.history.append(record)
            self.logger.log_epoch(record, extra=f"[{record['phase']}]")

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch  = epoch
                self._no_improve = 0
                if self.config.save_best:
                    self._save_checkpoint(self.config.checkpoint_path)
            else:
                self._no_improve += 1
                if self._no_improve >= self.config.patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(best val F1={self.best_val_f1:.4f} @ epoch {self.best_epoch})"
                    )
                    break

        self.logger.finish(self.best_val_f1, self.best_epoch)
        return self.history

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        train_loader_for_bn: Optional[DataLoader] = None,
        load_best: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate using the SWA-averaged model.
        If a train_loader_for_bn is provided, BN stats are updated first.
        """
        if load_best and self.config.save_best:
            self._load_checkpoint(self.config.checkpoint_path)
        if train_loader_for_bn is not None:
            update_bn(train_loader_for_bn, self.swa_model, device=self.device)
        return self._eval_epoch(loader, use_swa=True, detailed=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for step, (x, y, _) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            # Dense SWA: update averaged model every step during SWA phase
            if self._in_swa_phase and self.config.dense:
                self.swa_model.update_parameters(self.model)
                self._swa_step_count += 1

            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc, f1 = compute_metrics(preds, labels)
        return {"loss": total_loss / len(labels), "acc": acc, "f1": f1}

    @torch.no_grad()
    def _eval_epoch(
        self,
        loader: DataLoader,
        use_swa: bool = False,
        detailed: bool = False,
    ) -> Dict:
        eval_model = self.swa_model if use_swa else self.model
        eval_model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = eval_model(x)
            loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)

        if detailed:
            acc, f1, pcf1 = compute_metrics(preds, labels, per_class=True)
            return {"loss": total_loss / len(labels), "acc": acc, "f1": f1,
                    "per_class_f1": pcf1, "swa_steps": self._swa_step_count}
        acc, f1 = compute_metrics(preds, labels)
        return {"loss": total_loss / len(labels), "acc": acc, "f1": f1}

    def _save_checkpoint(self, path: str) -> None:
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "swa_model_state": self.swa_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_f1":     self.best_val_f1,
            "best_epoch":      self.best_epoch,
            "swa_steps":       self._swa_step_count,
            "config":          self.config,
        }, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.swa_model.load_state_dict(ckpt["swa_model_state"])
        self.logger.info(
            f"Loaded SWAD checkpoint (val F1={ckpt['best_val_f1']:.4f} "
            f"@ epoch {ckpt['best_epoch']}, SWA steps={ckpt['swa_steps']})"
        )
