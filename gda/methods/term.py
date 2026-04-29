"""
Method 4 – TERM: Tilted Empirical Risk Minimization
=====================================================
Adapted from: "Tilted Empirical Risk Minimization" (Li et al., ICLR 2021)

Key idea
--------
Standard ERM minimizes the *mean* loss across samples:

    L_ERM = (1/N) Σ ℓᵢ

This treats rare and common activities equally, causing the model to
underfit hard/minority examples (e.g., Exercising vs. Idle in a building
dataset with heavy class imbalance).

TERM replaces mean loss with a *tilted* risk objective:

    L_TERM(t) = (1/t) · log [ (1/N) Σ exp(t · ℓᵢ) ]

Properties:
  t → 0   : L_TERM → L_ERM (mean loss)
  t > 0   : Emphasizes high-loss (hard/minority) samples
  t < 0   : Emphasizes low-loss (easy) samples

For smart-building activity recognition:
  - Set t > 0  to focus on rare activities (Exercising, Leaving/Entering)
  - Reduces class imbalance sensitivity without explicit reweighting
  - Provides natural robustness to domain-specific hard samples

Smart-building adaptation
--------------------------
In buildings with strong class imbalance (e.g., 70% Idle, <5% Exercising),
TERM with t ∈ [2, 5] significantly improves minority class recall without
requiring explicit class weights or oversampling.

Usage
-----
    from gda.methods.term import TERMTrainer, TERMConfig

    cfg = TERMConfig(n_epochs=60, tilt=3.0)
    trainer = TERMTrainer(model, cfg, device="cuda")
    history = trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from gda.models.backbone import SensorActivityModel
from gda.utils.metrics import compute_metrics
from gda.utils.logging import TrainingLogger


@dataclass
class TERMConfig:
    """TERM hyper-parameters."""
    n_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    dropout: float = 0.3
    model_size: str = "medium"
    grad_clip: float = 1.0
    # ---- TERM-specific ----
    tilt: float = 3.0             # t parameter; t>0 → hard-example focus
    tilt_warmup_epochs: int = 5   # epochs with standard CE before tilting
    clamp_loss: float = 20.0      # clamp individual losses before exp (numerical stability)
    patience: int = 15
    save_best: bool = True
    checkpoint_path: str = "checkpoints/term_best.pt"


def tilted_loss(
    per_sample_losses: torch.Tensor,
    t: float,
    clamp: float = 20.0,
) -> torch.Tensor:
    """
    Compute TERM tilted risk given per-sample losses.

    L_TERM(t) = (1/t) · log[ mean(exp(t · ℓᵢ)) ]

    Parameters
    ----------
    per_sample_losses : (B,) per-sample cross-entropy values
    t : float  tilt parameter
    clamp : float  max value before exp (avoids overflow)

    Returns
    -------
    scalar tilted loss
    """
    if abs(t) < 1e-6:
        return per_sample_losses.mean()

    # Numerical stability: subtract max before exp  (log-sum-exp trick)
    scaled = t * per_sample_losses.clamp(max=clamp)
    max_val = scaled.max().detach()
    log_mean_exp = (scaled - max_val).exp().mean().log() + max_val
    return log_mean_exp / t


class TERMTrainer:
    """
    TERM trainer for smart-building IoT activity recognition.

    The tilted risk places extra emphasis on high-loss samples,
    which in the smart-building context means rare / hard activities
    like Exercising or Leaving/Entering that are often underrepresented.

    Parameters
    ----------
    model : SensorActivityModel
    config : TERMConfig
    device : str | torch.device
    """

    METHOD_NAME = "TERM"

    def __init__(
        self,
        model: SensorActivityModel,
        config: TERMConfig,
        device: str | torch.device = "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.logger = TrainingLogger(method=self.METHOD_NAME)

        # Reduction='none' → per-sample losses for tilting
        self.criterion = nn.CrossEntropyLoss(reduction="none")

        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.n_epochs, eta_min=1e-6
        )

        self.history: List[Dict] = []
        self.best_val_f1 = 0.0
        self.best_epoch  = 0
        self._no_improve  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> List[Dict]:
        """Full TERM training loop."""
        self.logger.start(self.config.n_epochs, self.model)
        self.logger.info(
            f"tilt t={self.config.tilt}, "
            f"warmup={self.config.tilt_warmup_epochs} epochs"
        )

        for epoch in range(1, self.config.n_epochs + 1):
            t0 = time.time()
            t = self.config.tilt if epoch > self.config.tilt_warmup_epochs else 0.0
            train_metrics = self._train_epoch(train_loader, t=t)
            val_metrics   = self._eval_epoch(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            record = {
                "epoch":      epoch,
                "tilt_t":     t,
                "train_loss": train_metrics["loss"],
                "train_acc":  train_metrics["acc"],
                "train_f1":   train_metrics["f1"],
                "val_loss":   val_metrics["loss"],
                "val_acc":    val_metrics["acc"],
                "val_f1":     val_metrics["f1"],
                "time_s":     elapsed,
            }
            self.history.append(record)
            self.logger.log_epoch(record, extra=f"[t={t:.1f}]")

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch  = epoch
                self._no_improve  = 0
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
        self, loader: DataLoader, load_best: bool = True
    ) -> Dict[str, float]:
        if load_best and self.config.save_best:
            self._load_checkpoint(self.config.checkpoint_path)
        return self._eval_epoch(loader, detailed=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(
        self, loader: DataLoader, t: float
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits, _ = self.model(x)
            per_sample_ce = self.criterion(logits, y)  # (B,)
            loss = tilted_loss(per_sample_ce, t=t, clamp=self.config.clamp_loss)
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc, f1 = compute_metrics(preds, labels)
        return {"loss": total_loss / len(labels), "acc": acc, "f1": f1}

    @torch.no_grad()
    def _eval_epoch(
        self, loader: DataLoader, detailed: bool = False
    ) -> Dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.model(x)
            # Eval always uses standard mean CE
            loss = self.criterion(logits, y).mean()
            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        N = len(labels)

        if detailed:
            acc, f1, pcf1 = compute_metrics(preds, labels, per_class=True)
            return {"loss": total_loss / N, "acc": acc, "f1": f1, "per_class_f1": pcf1}
        acc, f1 = compute_metrics(preds, labels)
        return {"loss": total_loss / N, "acc": acc, "f1": f1}

    def _save_checkpoint(self, path: str) -> None:
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_epoch":  self.best_epoch,
            "config":      self.config,
        }, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(
            f"Loaded TERM checkpoint (val F1={ckpt['best_val_f1']:.4f} "
            f"@ epoch {ckpt['best_epoch']})"
        )
