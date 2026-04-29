"""
Method 1 – Empirical Risk Minimization (ERM) Baseline
=======================================================
Standard supervised cross-entropy training on pooled source domains.
No domain generalization objective — serves as the baseline.

In the paper: ERM is the simplest GDA method where the model is trained
on all available source domains without any explicit domain alignment.
For smart buildings, this means pooling IoT data from multiple buildings.

Usage
-----
    from gda.methods.erm import ERMTrainer, ERMConfig

    cfg = ERMConfig(n_epochs=50, lr=1e-3, batch_size=128)
    trainer = ERMTrainer(model, cfg, device="cuda")
    history = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from gda.models.backbone import SensorActivityModel
from gda.utils.metrics import compute_metrics
from gda.utils.logging import TrainingLogger


@dataclass
class ERMConfig:
    """ERM training hyper-parameters."""
    n_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    dropout: float = 0.3
    model_size: str = "medium"           # 'small' | 'medium' | 'large'
    use_class_weights: bool = True        # weight CE by inverse class freq
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    patience: int = 15                    # early-stopping patience (epochs)
    scheduler: str = "cosine"            # 'cosine' | 'step' | 'none'
    save_best: bool = True
    checkpoint_path: str = "checkpoints/erm_best.pt"


class ERMTrainer:
    """
    Empirical Risk Minimization trainer for smart-building activity recognition.

    Parameters
    ----------
    model : SensorActivityModel
    config : ERMConfig
    device : str | torch.device
    class_weights : torch.Tensor | None
        Optional per-class weights for the CE loss.
    """

    METHOD_NAME = "ERM"

    def __init__(
        self,
        model: SensorActivityModel,
        config: ERMConfig,
        device: str | torch.device = "cpu",
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.logger = TrainingLogger(method=self.METHOD_NAME)

        # Loss
        cw = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(
            weight=cw,
            label_smoothing=config.label_smoothing,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

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
        """
        Full training loop.

        Returns
        -------
        history : list of epoch dicts with train/val metrics
        """
        self.logger.start(self.config.n_epochs, self.model)

        for epoch in range(1, self.config.n_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)
            val_metrics   = self._eval_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc":  train_metrics["acc"],
                "train_f1":   train_metrics["f1"],
                "val_loss":   val_metrics["loss"],
                "val_acc":    val_metrics["acc"],
                "val_f1":     val_metrics["f1"],
                "time_s":     elapsed,
            }
            self.history.append(record)
            self.logger.log_epoch(record)

            # Early stopping + best-model saving
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
        self, loader: DataLoader, load_best: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on a DataLoader. Optionally loads best checkpoint first.

        Returns
        -------
        dict with 'acc', 'f1', 'loss', 'per_class_f1'
        """
        if load_best and self.config.save_best:
            self._load_checkpoint(self.config.checkpoint_path)
        return self._eval_epoch(loader, detailed=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(x)
            loss = self.criterion(logits, y)
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
            loss = self.criterion(logits, y)
            total_loss += loss.item() * len(y)
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        acc, f1 = compute_metrics(preds, labels)
        result = {"loss": total_loss / len(labels), "acc": acc, "f1": f1}

        if detailed:
            acc, f1, per_class_f1 = compute_metrics(preds, labels, per_class=True)
            result["acc"] = acc
            result["f1"]  = f1
            result["per_class_f1"] = per_class_f1

        return result

    def _build_scheduler(self):
        if self.config.scheduler == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=self.config.n_epochs, eta_min=1e-6
            )
        if self.config.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.n_epochs // 3, gamma=0.1
            )
        return None

    def _save_checkpoint(self, path: str) -> None:
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "config": self.config,
        }, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(
            f"Loaded best checkpoint (val F1={ckpt['best_val_f1']:.4f} "
            f"@ epoch {ckpt['best_epoch']})"
        )
