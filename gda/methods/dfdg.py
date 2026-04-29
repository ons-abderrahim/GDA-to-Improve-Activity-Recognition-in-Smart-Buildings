"""
Method 3 – DFDG: Distribution-Free Domain Generalization
==========================================================
Adapted from: "Domain-Free Domain Generalization" (Jin et al., 2021)

Key idea
--------
Standard domain generalization methods require explicit domain labels
(e.g., building IDs) at training time, which may be unavailable in
practice. DFDG achieves domain alignment **without requiring domain
labels** by minimizing the *energy distance* between two randomly
partitioned halves of each batch.

Energy Distance between two feature sets A = {u_i} and B = {v_j}:

    ED(A, B) = 2·E[||u - v||] - E[||u - u'||] - E[||v - v'||]

This encourages the feature distribution to become uniform (domain-invariant)
without explicitly knowing which samples come from which building.

Smart-building adaptation
--------------------------
The total training loss is:

    L = L_CE(logits, labels) + λ_df · L_DF(features)

where L_DF is the energy distance between two random sub-batches.
This forces the Conv1D+BiGRU encoder to produce building-invariant
feature representations, improving zero-shot transfer to new buildings.

Usage
-----
    from gda.methods.dfdg import DFDGTrainer, DFDGConfig

    cfg = DFDGConfig(n_epochs=60, lambda_df=1.0)
    trainer = DFDGTrainer(model, cfg, device="cuda")
    history = trainer.fit(train_loader, val_loader)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
class DFDGConfig:
    """DFDG hyper-parameters."""
    n_epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    dropout: float = 0.3
    model_size: str = "medium"
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    # ---- DFDG-specific ----
    lambda_df: float = 1.0        # weight of energy-distance regularizer
    warmup_epochs: int = 5        # epochs before DFDG penalty is turned on
    normalize_features: bool = True  # L2-normalize features before ED
    patience: int = 15
    save_best: bool = True
    checkpoint_path: str = "checkpoints/dfdg_best.pt"


def energy_distance(
    A: torch.Tensor,
    B: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute the energy distance between two feature sets.

    Parameters
    ----------
    A : (n, d)  first set of feature vectors
    B : (m, d)  second set of feature vectors

    Returns
    -------
    scalar tensor : ED(A, B)

    Formula
    -------
    ED(A, B) = 2·E[||a - b||] - E[||a - a'||] - E[||b - b'||]

    where expectations are over all pairs of samples in each set.
    """

    def _mean_pairwise_dist(X: torch.Tensor) -> torch.Tensor:
        # ||xi - xj||_2 for all i≠j, averaged
        # Efficient via expansion: ||xi - xj||^2 = ||xi||^2 + ||xj||^2 - 2 xi·xj
        n = X.size(0)
        if n <= 1:
            return torch.tensor(0.0, device=X.device)
        dot = X @ X.t()                                 # (n, n)
        sq  = (X * X).sum(1, keepdim=True)              # (n, 1)
        dist2 = sq + sq.t() - 2 * dot                   # (n, n)
        dist2 = dist2.clamp(min=eps)
        dist  = dist2.sqrt()
        # exclude diagonal (self-distances)
        mask = 1 - torch.eye(n, device=X.device)
        return (dist * mask).sum() / (n * (n - 1))

    def _mean_cross_dist(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # E[||x - y||] for x ∈ A, y ∈ B
        n, m = X.size(0), Y.size(0)
        # (n, m) pairwise distance matrix
        diff = X.unsqueeze(1) - Y.unsqueeze(0)          # (n, m, d)
        dist = diff.norm(dim=-1).clamp(min=eps)          # (n, m)
        return dist.mean()

    cross = _mean_cross_dist(A, B)
    within_A = _mean_pairwise_dist(A)
    within_B = _mean_pairwise_dist(B)
    return 2 * cross - within_A - within_B


class DFDGTrainer:
    """
    DFDG trainer for smart-building IoT activity recognition.

    The energy-distance penalty is applied to encoder features
    (pre-classifier embeddings), pushing them toward a uniform,
    building-agnostic distribution.

    Parameters
    ----------
    model : SensorActivityModel
    config : DFDGConfig
    device : str | torch.device
    class_weights : torch.Tensor | None
    """

    METHOD_NAME = "DFDG"

    def __init__(
        self,
        model: SensorActivityModel,
        config: DFDGConfig,
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
        """Full DFDG training loop."""
        self.logger.start(self.config.n_epochs, self.model)
        self.logger.info(
            f"λ_df={self.config.lambda_df}, "
            f"warmup={self.config.warmup_epochs} epochs, "
            f"feature normalization={self.config.normalize_features}"
        )

        for epoch in range(1, self.config.n_epochs + 1):
            t0 = time.time()
            lam = self.config.lambda_df if epoch > self.config.warmup_epochs else 0.0
            train_metrics = self._train_epoch(train_loader, lam=lam)
            val_metrics   = self._eval_epoch(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            record = {
                "epoch":       epoch,
                "lambda_df":   lam,
                "train_loss":  train_metrics["loss"],
                "train_ce":    train_metrics["ce_loss"],
                "train_df":    train_metrics["df_loss"],
                "train_acc":   train_metrics["acc"],
                "train_f1":    train_metrics["f1"],
                "val_loss":    val_metrics["loss"],
                "val_acc":     val_metrics["acc"],
                "val_f1":      val_metrics["f1"],
                "time_s":      elapsed,
            }
            self.history.append(record)
            extra = f"CE={record['train_ce']:.4f} DF={record['train_df']:.4f}"
            self.logger.log_epoch(record, extra=extra)

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
        self, loader: DataLoader, lam: float
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = total_ce = total_df = 0.0
        all_preds, all_labels = [], []

        for x, y, _ in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            logits, features = self.model(x)  # features: (B, feat_dim)

            # Classification loss
            ce_loss = self.criterion(logits, y)

            # Distribution-Free alignment loss
            df_loss = torch.tensor(0.0, device=self.device)
            if lam > 0 and len(x) >= 4:
                df_loss = self._dfdg_loss(features)

            loss = ce_loss + lam * df_loss
            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            B = len(y)
            total_loss += loss.item()    * B
            total_ce   += ce_loss.item() * B
            total_df   += df_loss.item() * B
            all_preds.append(logits.argmax(1).cpu())
            all_labels.append(y.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        N = len(labels)
        acc, f1 = compute_metrics(preds, labels)
        return {
            "loss": total_loss / N, "ce_loss": total_ce / N,
            "df_loss": total_df / N, "acc": acc, "f1": f1,
        }

    def _dfdg_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Split batch randomly into halves A and B;
        compute energy distance between their feature sets.
        """
        if self.config.normalize_features:
            features = nn.functional.normalize(features, dim=-1)

        # Random split
        perm = torch.randperm(len(features), device=self.device)
        half = len(features) // 2
        A = features[perm[:half]]
        B = features[perm[half:2 * half]]
        return energy_distance(A, B)

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
        N = len(labels)

        if detailed:
            acc, f1, pcf1 = compute_metrics(preds, labels, per_class=True)
            return {"loss": total_loss / N, "acc": acc, "f1": f1, "per_class_f1": pcf1}
        acc, f1 = compute_metrics(preds, labels)
        return {"loss": total_loss / N, "acc": acc, "f1": f1}

    def _save_checkpoint(self, path: str) -> None:
        import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":  self.model.state_dict(),
            "best_val_f1":  self.best_val_f1,
            "best_epoch":   self.best_epoch,
            "config":       self.config,
        }, path)

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.logger.info(
            f"Loaded DFDG checkpoint (val F1={ckpt['best_val_f1']:.4f} "
            f"@ epoch {ckpt['best_epoch']})"
        )
