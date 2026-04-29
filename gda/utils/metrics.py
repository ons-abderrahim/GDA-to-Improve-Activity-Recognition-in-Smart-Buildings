"""
Metrics
========
Evaluation metrics for smart-building activity recognition.
All functions operate on CPU torch.Tensors or numpy arrays.
"""

from __future__ import annotations

from typing import Tuple, Optional, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    per_class: bool = False,
    average: str = "macro",
) -> Union[Tuple[float, float], Tuple[float, float, np.ndarray]]:
    """
    Compute accuracy and macro-averaged F1-score.

    Parameters
    ----------
    preds  : predicted class indices
    labels : ground-truth class indices
    per_class : if True, also return per-class F1 array
    average : sklearn averaging mode for F1 ('macro', 'weighted', 'micro')

    Returns
    -------
    (acc, f1)             if per_class=False
    (acc, f1, per_cls_f1) if per_class=True
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    acc = float(accuracy_score(labels, preds))
    f1  = float(f1_score(labels, preds, average=average, zero_division=0))

    if per_class:
        pcf1 = f1_score(labels, preds, average=None, zero_division=0)
        return acc, f1, pcf1

    return acc, f1


def classification_summary(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    activity_names: Optional[list] = None,
) -> str:
    """
    Return a formatted sklearn classification report.

    Parameters
    ----------
    activity_names : list of strings, one per class
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    target_names = activity_names if activity_names else None
    return classification_report(
        labels, preds,
        target_names=target_names,
        zero_division=0,
        digits=4,
    )


def confusion_matrix_array(
    preds: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Returns
    -------
    np.ndarray of shape (n_classes, n_classes)
    normalized to row fractions if normalize=True.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()

    cm = confusion_matrix(labels, preds)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm.astype(float) / np.maximum(row_sums, 1)
    return cm


def compare_methods(results: dict, activity_names: Optional[list] = None) -> str:
    """
    Format a side-by-side comparison table for multiple methods.

    Parameters
    ----------
    results : dict mapping method_name → dict with 'acc', 'f1', 'per_class_f1'

    Returns
    -------
    Formatted string table
    """
    lines = []
    sep = "─" * 60
    lines.append(sep)
    lines.append(f"  {'Method':<10} │ {'Acc':>8} │ {'Macro F1':>10}")
    lines.append(sep)
    for method, m in results.items():
        lines.append(
            f"  {method:<10} │ {m['acc']:>7.4f}  │ {m['f1']:>9.4f}"
        )
    lines.append(sep)

    if activity_names and all("per_class_f1" in m for m in results.values()):
        lines.append("\n  Per-class F1:")
        header = f"  {'Activity':<22}"
        for method in results:
            header += f" │ {method:>8}"
        lines.append(header)
        lines.append(sep)
        n_classes = len(next(iter(results.values()))["per_class_f1"])
        for i in range(n_classes):
            name = activity_names[i] if i < len(activity_names) else str(i)
            row = f"  {name:<22}"
            for m in results.values():
                row += f" │ {m['per_class_f1'][i]:>8.4f}"
            lines.append(row)
        lines.append(sep)

    return "\n".join(lines)
