"""
Evaluation Pipeline
====================
Leave-One-Building-Out (LOBO) benchmark runner.
Evaluates all GDA methods on all buildings as test domains
and produces a structured results table.

Usage
-----
    python -m gda.evaluation.benchmark \
        --data data/synthetic_smart_building.npz \
        --methods erm swad dfdg term \
        --epochs 40 \
        --output results/lobo_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gda.data.loaders import load_npz, make_lobo_splits
from gda.data.transforms import NormalizeSensor
from gda.data.dataset import SensorDataset
from gda.models.backbone import build_model
from gda.methods import ERMTrainer, ERMConfig
from gda.methods import SWADTrainer, SWADConfig
from gda.methods import DFDGTrainer, DFDGConfig
from gda.methods import TERMTrainer, TERMConfig
from gda.utils import set_seed, get_device, compute_metrics, compare_methods, save_results_json
from gda.utils.logging import get_logger

logger = get_logger("gda.eval")


# ---------------------------------------------------------------------------
# Per-method training wrappers
# ---------------------------------------------------------------------------

def train_erm(model, loaders, device, epochs, class_weights):
    cfg = ERMConfig(n_epochs=epochs, save_best=True,
                    checkpoint_path="checkpoints/erm_best.pt")
    trainer = ERMTrainer(model, cfg, device=device, class_weights=class_weights)
    trainer.fit(loaders["train"], loaders["val"])
    return trainer.evaluate(loaders["test"])


def train_swad(model, loaders, device, epochs, class_weights):
    swa_start = max(5, epochs // 2)
    cfg = SWADConfig(n_epochs=epochs, swa_start_epoch=swa_start,
                     save_best=True, checkpoint_path="checkpoints/swad_best.pt")
    trainer = SWADTrainer(model, cfg, device=device, class_weights=class_weights)
    trainer.fit(loaders["train"], loaders["val"])
    return trainer.evaluate(loaders["test"], train_loader_for_bn=loaders["train"])


def train_dfdg(model, loaders, device, epochs, class_weights):
    cfg = DFDGConfig(n_epochs=epochs, lambda_df=1.0, save_best=True,
                     checkpoint_path="checkpoints/dfdg_best.pt")
    trainer = DFDGTrainer(model, cfg, device=device, class_weights=class_weights)
    trainer.fit(loaders["train"], loaders["val"])
    return trainer.evaluate(loaders["test"])


def train_term(model, loaders, device, epochs, class_weights):
    cfg = TERMConfig(n_epochs=epochs, tilt=3.0, save_best=True,
                     checkpoint_path="checkpoints/term_best.pt")
    trainer = TERMTrainer(model, cfg, device=device)
    trainer.fit(loaders["train"], loaders["val"])
    return trainer.evaluate(loaders["test"])


METHOD_REGISTRY = {
    "erm":  train_erm,
    "swad": train_swad,
    "dfdg": train_dfdg,
    "term": train_term,
}


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_lobo_benchmark(
    data_path: str,
    methods: List[str],
    n_epochs: int = 40,
    batch_size: int = 128,
    seed: int = 42,
    output_path: Optional[str] = None,
    model_size: str = "medium",
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Run full Leave-One-Building-Out benchmark.

    For each building domain D:
      • Train on all other buildings
      • Evaluate on building D

    Returns
    -------
    results_dict : {method → {domain → metrics, 'mean' → avg metrics}}
    """
    set_seed(seed)
    device = device or get_device()

    # Load data
    X, y, domains = load_npz(data_path)
    n_buildings = int(domains.max()) + 1
    n_classes   = int(y.max()) + 1
    n_channels  = X.shape[2]

    # Compute global normalization stats from all data
    normalizer = NormalizeSensor.fit(X)
    X_norm = normalizer.transform_dataset(X)

    logger.info(
        f"LOBO Benchmark | buildings={n_buildings}, classes={n_classes}, "
        f"channels={n_channels}, methods={methods}"
    )

    # Load activity names if available
    raw = np.load(data_path, allow_pickle=True)
    act_names = (
        list(raw["activity_names"])
        if "activity_names" in raw
        else [str(i) for i in range(n_classes)]
    )

    results = {m: {} for m in methods}

    for test_domain in range(n_buildings):
        logger.info(f"\n{'='*55}")
        logger.info(f"  LOBO round: test_domain = Building {test_domain}")
        logger.info(f"{'='*55}")

        loaders = make_lobo_splits(
            X_norm, y, domains,
            test_domain=test_domain,
            batch_size=batch_size,
            seed=seed,
        )

        # Class weights from training set
        train_ds = loaders["train"].dataset
        cw = train_ds.class_weights() if hasattr(train_ds, "class_weights") else None

        for method_name in methods:
            logger.info(f"  → Method: {method_name.upper()}")
            set_seed(seed)  # reset seed per method for fair comparison

            model = build_model(
                n_channels=n_channels,
                n_classes=n_classes,
                model_size=model_size,
            )

            t0 = time.time()
            metrics = METHOD_REGISTRY[method_name](
                model, loaders, device, n_epochs, cw
            )
            elapsed = time.time() - t0

            metrics["domain"] = test_domain
            metrics["time_s"] = elapsed
            results[method_name][f"building_{test_domain}"] = metrics

            logger.info(
                f"  [{method_name.upper()}] Building {test_domain} → "
                f"acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}  "
                f"({elapsed:.1f}s)"
            )

    # Compute mean metrics across buildings
    for method_name in methods:
        domain_results = [v for k, v in results[method_name].items() if k.startswith("building_")]
        mean_acc = np.mean([r["acc"] for r in domain_results])
        mean_f1  = np.mean([r["f1"]  for r in domain_results])
        results[method_name]["mean"] = {"acc": mean_acc, "f1": mean_f1}

    # Print summary
    logger.info("\n" + "="*55)
    logger.info("  LOBO BENCHMARK SUMMARY")
    logger.info("="*55)
    summary_for_compare = {
        m: results[m]["mean"]
        for m in methods
    }
    # Add dummy per_class_f1 if not present for display
    print(compare_methods(summary_for_compare, activity_names=act_names))

    if output_path:
        save_results_json(results, output_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LOBO benchmark for GDA smart-building activity recognition."
    )
    parser.add_argument("--data", required=True, help="Path to .npz dataset")
    parser.add_argument(
        "--methods", nargs="+", default=["erm", "swad", "dfdg", "term"],
        choices=list(METHOD_REGISTRY),
        help="Methods to benchmark",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_size", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    args = parser.parse_args()

    run_lobo_benchmark(
        data_path=args.data,
        methods=args.methods,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        model_size=args.model_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
