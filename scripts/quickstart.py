#!/usr/bin/env python3
"""
Quick Start Script
==================
Generates synthetic smart-building sensor data, trains all four GDA methods
(ERM, SWAD, DFDG, TERM), runs LOBO evaluation, and prints a results table.

Usage
-----
    python scripts/quickstart.py [--epochs N] [--buildings N] [--device cpu|cuda]

Expected runtime (medium model, 5 buildings, 40 epochs each):
    CPU  : ~5-10 minutes
    GPU  : ~1-2 minutes
"""

import argparse
import sys
from pathlib import Path

# Make sure the package root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from gda.data.generator import SmartBuildingDataGenerator
from gda.data.loaders import make_lobo_splits, load_npz
from gda.data.transforms import NormalizeSensor
from gda.models.backbone import build_model
from gda.methods import ERMTrainer, ERMConfig
from gda.methods import SWADTrainer, SWADConfig
from gda.methods import DFDGTrainer, DFDGConfig
from gda.methods import TERMTrainer, TERMConfig
from gda.utils import set_seed, get_device, compare_methods
from gda.utils.logging import get_logger

logger = get_logger("quickstart")


def parse_args():
    p = argparse.ArgumentParser(description="GDA Smart Building Quick Start")
    p.add_argument("--epochs",    type=int, default=30,    help="Training epochs per method")
    p.add_argument("--buildings", type=int, default=5,     help="Number of simulated buildings")
    p.add_argument("--samples",   type=int, default=400,   help="Samples per building")
    p.add_argument("--batch",     type=int, default=64,    help="Batch size")
    p.add_argument("--seed",      type=int, default=42,    help="Random seed")
    p.add_argument("--device",    type=str, default="auto",help="'auto', 'cpu', 'cuda', or 'mps'")
    p.add_argument("--model_size",type=str, default="small",choices=["small","medium","large"])
    p.add_argument("--test_domain",type=int,default=0,     help="Building to use as test domain")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.device == "auto":
        device = get_device()
    else:
        import torch; device = torch.device(args.device)

    # ----------------------------------------------------------------
    # 1. Generate synthetic smart-building data
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  STEP 1: Generating synthetic IoT sensor data")
    print("="*60)
    gen = SmartBuildingDataGenerator(
        n_buildings=args.buildings,
        n_activities=6,
        window_size=50,
        seed=args.seed,
    )
    X, y, domains = gen.generate(n_samples_per_building=args.samples, imbalance=True)
    print(gen.summary())

    # Save generated data
    Path("data").mkdir(exist_ok=True)
    gen.save("data/synthetic_smart_building.npz")

    # ----------------------------------------------------------------
    # 2. Normalize and create LOBO splits
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print(f"  STEP 2: LOBO split (test building = {args.test_domain})")
    print("="*60)
    normalizer = NormalizeSensor.fit(X)
    X_norm = normalizer.transform_dataset(X)

    loaders = make_lobo_splits(
        X_norm, y, domains,
        test_domain=args.test_domain,
        batch_size=args.batch,
        seed=args.seed,
    )

    n_channels = X.shape[2]
    n_classes  = int(y.max()) + 1
    cw = loaders["train"].dataset.class_weights()

    # ----------------------------------------------------------------
    # 3. Train and evaluate all methods
    # ----------------------------------------------------------------
    all_results = {}

    # --- ERM ---
    print("\n" + "="*60)
    print("  STEP 3a: ERM – Empirical Risk Minimization")
    print("="*60)
    set_seed(args.seed)
    model = build_model(n_channels, n_classes, model_size=args.model_size)
    cfg = ERMConfig(n_epochs=args.epochs, batch_size=args.batch,
                    save_best=True, checkpoint_path="checkpoints/erm.pt")
    trainer = ERMTrainer(model, cfg, device=device, class_weights=cw)
    trainer.fit(loaders["train"], loaders["val"])
    all_results["ERM"] = trainer.evaluate(loaders["test"])

    # --- SWAD ---
    print("\n" + "="*60)
    print("  STEP 3b: SWAD – Stochastic Weight Averaging Densely")
    print("="*60)
    set_seed(args.seed)
    model = build_model(n_channels, n_classes, model_size=args.model_size)
    swa_start = max(5, args.epochs // 2)
    cfg = SWADConfig(n_epochs=args.epochs, swa_start_epoch=swa_start, batch_size=args.batch,
                     save_best=True, checkpoint_path="checkpoints/swad.pt")
    trainer = SWADTrainer(model, cfg, device=device, class_weights=cw)
    trainer.fit(loaders["train"], loaders["val"])
    all_results["SWAD"] = trainer.evaluate(loaders["test"], train_loader_for_bn=loaders["train"])

    # --- DFDG ---
    print("\n" + "="*60)
    print("  STEP 3c: DFDG – Distribution-Free Domain Generalization")
    print("="*60)
    set_seed(args.seed)
    model = build_model(n_channels, n_classes, model_size=args.model_size)
    cfg = DFDGConfig(n_epochs=args.epochs, lambda_df=1.0, batch_size=args.batch,
                     save_best=True, checkpoint_path="checkpoints/dfdg.pt")
    trainer = DFDGTrainer(model, cfg, device=device, class_weights=cw)
    trainer.fit(loaders["train"], loaders["val"])
    all_results["DFDG"] = trainer.evaluate(loaders["test"])

    # --- TERM ---
    print("\n" + "="*60)
    print("  STEP 3d: TERM – Tilted Empirical Risk Minimization")
    print("="*60)
    set_seed(args.seed)
    model = build_model(n_channels, n_classes, model_size=args.model_size)
    cfg = TERMConfig(n_epochs=args.epochs, tilt=3.0, batch_size=args.batch,
                     save_best=True, checkpoint_path="checkpoints/term.pt")
    trainer = TERMTrainer(model, cfg, device=device)
    trainer.fit(loaders["train"], loaders["val"])
    all_results["TERM"] = trainer.evaluate(loaders["test"])

    # ----------------------------------------------------------------
    # 4. Results table
    # ----------------------------------------------------------------
    print("\n" + "="*60)
    print("  STEP 4: Results Comparison")
    print("="*60)
    print(f"  Test domain: Building {args.test_domain}  |  Model: {args.model_size}")
    print(compare_methods(all_results, activity_names=gen.get_activity_names()))

    # Save results
    from gda.utils import save_results_json
    save_results_json(all_results, "results/quickstart_results.json")
    print("\n  Results saved to results/quickstart_results.json")
    print("  Run 'python scripts/quickstart.py --help' for more options.\n")


if __name__ == "__main__":
    main()
