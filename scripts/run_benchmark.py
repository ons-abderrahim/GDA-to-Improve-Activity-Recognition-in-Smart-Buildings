#!/usr/bin/env python3
"""
Run full Leave-One-Building-Out (LOBO) benchmark.

Usage
-----
    python scripts/run_benchmark.py \
        --data data/smart_building.npz \
        --methods erm swad dfdg term \
        --epochs 60 \
        --output results/lobo_results.json
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gda.evaluation.benchmark import run_lobo_benchmark
from gda.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--methods",    nargs="+", default=["erm", "swad", "dfdg", "term"])
    p.add_argument("--epochs",     type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--model_size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--output",     default="results/lobo_results.json")
    p.add_argument("--device",     default="auto")
    args = p.parse_args()

    import torch
    device = get_device() if args.device == "auto" else torch.device(args.device)

    run_lobo_benchmark(
        data_path=args.data,
        methods=args.methods,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        model_size=args.model_size,
        output_path=args.output,
        device=device,
    )


if __name__ == "__main__":
    main()
