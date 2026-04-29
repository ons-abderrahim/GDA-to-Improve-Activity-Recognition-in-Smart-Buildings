#!/usr/bin/env python3
"""
Generate synthetic smart-building sensor dataset.

Usage
-----
    python scripts/generate_data.py \
        --buildings 5 \
        --samples 1000 \
        --window_size 50 \
        --output data/smart_building.npz
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gda.data.generator import SmartBuildingDataGenerator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--buildings",   type=int, default=5,   help="Number of building domains")
    p.add_argument("--samples",     type=int, default=1000,help="Samples per building")
    p.add_argument("--activities",  type=int, default=6,   help="Number of activity classes (2-6)")
    p.add_argument("--window_size", type=int, default=50,  help="Timesteps per window")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output",      type=str, default="data/smart_building.npz")
    p.add_argument("--balanced",    action="store_true",   help="Balanced class distribution")
    args = p.parse_args()

    gen = SmartBuildingDataGenerator(
        n_buildings=args.buildings,
        n_activities=args.activities,
        window_size=args.window_size,
        seed=args.seed,
    )
    X, y, domains = gen.generate(
        n_samples_per_building=args.samples,
        imbalance=not args.balanced,
    )
    print(gen.summary())
    gen.save(args.output)


if __name__ == "__main__":
    main()
