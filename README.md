# GDA Smart Building – Generalized Domain Adaptation for IoT Activity Recognition

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="Status">
</p>

> **Official implementation of:**  
> *"Scalable Activity Recognition in Smart Buildings via Generalized Domain Adaptation of IoT Sensor Data"*  
> Published in *Journal of Building Engineering* (Elsevier, 2025)

---

## Overview

Smart buildings deploy IoT sensors (PIR motion detectors, door contacts, temperature, CO₂, power meters) to recognize occupant activities and enable intelligent automation — HVAC optimization, energy savings, security, and comfort. However, **domain shift** is inevitable: each building has different sensor placements, hardware, and occupant routines.

This repository provides the **first systematic study** of **Generalized Domain Adaptation (GDA)** methods for 1-D IoT time-series activity recognition. All three methods are fully adapted for sensor data and evaluated under a rigorous **Leave-One-Building-Out (LOBO)** protocol.

### Three GDA Methods, One Backbone

| Method | Key Idea | Building Transfer Strategy |
|--------|----------|--------------------------|
| **ERM** | Empirical Risk Minimization (baseline) | Pooled multi-building training |
| **SWAD** | Stochastic Weight Averaging Densely | Flat-minima averaging → broader generalization |
| **DFDG** | Distribution-Free Domain Generalization | Energy-distance alignment without domain labels |
| **TERM** | Tilted Empirical Risk Minimization | Upweights rare/hard activities (imbalance robustness) |

All methods share a single **1-D temporal backbone**: `Conv1D → BiGRU → Temporal Attention → MLP`.

---

## Architecture

```
Input: (B, C, T)  — batch × sensor channels × timesteps
         │
         ▼
┌──────────────────────────────────┐
│  Conv1D Feature Extractor        │
│  ConvBlock × 3 (64→128→128)     │  ← local patterns (bursts, pulses)
│  Residual + BN + GELU            │
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│  Bidirectional GRU               │
│  2 layers, hidden=128            │  ← temporal context
└───────────────┬──────────────────┘
                │
                ▼
┌──────────────────────────────────┐
│  Temporal Attention Pooling      │  ← focus on informative timesteps
└───────────────┬──────────────────┘
                │ features (B, 256)
                ▼
┌──────────────────────────────────┐
│  MLP Classifier                  │
│  Linear → LayerNorm → GELU       │
│  → Linear → logits (B, n_classes)│
└──────────────────────────────────┘
```

**Sensor channels (9 total):**
| # | Sensor | Type | Activity signal |
|---|--------|------|-----------------|
| 0 | PIR Kitchen | Binary | Cooking, Leaving |
| 1 | PIR Office | Binary | Working, Idle |
| 2 | PIR Hallway | Binary | Entering, Exercising |
| 3 | Door Contact | Binary | Leaving, Cooking |
| 4 | Temperature (°C) | Continuous | Cooking (+), Sleeping (–) |
| 5 | Humidity (%) | Continuous | Cooking (+), Exercising (+) |
| 6 | CO₂ (ppm) | Continuous | Occupancy proxy |
| 7 | Light (lux) | Continuous | Working (+), Sleeping (–) |
| 8 | Power (W) | Continuous | Cooking (+), Idle (–) |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/your-username/gda-smart-building.git
cd gda-smart-building
pip install -r requirements.txt
# Or install as editable package:
pip install -e .
```

### 2. Run the quick demo

```bash
python scripts/quickstart.py --epochs 30 --buildings 5 --device auto
```

This will:
1. Generate synthetic smart-building IoT sensor data (5 buildings, 6 activities)
2. Train all 4 methods (ERM, SWAD, DFDG, TERM)
3. Evaluate on the held-out test building
4. Print a comparison table

**Expected output:**
```
──────────────────────────────────────────────────────────
  Method     │      Acc │   Macro F1
──────────────────────────────────────────────────────────
  ERM        │   0.7231  │    0.6854
  SWAD       │   0.7589  │    0.7241
  DFDG       │   0.7412  │    0.7108
  TERM       │   0.7346  │    0.7193
──────────────────────────────────────────────────────────
```

---

## Repository Structure

```
gda_smart_building/
├── gda/
│   ├── data/
│   │   ├── generator.py       # Synthetic IoT sensor data generator
│   │   ├── dataset.py         # PyTorch Dataset wrapper (N,T,C) → (B,C,T)
│   │   ├── loaders.py         # NPZ loader + LOBO split factory
│   │   └── transforms.py      # Normalization + augmentation transforms
│   ├── models/
│   │   └── backbone.py        # Conv1D + BiGRU + Attention backbone
│   ├── methods/
│   │   ├── erm.py             # ERM baseline trainer
│   │   ├── swad.py            # SWAD dense weight averaging trainer
│   │   ├── dfdg.py            # DFDG energy-distance alignment trainer
│   │   └── term.py            # TERM tilted risk trainer
│   ├── evaluation/
│   │   └── benchmark.py       # LOBO benchmark runner
│   └── utils/
│       ├── metrics.py         # Accuracy, F1, confusion matrix, comparison tables
│       ├── logging.py         # Structured training logger
│       └── misc.py            # Seeds, device, checkpoint helpers
│
├── scripts/
│   ├── quickstart.py          # End-to-end demo script
│   ├── generate_data.py       # Data generation CLI
│   └── run_benchmark.py       # Full LOBO benchmark CLI
│
├── tests/
│   ├── test_data.py           # Data module unit tests
│   ├── test_model.py          # Backbone unit tests
│   └── test_methods.py        # Method smoke tests + loss function tests
│
├── configs/                   # YAML config files (coming soon)
├── notebooks/                 # Jupyter exploration notebooks (coming soon)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Usage

### Generate Your Own Dataset

```python
from gda.data.generator import SmartBuildingDataGenerator

gen = SmartBuildingDataGenerator(
    n_buildings=8,       # number of building domains
    n_activities=6,      # Idle, Cooking, Working, Exercising, Sleeping, Leaving
    window_size=50,      # timesteps per sensor window
    seed=42,
)
X, y, domains = gen.generate(n_samples_per_building=1000, imbalance=True)
# X: (N, T, C) = (8000, 50, 9)
gen.save("data/my_dataset.npz")
print(gen.summary())
```

### Load a Real Dataset

```python
from gda.data.loaders import load_npz

X, y, domains = load_npz("data/my_dataset.npz")
```

For CASAS-format CSV data:
```python
from gda.data.loaders import load_casas_style

X, y, domains = load_casas_style(
    x_path="sensors.csv",
    y_path="labels.csv",
    window_size=50,
    stride=25,
)
```

### Leave-One-Building-Out Splits

```python
from gda.data.loaders import make_lobo_splits

loaders = make_lobo_splits(
    X, y, domains,
    test_domain=2,        # hold out building 2 as unseen test domain
    batch_size=128,
)
# loaders: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
```

### Train Each Method

#### ERM (Baseline)
```python
from gda.models.backbone import build_model
from gda.methods import ERMTrainer, ERMConfig

model = build_model(n_channels=9, n_classes=6, model_size="medium")
config = ERMConfig(n_epochs=60, lr=1e-3, batch_size=128)
trainer = ERMTrainer(model, config, device="cuda")
trainer.fit(loaders["train"], loaders["val"])
metrics = trainer.evaluate(loaders["test"])
# {'acc': 0.7231, 'f1': 0.6854, 'per_class_f1': [...]}
```

#### SWAD
```python
from gda.methods import SWADTrainer, SWADConfig

model = build_model(n_channels=9, n_classes=6)
config = SWADConfig(
    n_epochs=60,
    swa_start_epoch=30,    # start dense averaging at epoch 30
    dense=True,            # update average every step (recommended)
)
trainer = SWADTrainer(model, config, device="cuda")
trainer.fit(loaders["train"], loaders["val"])
metrics = trainer.evaluate(loaders["test"], train_loader_for_bn=loaders["train"])
```

#### DFDG
```python
from gda.methods import DFDGTrainer, DFDGConfig

model = build_model(n_channels=9, n_classes=6)
config = DFDGConfig(
    n_epochs=60,
    lambda_df=1.0,         # energy-distance penalty weight
    warmup_epochs=5,       # CE-only warmup before penalty activates
)
trainer = DFDGTrainer(model, config, device="cuda")
trainer.fit(loaders["train"], loaders["val"])
metrics = trainer.evaluate(loaders["test"])
```

#### TERM
```python
from gda.methods import TERMTrainer, TERMConfig

model = build_model(n_channels=9, n_classes=6)
config = TERMConfig(
    n_epochs=60,
    tilt=3.0,              # t > 0: focus on rare/hard activities
)
trainer = TERMTrainer(model, config, device="cuda")
trainer.fit(loaders["train"], loaders["val"])
metrics = trainer.evaluate(loaders["test"])
```

### Full LOBO Benchmark

```bash
# Generate data first
python scripts/generate_data.py --buildings 8 --samples 1000 --output data/smart_building.npz

# Run full benchmark (all methods × all buildings)
python scripts/run_benchmark.py \
    --data data/smart_building.npz \
    --methods erm swad dfdg term \
    --epochs 60 \
    --output results/lobo_results.json
```

---

## Methods – Technical Details

### ERM (Empirical Risk Minimization)

The standard baseline: train on all source buildings, minimize mean cross-entropy.

$$\mathcal{L}_{\text{ERM}} = \frac{1}{N} \sum_{i=1}^N \ell(f(x_i), y_i)$$

### SWAD (Stochastic Weight Averaging Densely)

Finds flatter loss minima by averaging weights during late training:

$$\bar{\theta}_t = \frac{1}{t - t_{\text{start}} + 1} \sum_{k=t_{\text{start}}}^{t} \theta_k$$

**Dense** mode updates $\bar\theta$ at every gradient step (not just every epoch), producing smoother, more generalized minima. Batch normalization statistics are recomputed on training data after averaging.

### DFDG (Distribution-Free Domain Generalization)

No domain labels needed. Splits each batch randomly into halves A, B and minimizes their **energy distance** in feature space:

$$\mathcal{L}_{\text{DF}} = 2\,\mathbb{E}[\|u - v\|] - \mathbb{E}[\|u - u'\|] - \mathbb{E}[\|v - v'\|]$$

Total loss: $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{df}} \cdot \mathcal{L}_{\text{DF}}$

This pushes encoder features toward a building-agnostic, uniform distribution.

### TERM (Tilted Empirical Risk Minimization)

Replaces mean CE with a log-sum-exp tilted risk:

$$\mathcal{L}_{\text{TERM}}(t) = \frac{1}{t} \log\!\left[\frac{1}{N}\sum_{i=1}^N \exp(t\,\ell_i)\right]$$

| $t$ | Effect |
|-----|--------|
| $t = 0$ | Recovers mean CE |
| $t > 0$ | Emphasizes **high-loss** (rare activities: Exercising, Leaving) |
| $t < 0$ | Emphasizes **easy** samples |

Recommended: $t \in [2, 5]$ for imbalanced smart-building datasets.

---

## Hyper-parameter Guide

| Parameter | Default | Recommended Range | Notes |
|-----------|---------|-------------------|-------|
| `lr` | 1e-3 | 5e-4 – 2e-3 | AdamW with cosine schedule |
| `weight_decay` | 1e-4 | 1e-5 – 1e-3 | Regularization |
| `batch_size` | 128 | 64 – 256 | Larger → better DFDG estimates |
| `n_epochs` | 60 | 40 – 100 | Use early stopping |
| `swa_start_epoch` | 30 | 0.5–0.7 × epochs | Start after plateau |
| `lambda_df` | 1.0 | 0.5 – 2.0 | Increase if domain gap is large |
| `tilt` | 3.0 | 1.0 – 5.0 | Reduce if training is unstable |
| `dropout` | 0.3 | 0.1 – 0.5 | Increase for small datasets |

---

## Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
pytest

# Run specific test module
pytest tests/test_methods.py -v

# Run with coverage
pip install pytest-cov
pytest --cov=gda --cov-report=term-missing
```

---

## Using Your Own Real-World Data

The framework is designed to work with any dataset formatted as `(N, T, C)` windows. To use your own data:

1. **Sliding window segmentation:** Use `load_casas_style()` for CSV data, or implement your own windowing.
2. **Format as NPZ:** Save with `np.savez_compressed(path, X=X, y=y, domains=domains)`.
3. **Load and train:**
```python
from gda.data.loaders import load_npz, make_lobo_splits
from gda.data.transforms import NormalizeSensor

X, y, domains = load_npz("your_data.npz")
norm = NormalizeSensor.fit(X)
X = norm.transform_dataset(X)
loaders = make_lobo_splits(X, y, domains, test_domain=0)
```

**Compatible public datasets:**
- [CASAS Smart Home](http://casas.wsu.edu/datasets/)
- [OPPORTUNITY Activity Recognition](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
- [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{abderrahim2026scalable,
  title={Scalable activity recognition in smart buildings via generalized domain adaptation of IoT sensor data},
  author={Abderrahim, Ons and Dridi, Jawher and Amayri, Manar and Bouguila, Nizar},
  journal={Energy and Buildings},
  volume={351},
  pages={116692},
  year={2026},
  publisher={Elsevier}
}
```

---

## Related Work

- **SWAD:** Cha et al., "SWAD: Domain Generalization by Seeking Flat Minima," NeurIPS 2021
- **DFDG:** Jin et al., "Domain-Free Domain Generalization," 2021
- **TERM:** Li et al., "Tilted Empirical Risk Minimization," ICLR 2021
- **DomainBed:** Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization," ICLR 2021

---

## Acknowledgments

This research was supported by the **Natural Sciences and Engineering Research Council of Canada (NSERC)**.
