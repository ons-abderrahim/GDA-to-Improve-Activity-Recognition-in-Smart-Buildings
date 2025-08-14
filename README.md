
# GDA‑1D: Domain Generalization for 1‑D Activity Recognition (SWA/SWAD, DFDG, TERM)

This repository provides the codes for the Scalable Activity Recognition in Smart Buildings via Generalized
Domain Adaptation of IoT Sensor Data paper:
- SWA (SWAD‑style dense averaging near convergence)
- Distribution‑Free Domain Generalization (DFDG)
- Tilted Empirical Risk Minimization (TERM)

All three run on the **same 1‑D backbone** (Conv1D + BiGRU) and expect data shaped `[N, T, C]` (N samples, T timesteps, C channels). Scripts report **Accuracy** and **F1-score** each epoch.

---

## TL;DR

```bash
# 1) Install

# 2) Prepare data

# 3) Train one of the methods
python methods/erm_train.py  --train train.npz --val val.npz --classes 5
python methods/swa_train.py  --train train.npz --val val.npz --classes 5 --epochs 60 --swa_start 30
python methods/dfdg_train.py --train train.npz --val val.npz --classes 5 --lambda_df 1.0
python methods/term_train.py --train train.npz --val val.npz --classes 5 --tilt 3.0
```

---

## Why 1‑D adaptations?

The referenced methods were originally released for images or tabular settings. Activity recognition from building sensors or wearables is naturally **1‑D temporal**. This repo implements:
- a **temporal encoder** for `[B, C, T]` inputs (Conv1D → BiGRU → temporal pooling),
- consistent training/evaluation loops,
- plug‑and‑play regularizers for DFDG and TERM,
- SWA with **dense averaging** in late epochs (SWAD‑style behavior).

You can drop in your dataset as long as you provide an `.npz` with the expected arrays.

---

## Repository layout

```
gda_1d/
├── common/
│   ├── data.py          # NPZ loader → PyTorch DataLoader (returns [C, T] tensors)
│   ├── model_1d.py      # Conv1D + BiGRU backbone + MLP classifier
│   └── utils.py         # seeding + Accuracy/F1-score helpers
├── methods/
│   ├── erm_train.py     # plain ERM baseline (recommended sanity check)
│   ├── swa_train.py     # SWA with dense averaging after --swa_start
│   ├── dfdg_train.py    # DFDG with energy‑distance feature alignment
│   └── term_train.py    # TERM with tilted risk on per‑sample CE
└── README.md
```

---

## Installation

Minimal working environment:

```
python>=3.9
torch>=2.0
scikit-learn>=1.2
numpy>=1.24
```


---

## Method 1 — SWAD 

**What it is.** Stochastic Weight Averaging maintains a running average of weights during late training for a flatter, better‑generalizing solution. We implement **dense updates per step** after a chosen `--swa_start`, resembling the dense averaging used in SWAD.

**Key arguments**
- `--epochs`: total training epochs
- `--swa_start`: first epoch where averaging begins
- `--lr`: base LR; a cosine SWALR schedules LR once averaging starts
- `--batch`: batch size

**Run**

```bash
python methods/swa_train.py --train train.npz --val val.npz --classes 5 --epochs 60 --swa_start 30
```

**Tips**
- Start averaging when the ERM loss/val metrics plateau.
- Ensure BN statistics are **updated** on train data before evaluating SWA weights (the script does this with `update_bn`).

---

## Method 2 — DFDG (Distribution‑Free Domain Generalization)

**What it is.** Encourages domain‑invariant features **without domain labels** using a two‑sample statistic on latent features. Each batch is randomly split into two groups; we minimize **energy distance** between the groups:

\[ \mathcal{L}_{\text{DF}} = \operatorname{ED}(f(A), f(B)) = 2\,\mathbb{E}\|u-v\| - \mathbb{E}\|u-u'\| - \mathbb{E}\|v-v'\|. \]

The total loss is `CE + λ * L_DF`.

**Key arguments**
- `--lambda_df`: regularizer strength
- `--lr`, `--epochs`, `--batch` as usual

**Run**

```bash
python methods/dfdg_train.py --train train.npz --val val.npz --classes 5 --lambda_df 1.0
```

**Tips**
- Increase `--lambda_df` if you suspect domain overfitting; decrease if training becomes unstable.
- Larger batches make the two‑sample estimate less noisy.

---

## Method 3 — ERM 

**What it is.** Replaces mean loss with **tilted risk** to reweight hard/minority samples:

\[ \mathcal{L}_{\text{TERM}}(t) = \frac{1}{t}\log\, \mathbb{E}[\exp(t\, \ell_i)]. \]

As `t → 0`, TERM recovers mean CE; `t>0` emphasizes high‑loss examples; `t<0` emphasizes easy ones.

**Key arguments**
- `--tilt`: positive values focus on hard samples (e.g., 1–5)
- `--lr`, `--epochs`, `--batch` as usual

**Run**

```bash
python methods/term_train.py --train train.npz --val val.npz --classes 5 --tilt 3.0
```

**Tips**
- For imbalanced classes or rare activities, try `--tilt 2.0–5.0`.
- If overfitting spikes, reduce `--tilt` or add dropout/weight decay.

---

## Training and evaluation

Each script prints **Accuracy** and **F1-score** on the validation set after every epoch. Models are saved as `*_model.pt` in the working directory.

Simple evaluation snippet (example for TERM checkpoint):

```python
import torch, numpy as np
from common.data import make_loader
from common.model_1d import Model1D
from common.utils import metrics_from_logits

val_loader = make_loader('val.npz', batch_size=256, shuffle=False)
C = np.load('val.npz')["X"].shape[-1]
model = Model1D(C, num_classes=5)
model.load_state_dict(torch.load('term_model.pt', map_location='cpu'))
model.eval()

all_logits, all_y = [], []
with torch.no_grad():
    for x,y in val_loader:
        logits,_ = model(x)
        all_logits.append(logits); all_y.append(y)
logits = torch.cat(all_logits); y = torch.cat(all_y)
acc,f1 = metrics_from_logits(logits, y)
print(f"Val acc={acc:.4f}  F1-score={f1:.4f}")
```

---

## Recommended hyperparameters (starting points)

- Batch size: 128–256
- LR: 1e‑3 with Adam, weight decay 1e‑4
- Epochs: 40–80 depending on dataset size
- SWA: start at epoch 0.5–0.7× total
- DFDG: `lambda_df` in [0.5, 2.0]
- TERM: `tilt` in [1.0, 5.0] (smaller if unstable)

---

## Reproducibility

- Fixed seeds in all scripts
- Deterministic metrics (Accuracy, F1-score)


---
#Acknowledgments
The completion of this research was made possible
thanks to the Natural Sciences and Engineering Research
Council of Canada (NSERC)
