
# GDA‑1D: Domain Generalization for 1‑D Activity Recognition (SWA/SWAD, DFDG, TERM)

This repository provides **clean 1‑D adaptations** of three widely used generalization methods for *multivariate sensor time‑series* (smart‑building activity recognition, wearables, IoT):
- SWA (SWAD‑style dense averaging near convergence)
- Distribution‑Free Domain Generalization (DFDG)
- Tilted Empirical Risk Minimization (TERM)

All three run on the **same 1‑D backbone** (Conv1D + BiGRU) and expect data shaped `[N, T, C]` (N samples, T timesteps, C channels). Scripts report **Accuracy** and **F1-score** each epoch.

---

## TL;DR

```bash
# 1) Install
pip install -r requirements.txt  # see minimal list below

# 2) Prepare data
python - << 'PY'
import numpy as np
# X: [N, T, C] float32   y: [N] int64
# optional: domain: [N] int64 (not required)
np.savez('train.npz', X=X_train, y=y_train, domain=domains_train)
np.savez('val.npz',   X=X_val,   y=y_val,   domain=domains_val)
PY

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

Example:

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # pick your CUDA/CPU wheel
pip install scikit-learn numpy
```

---

## Data format

Provide two (or three) NPZ files:

- `train.npz` with `X:[N,T,C] float32`, `y:[N] int64`, optional `domain:[N] int64`
- `val.npz`   with the same keys
- optionally `test.npz` for a final hold‑out

Notes:
- Standardize each channel (z‑score per sensor) before saving to NPZ.
- Variable‑length sequences should be **padded and masked** beforehand (this repo assumes fixed‑length T).
- DFDG **does not require domain labels**; `domain` is kept only for analysis/stratified splits if you need it.

Quick CSV→NPZ sketch:

```python
import numpy as np, pandas as pd
# df columns: time, channel_0..channel_{C-1}, label
# build sliding windows of length T, stack to shape [N,T,C] and labels [N]
# ... your preprocessing here ...
np.savez('train.npz', X=X_train.astype(np.float32), y=y_train.astype(np.int64))
```

---

## Backbone: 1‑D Temporal Encoder

The default encoder is **Conv1D → BiGRU → mean pooling → MLP**:

- Conv1D layers: temporal feature extraction per channel with BN + ReLU + Dropout
- BiGRU: captures **long‑range** bidirectional context
- Temporal mean pooling over the sequence
- MLP classifier to logits

You can swap the encoder in `common/model_1d.py` (e.g., Temporal CNN, TST/Transformer, WaveNet) without touching the method logic.

---

## Method 1 — SWA (SWAD‑style dense averaging)

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

## Method 3 — TERM (Tilted ERM)

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
- Deterministic metrics (Accuracy, F1)
- Consistent backbone across methods

If you change the backbone, keep the **feature dimension** stable (or update the classifier accordingly).

---

## Troubleshooting

- Shapes: tensors must be `[B, C, T]` inside the model; the DataLoader returns `[C, T]` per sample.
- Scaling: per‑channel standardization is critical for smooth optimization.
- BN stats with SWA: ensure the BN refresh step runs before validation (script handles this).
- Imbalance: prefer F1-score; increase TERM `--tilt` moderately; consider class‑balanced sampling upstream.
- OOM: reduce batch size or GRU hidden size in `model_1d.py`.

---

