"""
Temporal Encoder Backbone for IoT Sensor Activity Recognition
=============================================================
Architecture:
  1. Conv1D feature extraction block  (captures local temporal patterns)
  2. Bidirectional GRU encoder        (captures long-range dependencies)
  3. Temporal attention pooling        (weighted aggregation over time)
  4. MLP classifier head

Input shape:  (B, C, T)  — batch, channels, timesteps  (channels-first)
Output:       (logits, features)  where features are the pre-softmax embedding

Design rationale
----------------
Smart-building IoT data is 1-D time-series with:
  - Mixed binary (PIR, door contact) and continuous channels
  - Short bursty events (door open/close) AND slow drifts (temperature)
  - Strong class imbalance (idle >> rare activities)

The Conv1D layers handle local patterns (bursts, pulses), the BiGRU
captures temporal context, and the attention pooling lets the model
focus on informative timesteps regardless of where they occur.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock1D(nn.Module):
    """
    1-D convolutional block:
      Conv1d → BatchNorm1d → GELU → Dropout

    Parameters
    ----------
    in_ch, out_ch : int   input / output channels
    kernel : int          kernel size (should be odd)
    stride : int          stride
    dropout : float       dropout probability
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=pad, bias=False)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # residual projection if dimensions change
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.drop(self.act(self.bn(self.conv(x))))
        return out + residual


class TemporalAttentionPooling(nn.Module):
    """
    Soft attention over the time dimension.
    Computes a weighted sum of GRU hidden states.

    Input:  (B, T, H)  hidden states from GRU
    Output: (B, H)     context vector
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: (B, T, H)
        scores = self.attn(h).squeeze(-1)           # (B, T)
        weights = F.softmax(scores, dim=-1)          # (B, T)
        context = (h * weights.unsqueeze(-1)).sum(1) # (B, H)
        return context, weights


class TemporalEncoder(nn.Module):
    """
    Conv1D + BiGRU temporal feature extractor.

    Parameters
    ----------
    n_channels : int
        Number of input sensor channels C.
    conv_channels : list[int]
        Output channels for each Conv block.
    gru_hidden : int
        GRU hidden size (each direction).
    gru_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability used throughout.
    """

    def __init__(
        self,
        n_channels: int,
        conv_channels: Tuple[int, ...] = (64, 128, 128),
        gru_hidden: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_dim = gru_hidden * 2  # bidirectional

        # ----- Conv stack -----
        conv_layers = []
        ch_in = n_channels
        for i, ch_out in enumerate(conv_channels):
            kernel = 7 if i == 0 else 5
            conv_layers.append(ConvBlock1D(ch_in, ch_out, kernel=kernel, dropout=dropout * 0.5))
            ch_in = ch_out
        self.conv_stack = nn.Sequential(*conv_layers)

        # ----- BiGRU -----
        self.gru = nn.GRU(
            input_size=ch_in,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ----- Temporal attention pooling -----
        self.attn_pool = TemporalAttentionPooling(self.feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C, T)  channels-first sensor windows

        Returns
        -------
        features : (B, feature_dim)  temporal context embedding
        attn_weights : (B, T')       attention weights (post-conv T)
        """
        # Conv feature extraction: (B, C, T) → (B, conv_ch, T')
        h = self.conv_stack(x)
        # Transpose for GRU: (B, T', conv_ch)
        h = h.permute(0, 2, 1)
        # BiGRU: (B, T', feature_dim)
        h, _ = self.gru(h)
        h = self.dropout(h)
        # Attention pooling: (B, feature_dim)
        features, attn_weights = self.attn_pool(h)
        return features, attn_weights


class SensorActivityModel(nn.Module):
    """
    Full model: TemporalEncoder + MLP classifier head.

    This is the main model used across all GDA methods (ERM, SWAD, DFDG).
    The encoder extracts domain-invariant temporal features; the classifier
    maps them to activity class logits.

    Parameters
    ----------
    n_channels : int
        Number of IoT sensor input channels.
    n_classes : int
        Number of activity classes.
    conv_channels : tuple
        Conv1D channel progression.
    gru_hidden : int
        GRU hidden units per direction.
    gru_layers : int
        Number of GRU layers.
    mlp_hidden : int
        MLP classifier hidden size.
    dropout : float
        Global dropout rate.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        conv_channels: Tuple[int, ...] = (64, 128, 128),
        gru_hidden: int = 128,
        gru_layers: int = 2,
        mlp_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = TemporalEncoder(
            n_channels=n_channels,
            conv_channels=conv_channels,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        feat_dim = self.encoder.feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C, T)

        Returns
        -------
        logits   : (B, n_classes)
        features : (B, feature_dim)   — pre-classifier embedding
        """
        features, _ = self.encoder(x)
        logits = self.classifier(features)
        return logits, features

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return feature embeddings only (for DFDG alignment)."""
        features, _ = self.encoder(x)
        return features

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SensorActivityModel("
            f"params={self.n_parameters():,}, "
            f"feature_dim={self.feature_dim})"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    n_channels: int,
    n_classes: int,
    model_size: str = "medium",
    dropout: float = 0.3,
) -> SensorActivityModel:
    """
    Convenience factory for common model sizes.

    model_size options
    ------------------
    'small'  : conv=(32,64),    gru=64,  mlp=128
    'medium' : conv=(64,128,128), gru=128, mlp=256   [default]
    'large'  : conv=(64,128,256,256), gru=256, mlp=512
    """
    configs = {
        "small":  dict(conv_channels=(32, 64),       gru_hidden=64,  gru_layers=1, mlp_hidden=128),
        "medium": dict(conv_channels=(64, 128, 128), gru_hidden=128, gru_layers=2, mlp_hidden=256),
        "large":  dict(conv_channels=(64, 128, 256, 256), gru_hidden=256, gru_layers=2, mlp_hidden=512),
    }
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs)}, got '{model_size}'")
    cfg = configs[model_size]
    return SensorActivityModel(
        n_channels=n_channels,
        n_classes=n_classes,
        dropout=dropout,
        **cfg,
    )
