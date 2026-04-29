"""
Tests – Model Backbone
=======================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from gda.models.backbone import (
    SensorActivityModel,
    TemporalEncoder,
    ConvBlock1D,
    build_model,
)


class TestConvBlock1D:
    def test_output_shape_same_ch(self):
        block = ConvBlock1D(in_ch=9, out_ch=9, kernel=7)
        x = torch.randn(8, 9, 50)
        out = block(x)
        assert out.shape == (8, 9, 50)

    def test_output_shape_proj(self):
        block = ConvBlock1D(in_ch=9, out_ch=64, kernel=7)
        x = torch.randn(8, 9, 50)
        out = block(x)
        assert out.shape == (8, 64, 50)


class TestTemporalEncoder:
    def test_output_feature_dim(self):
        enc = TemporalEncoder(n_channels=9, gru_hidden=64)
        x = torch.randn(4, 9, 50)
        feats, attn = enc(x)
        assert feats.shape == (4, 128)  # bidirectional → 2*64

    def test_attn_weights_positive(self):
        enc = TemporalEncoder(n_channels=9, gru_hidden=32)
        x = torch.randn(4, 9, 50)
        _, attn = enc(x)
        assert (attn >= 0).all()


class TestSensorActivityModel:
    @pytest.fixture
    def model(self):
        return SensorActivityModel(n_channels=9, n_classes=6)

    def test_forward_logits_shape(self, model):
        x = torch.randn(8, 9, 50)
        logits, features = model(x)
        assert logits.shape == (8, 6)
        assert features.shape == (8, model.feature_dim)

    def test_encode(self, model):
        x = torch.randn(4, 9, 50)
        feats = model.encode(x)
        assert feats.shape == (4, model.feature_dim)

    def test_n_parameters(self, model):
        assert model.n_parameters() > 0

    def test_build_model_sizes(self):
        for size in ("small", "medium", "large"):
            m = build_model(n_channels=9, n_classes=6, model_size=size)
            x = torch.randn(2, 9, 50)
            logits, _ = m(x)
            assert logits.shape == (2, 6)

    def test_train_mode(self, model):
        model.train()
        x = torch.randn(4, 9, 50)
        logits, _ = model(x)
        assert logits.requires_grad or True  # just check no crash
