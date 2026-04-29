"""
Tests – GDA Methods (Smoke Tests)
===================================
Run tiny training loops to verify all methods execute without error.
Uses small model + few epochs + tiny dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from gda.data.generator import SmartBuildingDataGenerator
from gda.data.loaders import make_lobo_splits
from gda.data.transforms import NormalizeSensor
from gda.models.backbone import build_model
from gda.methods import ERMTrainer, ERMConfig
from gda.methods import SWADTrainer, SWADConfig
from gda.methods import DFDGTrainer, DFDGConfig
from gda.methods import TERMTrainer, TERMConfig
from gda.utils import set_seed


@pytest.fixture(scope="module")
def tiny_loaders():
    set_seed(0)
    gen = SmartBuildingDataGenerator(n_buildings=3, n_activities=4, window_size=20, seed=0)
    X, y, domains = gen.generate(n_samples_per_building=60, imbalance=False)
    norm = NormalizeSensor.fit(X)
    X = norm.transform_dataset(X)
    loaders = make_lobo_splits(X, y, domains, test_domain=2, batch_size=16, seed=0)
    return loaders


@pytest.fixture(scope="module")
def tiny_model_kwargs():
    return dict(n_channels=9, n_classes=4, model_size="small")


DEVICE = torch.device("cpu")
EPOCHS = 3


class TestERMSmoke:
    def test_fit_and_evaluate(self, tiny_loaders, tiny_model_kwargs):
        set_seed(0)
        model = build_model(**tiny_model_kwargs)
        cfg = ERMConfig(n_epochs=EPOCHS, patience=100, save_best=False)
        trainer = ERMTrainer(model, cfg, device=DEVICE)
        history = trainer.fit(tiny_loaders["train"], tiny_loaders["val"])
        assert len(history) == EPOCHS
        metrics = trainer.evaluate(tiny_loaders["test"], load_best=False)
        assert 0.0 <= metrics["acc"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0


class TestSWADSmoke:
    def test_fit_and_evaluate(self, tiny_loaders, tiny_model_kwargs):
        set_seed(0)
        model = build_model(**tiny_model_kwargs)
        cfg = SWADConfig(n_epochs=EPOCHS, swa_start_epoch=2, patience=100, save_best=False)
        trainer = SWADTrainer(model, cfg, device=DEVICE)
        history = trainer.fit(tiny_loaders["train"], tiny_loaders["val"])
        assert len(history) == EPOCHS
        metrics = trainer.evaluate(tiny_loaders["test"], load_best=False)
        assert 0.0 <= metrics["acc"] <= 1.0


class TestDFDGSmoke:
    def test_fit_and_evaluate(self, tiny_loaders, tiny_model_kwargs):
        set_seed(0)
        model = build_model(**tiny_model_kwargs)
        cfg = DFDGConfig(n_epochs=EPOCHS, lambda_df=0.5, patience=100, save_best=False)
        trainer = DFDGTrainer(model, cfg, device=DEVICE)
        history = trainer.fit(tiny_loaders["train"], tiny_loaders["val"])
        assert len(history) == EPOCHS
        metrics = trainer.evaluate(tiny_loaders["test"], load_best=False)
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_energy_distance_positive(self):
        from gda.methods.dfdg import energy_distance
        A = torch.randn(8, 32)
        B = torch.randn(8, 32)
        ed = energy_distance(A, B)
        assert isinstance(ed.item(), float)

    def test_energy_distance_self_near_zero(self):
        from gda.methods.dfdg import energy_distance
        A = torch.randn(10, 32)
        ed = energy_distance(A, A)
        # ED(A, A) should be ≤ 0 (equal distributions)
        assert ed.item() <= 1e-3


class TestTERMSmoke:
    def test_fit_and_evaluate(self, tiny_loaders, tiny_model_kwargs):
        set_seed(0)
        model = build_model(**tiny_model_kwargs)
        cfg = TERMConfig(n_epochs=EPOCHS, tilt=2.0, patience=100, save_best=False)
        trainer = TERMTrainer(model, cfg, device=DEVICE)
        history = trainer.fit(tiny_loaders["train"], tiny_loaders["val"])
        assert len(history) == EPOCHS
        metrics = trainer.evaluate(tiny_loaders["test"], load_best=False)
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_tilted_loss_recovers_mean_at_zero(self):
        from gda.methods.term import tilted_loss
        losses = torch.tensor([0.5, 1.0, 1.5, 2.0])
        tl = tilted_loss(losses, t=0.0)
        mean = losses.mean()
        torch.testing.assert_close(tl, mean, atol=1e-4, rtol=0)

    def test_tilted_loss_t_positive_geq_mean(self):
        from gda.methods.term import tilted_loss
        losses = torch.tensor([0.1, 0.5, 2.0, 5.0])
        mean_loss = losses.mean().item()
        tl = tilted_loss(losses, t=3.0).item()
        # Tilted (t>0) should emphasize hard samples → loss ≥ mean
        assert tl >= mean_loss - 1e-4
