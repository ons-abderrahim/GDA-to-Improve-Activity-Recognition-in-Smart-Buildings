"""
Tests – Data Module
====================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from gda.data.generator import SmartBuildingDataGenerator, N_SENSORS, ACTIVITY_NAMES
from gda.data.dataset import SensorDataset, make_dataloader
from gda.data.loaders import make_lobo_splits, make_train_val_test_split
from gda.data.transforms import NormalizeSensor, AddSensorNoise, RandomWindowDrop


# ----------------------------------------------------------------
# Generator tests
# ----------------------------------------------------------------

class TestSmartBuildingDataGenerator:
    def test_output_shapes(self):
        gen = SmartBuildingDataGenerator(n_buildings=3, n_activities=4, window_size=30, seed=0)
        X, y, domains = gen.generate(n_samples_per_building=50)
        assert X.shape == (150, 30, N_SENSORS)
        assert y.shape == (150,)
        assert domains.shape == (150,)

    def test_label_range(self):
        gen = SmartBuildingDataGenerator(n_buildings=2, n_activities=6, seed=1)
        X, y, domains = gen.generate(n_samples_per_building=100)
        assert y.min() >= 0
        assert y.max() <= 5

    def test_domain_range(self):
        gen = SmartBuildingDataGenerator(n_buildings=4, seed=2)
        X, y, domains = gen.generate(n_samples_per_building=50)
        assert set(domains.tolist()).issubset(set(range(4)))

    def test_dtype(self):
        gen = SmartBuildingDataGenerator(n_buildings=2, seed=3)
        X, y, domains = gen.generate(n_samples_per_building=30)
        assert X.dtype == np.float32
        assert y.dtype == np.int64
        assert domains.dtype == np.int64

    def test_save_load(self, tmp_path):
        gen = SmartBuildingDataGenerator(n_buildings=2, seed=4)
        X, y, domains = gen.generate(n_samples_per_building=50)
        out = tmp_path / "test.npz"
        gen.save(out)
        data = np.load(out, allow_pickle=True)
        assert "X" in data and "y" in data and "domains" in data
        assert np.allclose(data["X"], X)

    def test_invalid_n_activities(self):
        with pytest.raises(ValueError):
            SmartBuildingDataGenerator(n_activities=7)


# ----------------------------------------------------------------
# Dataset tests
# ----------------------------------------------------------------

class TestSensorDataset:
    @pytest.fixture
    def sample_data(self):
        X = np.random.rand(100, 50, 9).astype(np.float32)
        y = np.random.randint(0, 6, 100).astype(np.int64)
        d = np.random.randint(0, 3, 100).astype(np.int64)
        return X, y, d

    def test_length(self, sample_data):
        X, y, d = sample_data
        ds = SensorDataset(X, y, d)
        assert len(ds) == 100

    def test_item_shapes(self, sample_data):
        X, y, d = sample_data
        ds = SensorDataset(X, y, d)
        x_t, y_t, d_t = ds[0]
        assert x_t.shape == (9, 50)   # channels-first
        assert y_t.shape == ()
        assert d_t.shape == ()

    def test_dataloader(self, sample_data):
        X, y, d = sample_data
        ds = SensorDataset(X, y, d)
        dl = make_dataloader(ds, batch_size=16, shuffle=False)
        batch = next(iter(dl))
        x, y_b, d_b = batch
        assert x.shape == (16, 9, 50)

    def test_class_weights_sum(self, sample_data):
        X, y, d = sample_data
        ds = SensorDataset(X, y, d)
        cw = ds.class_weights()
        assert cw.shape == (6,)
        assert all(cw >= 0)


# ----------------------------------------------------------------
# Transforms tests
# ----------------------------------------------------------------

class TestTransforms:
    def test_normalize_zero_mean(self):
        X = np.random.rand(100, 50, 9).astype(np.float32)
        norm = NormalizeSensor.fit(X)
        X_n = norm.transform_dataset(X)
        # Each channel should have ~zero mean across N*T
        means = X_n.reshape(-1, 9).mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=0.01)

    def test_add_noise_shape(self):
        window = np.zeros((50, 9), dtype=np.float32)
        aug = AddSensorNoise(std=0.1)
        out = aug(window)
        assert out.shape == window.shape
        # Binary channels (0-3) should remain zero
        np.testing.assert_array_equal(out[:, :4], 0)

    def test_random_drop(self):
        window = np.ones((50, 9), dtype=np.float32)
        aug = RandomWindowDrop(drop_prob=1.0)  # always drop
        out = aug(window)
        # All channels should be zero
        np.testing.assert_array_equal(out, 0)


# ----------------------------------------------------------------
# Splits tests
# ----------------------------------------------------------------

class TestSplits:
    @pytest.fixture
    def data(self):
        n = 300
        X = np.random.rand(n, 50, 9).astype(np.float32)
        y = np.tile(np.arange(6), n // 6).astype(np.int64)
        d = np.repeat(np.arange(5), n // 5).astype(np.int64)
        return X, y, d

    def test_lobo_no_overlap(self, data):
        X, y, d = data
        loaders = make_lobo_splits(X, y, d, test_domain=2)
        test_domains = set(loaders["test"].dataset.domains.tolist())
        train_domains = set(loaders["train"].dataset.domains.tolist())
        assert test_domains == {2}
        assert 2 not in train_domains

    def test_splits_cover_all(self, data):
        X, y, d = data
        loaders = make_lobo_splits(X, y, d, test_domain=0)
        n_train = len(loaders["train"].dataset)
        n_val   = len(loaders["val"].dataset)
        n_test  = len(loaders["test"].dataset)
        assert n_train + n_val + n_test == len(X)
