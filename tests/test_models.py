"""Unit tests for all three regime detection models (HMM, GMM, Autoencoder).

The goal here is to ensure configuration defaults, training routines and
prediction interfaces behave as expected on small synthetic datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.autoencoder import (
    AutoencoderConfig,
    SimpleAutoencoder,
    fit_autoencoder,
    predict_autoencoder,
)
from src.models.autoencoder import prepare_features as ae_prepare
from src.models.autoencoder import (
    train_autoencoder,
)
from src.models.gmm import GMMConfig, fit_gmm, predict_gmm
from src.models.gmm import prepare_features as gmm_prepare
from src.models.hmm import HMMConfig, decode_hmm, fit_hmm
from src.models.hmm import prepare_features as hmm_prepare


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    data = {
        "ret": np.random.randn(200) * 0.02,
        "vol": np.abs(np.random.randn(200)) * 0.3,
        "corr": np.random.randn(200) * 0.5,
    }
    df = pd.DataFrame(data, index=dates)
    return df


# HMM Tests
def test_hmm_config_defaults():
    cfg = HMMConfig()
    assert cfg.n_states == 2
    assert cfg.covariance_type == "full"
    assert cfg.n_iter == 500
    assert cfg.tol == 1e-3
    assert cfg.random_state == 42


def test_hmm_prepare_features(sample_features):
    X, idx, scaler, feats = hmm_prepare(sample_features)
    assert X.shape[0] == len(sample_features)
    assert X.shape[1] == 3
    assert len(idx) == len(sample_features)
    assert len(feats) == 3
    # Check standardization
    assert np.abs(X.mean(axis=0)).max() < 0.1
    assert np.abs(X.std(axis=0) - 1.0).max() < 0.1


def test_fit_hmm(sample_features):
    cfg = HMMConfig(n_states=2, n_iter=50)
    model, idx, scaler, feats = fit_hmm(sample_features, cfg)

    assert model.n_components == 2
    assert len(idx) == len(sample_features)
    assert len(feats) == 3


def test_decode_hmm(sample_features):
    cfg = HMMConfig(n_states=2, n_iter=50)
    model, idx, scaler, feats = fit_hmm(sample_features, cfg)
    regimes, X = decode_hmm(model, sample_features, scaler, feats)

    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(sample_features)
    assert regimes.name == "regime"
    assert set(regimes.unique()).issubset({0, 1})


def test_hmm_with_custom_features(sample_features):
    cfg = HMMConfig(features=("ret", "vol"))
    model, idx, scaler, feats = fit_hmm(sample_features, cfg)
    assert len(feats) == 2
    assert "ret" in feats
    assert "vol" in feats


# GMM Tests
def test_gmm_config_defaults():
    cfg = GMMConfig()
    assert cfg.n_components == 2
    assert cfg.covariance_type == "full"
    assert cfg.n_init == 10
    assert cfg.max_iter == 500
    assert cfg.tol == 1e-3
    assert cfg.random_state == 42


def test_gmm_prepare_features(sample_features):
    X, idx, scaler, feats = gmm_prepare(sample_features)
    assert X.shape[0] == len(sample_features)
    assert X.shape[1] == 3
    assert len(idx) == len(sample_features)


def test_fit_gmm(sample_features):
    cfg = GMMConfig(n_components=2, max_iter=50)
    model, idx, scaler, feats = fit_gmm(sample_features, cfg)

    assert model.n_components == 2
    assert len(idx) == len(sample_features)
    assert hasattr(model, "means_")
    assert hasattr(model, "covariances_")


def test_predict_gmm(sample_features):
    cfg = GMMConfig(n_components=2, max_iter=50)
    model, idx, scaler, feats = fit_gmm(sample_features, cfg)
    regimes, X = predict_gmm(model, sample_features, scaler, feats)

    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(sample_features)
    assert regimes.name == "regime"
    assert set(regimes.unique()).issubset({0, 1})


def test_gmm_bic_aic(sample_features):
    cfg = GMMConfig(n_components=2, max_iter=50)
    model, idx, scaler, feats = fit_gmm(sample_features, cfg)
    X, _, _, _ = gmm_prepare(sample_features, feats)

    bic = model.bic(X)
    aic = model.aic(X)

    assert isinstance(bic, float)
    assert isinstance(aic, float)
    assert bic > 0
    assert aic > 0


# Autoencoder Tests
def test_autoencoder_config_defaults():
    cfg = AutoencoderConfig()
    assert cfg.encoding_dim == 4
    assert cfg.hidden_dims == (8,)
    assert cfg.n_clusters == 2
    assert cfg.epochs == 100
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 1e-3
    assert cfg.random_state == 42


def test_simple_autoencoder_forward():
    model = SimpleAutoencoder(input_dim=5, encoding_dim=2, hidden_dims=(8,))
    import torch

    x = torch.randn(10, 5)
    x_recon, z = model(x)

    assert x_recon.shape == (10, 5)
    assert z.shape == (10, 2)


def test_simple_autoencoder_encode():
    model = SimpleAutoencoder(input_dim=5, encoding_dim=2, hidden_dims=(8,))
    import torch

    x = torch.randn(10, 5)
    z = model.encode(x)

    assert z.shape == (10, 2)


def test_ae_prepare_features(sample_features):
    X, idx, scaler, feats = ae_prepare(sample_features)
    assert X.shape[0] == len(sample_features)
    assert X.shape[1] == 3
    assert len(idx) == len(sample_features)


def test_train_autoencoder(sample_features):
    X, _, _, _ = ae_prepare(sample_features)
    cfg = AutoencoderConfig(encoding_dim=2, hidden_dims=(4,), epochs=10, batch_size=32)

    model, embeddings = train_autoencoder(X, cfg)

    assert isinstance(model, SimpleAutoencoder)
    assert embeddings.shape == (len(X), 2)


def test_fit_autoencoder(sample_features):
    cfg = AutoencoderConfig(encoding_dim=2, hidden_dims=(4,), epochs=10, n_clusters=2)
    ae_model, kmeans, idx, scaler, feats = fit_autoencoder(sample_features, cfg)

    assert isinstance(ae_model, SimpleAutoencoder)
    assert kmeans.n_clusters == 2
    assert len(idx) == len(sample_features)


def test_predict_autoencoder(sample_features):
    cfg = AutoencoderConfig(encoding_dim=2, hidden_dims=(4,), epochs=10, n_clusters=2)
    ae_model, kmeans, idx, scaler, feats = fit_autoencoder(sample_features, cfg)
    regimes, embeddings = predict_autoencoder(ae_model, kmeans, sample_features, scaler, feats)

    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(sample_features)
    assert regimes.name == "regime"
    assert set(regimes.unique()).issubset({0, 1})
    assert embeddings.shape == (len(sample_features), 2)


def test_autoencoder_with_custom_features(sample_features):
    cfg = AutoencoderConfig(features=("ret", "vol"), epochs=5)
    ae_model, kmeans, idx, scaler, feats = fit_autoencoder(sample_features, cfg)
    assert len(feats) == 2
    assert "ret" in feats
    assert "vol" in feats
