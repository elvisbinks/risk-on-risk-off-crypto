"""Performance benchmarking tests for regime detection models.

These tests measure training time, prediction time, and memory usage
to ensure models meet performance requirements.

Lightweight performance checks for the main pipeline components.

The goal is not micro-optimisation but to catch accidental slowdowns in
model training, prediction and feature engineering on modest-sized data.
"""

import time

import numpy as np
import pandas as pd
import pytest

from src.models.autoencoder import AutoencoderConfig, fit_autoencoder, predict_autoencoder
from src.models.gmm import GMMConfig, fit_gmm, predict_gmm
from src.models.hmm import HMMConfig, decode_hmm, fit_hmm


@pytest.fixture
def large_features():
    """Generate large feature dataset for performance testing."""
    np.random.seed(42)
    n_samples = 2000
    n_features = 12
    dates = pd.date_range("2016-01-01", periods=n_samples, freq="D")
    data = np.random.randn(n_samples, n_features)
    columns = [f"feature_{i}" for i in range(n_features)]
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.mark.slow
def test_hmm_training_speed(large_features):
    """Test HMM training completes within acceptable time."""
    config = HMMConfig(n_states=2, n_iter=100)

    start_time = time.time()
    model, idx, scaler, feats = fit_hmm(large_features, config)
    elapsed = time.time() - start_time

    # Should train in less than 5 seconds
    assert elapsed < 5.0, f"HMM training took {elapsed:.2f}s, expected <5s"
    assert model is not None
    print(f"HMM training time: {elapsed:.3f}s")


@pytest.mark.slow
def test_hmm_prediction_speed(large_features):
    """Test HMM prediction speed."""
    config = HMMConfig(n_states=2, n_iter=100)
    model, idx, scaler, feats = fit_hmm(large_features, config)

    start_time = time.time()
    regimes, X = decode_hmm(model, large_features, scaler, feats)
    elapsed = time.time() - start_time

    # Should predict in less than 1 second
    assert elapsed < 1.0, f"HMM prediction took {elapsed:.2f}s, expected <1s"
    assert len(regimes) > 0
    print(f"HMM prediction time: {elapsed:.3f}s")


@pytest.mark.slow
def test_gmm_training_speed(large_features):
    """Test GMM training completes within acceptable time."""
    config = GMMConfig(n_components=2, n_init=5, max_iter=100)

    start_time = time.time()
    model, idx, scaler, feats = fit_gmm(large_features, config)
    elapsed = time.time() - start_time

    # Should train in less than 3 seconds
    assert elapsed < 3.0, f"GMM training took {elapsed:.2f}s, expected <3s"
    assert model is not None
    print(f"GMM training time: {elapsed:.3f}s")


@pytest.mark.slow
def test_gmm_prediction_speed(large_features):
    """Test GMM prediction speed."""
    config = GMMConfig(n_components=2, n_init=5, max_iter=100)
    model, idx, scaler, feats = fit_gmm(large_features, config)

    start_time = time.time()
    regimes, X = predict_gmm(model, large_features, scaler, feats)
    elapsed = time.time() - start_time

    # Should predict in less than 0.5 seconds
    assert elapsed < 0.5, f"GMM prediction took {elapsed:.2f}s, expected <0.5s"
    assert len(regimes) > 0
    print(f"GMM prediction time: {elapsed:.3f}s")


@pytest.mark.slow
def test_autoencoder_training_speed(large_features):
    """Test Autoencoder training completes within acceptable time."""
    config = AutoencoderConfig(
        encoding_dim=4,
        hidden_dims=(8,),
        n_clusters=2,
        epochs=50,
        batch_size=64,
    )

    start_time = time.time()
    ae_model, kmeans, idx, scaler, feats = fit_autoencoder(large_features, config)
    elapsed = time.time() - start_time

    # Should train in less than 15 seconds
    assert elapsed < 15.0, f"Autoencoder training took {elapsed:.2f}s, expected <15s"
    assert ae_model is not None
    assert kmeans is not None
    print(f"Autoencoder training time: {elapsed:.3f}s")


@pytest.mark.slow
def test_autoencoder_prediction_speed(large_features):
    """Test Autoencoder prediction speed."""
    config = AutoencoderConfig(
        encoding_dim=4,
        hidden_dims=(8,),
        n_clusters=2,
        epochs=50,
        batch_size=64,
    )
    ae_model, kmeans, idx, scaler, feats = fit_autoencoder(large_features, config)

    start_time = time.time()
    regimes, embeddings = predict_autoencoder(ae_model, kmeans, large_features, scaler, feats)
    elapsed = time.time() - start_time

    # Should predict in less than 1 second
    assert elapsed < 1.0, f"Autoencoder prediction took {elapsed:.2f}s, expected <1s"
    assert len(regimes) > 0
    print(f"Autoencoder prediction time: {elapsed:.3f}s")


@pytest.mark.slow
def test_feature_engineering_speed():
    """Test feature engineering completes within acceptable time."""
    from src.features.build_features import _annualize, _returns

    np.random.seed(42)
    n_samples = 2000
    dates = pd.date_range("2016-01-01", periods=n_samples, freq="D")
    close = pd.Series(np.random.randn(n_samples).cumsum() + 100, index=dates)

    start_time = time.time()
    returns = _returns(close, kind="log")
    _ = _annualize(returns.rolling(21).std(), trading_days=252)
    elapsed = time.time() - start_time

    # Should compute in less than 0.1 seconds
    assert elapsed < 0.1, f"Feature engineering took {elapsed:.2f}s, expected <0.1s"
    assert len(returns) == len(close)
    print(f"Feature engineering time: {elapsed:.3f}s")


@pytest.mark.slow
def test_model_comparison_speed(large_features):
    """Test that all models can be trained and compared in reasonable time."""
    hmm_config = HMMConfig(n_states=2, n_iter=100)
    gmm_config = GMMConfig(n_components=2, n_init=5, max_iter=100)
    ae_config = AutoencoderConfig(encoding_dim=4, hidden_dims=(8,), n_clusters=2, epochs=50)

    start_time = time.time()

    # Train all models
    hmm_model, _, hmm_scaler, hmm_feats = fit_hmm(large_features, hmm_config)
    gmm_model, _, gmm_scaler, gmm_feats = fit_gmm(large_features, gmm_config)
    ae_model, kmeans, _, ae_scaler, ae_feats = fit_autoencoder(large_features, ae_config)

    # Get predictions
    hmm_regimes, _ = decode_hmm(hmm_model, large_features, hmm_scaler, hmm_feats)
    gmm_regimes, _ = predict_gmm(gmm_model, large_features, gmm_scaler, gmm_feats)
    ae_regimes, _ = predict_autoencoder(ae_model, kmeans, large_features, ae_scaler, ae_feats)

    elapsed = time.time() - start_time

    # Full pipeline should complete in less than 30 seconds
    assert elapsed < 30.0, f"Full pipeline took {elapsed:.2f}s, expected <30s"
    assert len(hmm_regimes) > 0
    assert len(gmm_regimes) > 0
    assert len(ae_regimes) > 0
    print(f"Full pipeline time: {elapsed:.3f}s")
