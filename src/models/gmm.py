"""Gaussian Mixture Model for regime detection in cryptocurrency markets.

This module implements a GMM-based clustering approach to detect Risk-On/Risk-Off
regimes without temporal assumptions.
"""

# Enable future annotations for cleaner type hints
from __future__ import annotations

# Dataclass for configuration with auto-generated methods
from dataclasses import dataclass

# Type hints for function signatures
from typing import Iterable, Optional, Tuple

# NumPy for numerical operations
import numpy as np

# Pandas for time-series data with DatetimeIndex
import pandas as pd

# GaussianMixture: probabilistic clustering assuming data comes from mixture of Gaussians
# Unlike HMM, GMM treats each observation independently (no temporal dependencies)
from sklearn.mixture import GaussianMixture

# StandardScaler for feature normalization (zero mean, unit variance)
from sklearn.preprocessing import StandardScaler


@dataclass
class GMMConfig:
    """Configuration for Gaussian Mixture Model regime detection.

    Attributes:
        n_components: Number of mixture components (regimes) to fit.
        covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical').
        n_init: Number of initializations to perform (best is kept).
        max_iter: Maximum number of EM iterations.
        tol: Convergence threshold for log-likelihood.
        random_state: Random seed for reproducibility.
        features: Optional tuple of feature column names to use. If None, uses all features.
    """

    # Number of mixture components (clusters/regimes)
    # Each component represents a distinct market regime
    n_components: int = 2

    # Covariance structure for each component
    # 'full': each component has its own general covariance matrix
    covariance_type: str = "full"

    # Number of random initializations to try
    # GMM is sensitive to initialization; multiple runs help find global optimum
    n_init: int = 10

    # Maximum EM iterations per initialization
    max_iter: int = 500

    # Convergence threshold: stop when log-likelihood change < tol
    tol: float = 1e-3

    # Random seed for reproducible results
    random_state: int = 42

    # Optional feature subset for testing different combinations
    features: Optional[Tuple[str, ...]] = None


def prepare_features(
    df: pd.DataFrame, feature_cols: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, pd.Index, StandardScaler, Tuple[str, ...]]:
    """Prepare and standardize features for GMM clustering.

    Args:
        df: DataFrame containing features with DatetimeIndex.
        feature_cols: Optional iterable of column names to use. If None, uses all columns
            except 'Date'.

    Returns:
        Tuple containing:
            - X: Standardized feature matrix (n_samples, n_features).
            - idx: DatetimeIndex of valid samples after dropping NaN.
            - scaler: Fitted StandardScaler for later transformation.
            - feature_cols: Tuple of feature column names used.
    """
    # Use all columns except 'Date' if no specific features requested
    if feature_cols is None:
        feature_cols = tuple(c for c in df.columns if c not in ("Date",))

    # Select features and remove missing values (GMM cannot handle NaN)
    X_df = df.loc[:, feature_cols].dropna()

    # Preserve date index for mapping clusters back to time periods
    idx = X_df.index

    # Initialize scaler for standardization
    scaler = StandardScaler()

    # Fit scaler and transform: (X - mean) / std
    # Ensures all features have equal influence regardless of scale
    X = scaler.fit_transform(X_df.values)

    return X, idx, scaler, tuple(feature_cols)


def fit_gmm(
    df: pd.DataFrame, cfg: Optional[GMMConfig] = None
) -> Tuple[GaussianMixture, pd.Index, StandardScaler, Tuple[str, ...]]:
    """Fit a Gaussian Mixture Model to cluster market regimes.

    Uses the Expectation-Maximization algorithm to learn mixture components.
    Unlike HMM, GMM treats each observation independently (no temporal structure).

    Args:
        df: DataFrame with features and DatetimeIndex.
        cfg: GMMConfig object with model parameters. If None, uses defaults.

    Returns:
        Tuple containing:
            - model: Fitted GaussianMixture model.
            - idx: DatetimeIndex of training samples.
            - scaler: Fitted StandardScaler used for feature normalization.
            - feats: Tuple of feature names used in training.

    Example:
        >>> config = GMMConfig(n_components=2, covariance_type='full')
        >>> model, idx, scaler, feats = fit_gmm(features, config)
        >>> print(f"BIC: {model.bic(scaler.transform(features[feats].values)):.2f}")
    """
    # Use provided config or create default
    cfg = cfg or GMMConfig()

    # Prepare and standardize features
    X, idx, scaler, feats = prepare_features(df, cfg.features)

    # Initialize Gaussian Mixture Model
    # GMM assumes data is generated from a mixture of K Gaussian distributions
    model = GaussianMixture(
        # Number of mixture components (regimes)
        n_components=cfg.n_components,
        # Covariance structure for each component
        covariance_type=cfg.covariance_type,
        # Number of initializations (best result is kept)
        # Multiple runs help avoid local optima
        n_init=cfg.n_init,
        # Maximum EM iterations per initialization
        max_iter=cfg.max_iter,
        # Convergence threshold for log-likelihood
        tol=cfg.tol,
        # Random seed for reproducibility
        random_state=cfg.random_state,
    )

    # Fit model using EM algorithm
    # E-step: compute posterior probabilities (which component generated each point)
    # M-step: update component parameters (means, covariances, weights)
    model.fit(X)

    return model, idx, scaler, feats


def predict_gmm(
    model: GaussianMixture,
    df: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: Iterable[str],
) -> Tuple[pd.Series, np.ndarray]:
    """Predict regime labels for given features using trained GMM.

    Assigns each observation to the mixture component with highest posterior probability.

    Args:
        model: Trained GaussianMixture model.
        df: DataFrame with features to predict.
        scaler: Fitted StandardScaler from training.
        feature_cols: Feature column names to use (must match training features).

    Returns:
        Tuple containing:
            - regimes: Series with regime labels (0, 1, ..., n_components-1) indexed by date.
            - X: Standardized feature matrix used for prediction.

    Example:
        >>> regimes, X = predict_gmm(model, new_features, scaler, feature_cols)
        >>> print(f"Regime distribution:\n{regimes.value_counts()}")
    """
    # Extract same features used in training
    X_df = df.loc[:, feature_cols].dropna()

    # Preserve date index
    idx = X_df.index

    # Transform using training scaler (not fit_transform!)
    # Must use same normalization as training data
    X = scaler.transform(X_df.values)

    # Predict cluster assignment for each observation
    # GMM assigns each point to component with highest posterior probability
    # Unlike HMM, predictions are independent (no temporal smoothing)
    z = model.predict(X)

    # Return regime labels with date index
    return pd.Series(z, index=idx, name="regime"), X
