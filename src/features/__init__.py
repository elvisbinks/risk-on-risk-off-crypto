"""Feature engineering utilities.

This subpackage turns raw OHLCV data into model-ready features such as
returns, volatility, volume z-scores and rolling correlations.
"""

from .build_features import build_features

__all__ = ["build_features"]
