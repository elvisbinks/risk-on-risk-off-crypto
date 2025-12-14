"""Tests for the feature engineering helpers and configuration.

We validate both small numerical helpers (returns, volatility, correlations)
and the higher-level ``build_features`` pipeline on temporary CSV data.
"""

from __future__ import annotations

import pathlib as pl
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    FeatureConfig,
    _annualize,
    _read_symbol_csv,
    _returns,
    _rolling_corr_aligned,
    build_features,
)


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = {
        "Open": np.random.randn(100).cumsum() + 100,
        "High": np.random.randn(100).cumsum() + 105,
        "Low": np.random.randn(100).cumsum() + 95,
        "Close": np.random.randn(100).cumsum() + 100,
        "Adj Close": np.random.randn(100).cumsum() + 100,
        "Volume": np.random.randint(1000000, 10000000, 100),
    }
    df = pd.DataFrame(data, index=dates)
    df.index.name = "Date"
    return df


@pytest.fixture
def temp_data_dir(sample_csv_data):
    """Create temporary directory with sample CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = pl.Path(tmpdir) / "raw"
        raw_dir.mkdir()

        # Save sample data for multiple symbols
        for symbol in ["BTC-USD", "ETH-USD", "GSPC", "VIX"]:
            sample_csv_data.to_csv(raw_dir / f"{symbol}.csv")

        yield raw_dir


def test_returns_log():
    close = pd.Series([100, 105, 103, 108])
    ret = _returns(close, "log")
    assert len(ret) == 4
    assert pd.isna(ret.iloc[0])
    assert ret.iloc[1] == pytest.approx(np.log(105 / 100), rel=1e-6)


def test_returns_pct():
    close = pd.Series([100, 105, 103, 108])
    ret = _returns(close, "pct")
    assert len(ret) == 4
    assert pd.isna(ret.iloc[0])
    assert ret.iloc[1] == pytest.approx(0.05, rel=1e-6)


def test_returns_invalid_kind():
    close = pd.Series([100, 105, 103])
    with pytest.raises(ValueError, match="Unknown ret_kind"):
        _returns(close, "invalid")


def test_annualize():
    std = pd.Series([0.01, 0.02, 0.015])
    ann = _annualize(std, 252)
    expected = std * np.sqrt(252)
    pd.testing.assert_series_equal(ann, expected)


def test_rolling_corr_aligned():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    a = pd.Series(np.random.randn(50), index=dates)
    b = pd.Series(np.random.randn(50), index=dates)

    corr = _rolling_corr_aligned(a, b, window=10, min_frac=0.6)

    assert len(corr) == len(a)
    assert corr.index.equals(a.index)
    # First few values should be NaN due to window
    assert pd.isna(corr.iloc[:5]).all()


def test_read_symbol_csv(temp_data_dir):
    df = _read_symbol_csv(temp_data_dir, "BTC-USD")
    assert isinstance(df, pd.DataFrame)
    assert "Close" in df.columns
    assert "Volume" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Date"


def test_read_symbol_csv_missing_file(temp_data_dir):
    with pytest.raises(FileNotFoundError, match="Missing raw CSV"):
        _read_symbol_csv(temp_data_dir, "NONEXISTENT")


def test_feature_config_defaults():
    cfg = FeatureConfig()
    assert cfg.raw_dir == "data/raw"
    assert cfg.processed_dir == "data/processed"
    assert cfg.ret_kind == "log"
    assert cfg.vol_window == 21
    assert cfg.vol_annualize is True
    assert cfg.vol_trading_days == 252
    assert cfg.vol_z_window == 60
    assert cfg.corr_window == 21


def test_build_features(temp_data_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = pl.Path(tmpdir) / "processed"
        cfg = FeatureConfig(
            raw_dir=str(temp_data_dir),
            processed_dir=str(processed_dir),
            symbols=("BTC-USD", "ETH-USD", "^GSPC", "^VIX"),
            vol_window=10,
            vol_z_window=20,
            corr_window=10,
        )

        features = build_features(cfg)

        assert isinstance(features, pd.DataFrame)
        assert not features.empty
        assert "r_spx" in features.columns
        assert "r_vix" in features.columns
        assert "BTC-USD_ret" in features.columns
        assert "ETH-USD_ret" in features.columns
        assert features.index.name == "Date"

        # Check output file was created
        out_file = processed_dir / "features.csv"
        assert out_file.exists()


def test_build_features_with_custom_ret_kind(temp_data_dir):
    with tempfile.TemporaryDirectory() as tmpdir:
        processed_dir = pl.Path(tmpdir) / "processed"
        cfg = FeatureConfig(
            raw_dir=str(temp_data_dir),
            processed_dir=str(processed_dir),
            symbols=("BTC-USD", "^GSPC"),
            ret_kind="pct",
        )

        features = build_features(cfg)
        assert isinstance(features, pd.DataFrame)
        assert "BTC-USD_ret" in features.columns
