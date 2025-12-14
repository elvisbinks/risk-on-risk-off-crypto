"""Tests for evaluation metrics and plotting helpers.

These tests exercise both numerical outputs (stats, transition matrices)
and the plotting functions to ensure they run without errors and save files.
"""

from __future__ import annotations

import pathlib as pl
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_all_regime_stats,
    compute_all_transition_matrices,
    compute_regime_stats,
    compute_transition_matrix,
)
from src.evaluation.plots import (
    plot_conditional_stats,
    plot_regime_timeline,
    plot_transition_matrix,
)


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    data = {
        "BTC-USD_ret": np.random.randn(100) * 0.02,
        "BTC-USD_vol21": np.abs(np.random.randn(100)) * 0.3,
        "ETH-USD_ret": np.random.randn(100) * 0.025,
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_regimes():
    """Create sample regime series."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    regimes = pd.Series(np.random.choice([0, 1], size=100), index=dates, name="regime")
    return regimes


def test_compute_regime_stats(sample_features, sample_regimes):
    stats = compute_regime_stats(sample_features, sample_regimes, return_col="BTC-USD_ret")

    assert isinstance(stats, pd.DataFrame)
    assert "BTC-USD_ret_mean" in stats.columns
    assert "BTC-USD_ret_std" in stats.columns
    assert "BTC-USD_ret_count" in stats.columns
    assert len(stats) == 2  # Two regimes


def test_compute_regime_stats_custom_column(sample_features, sample_regimes):
    stats = compute_regime_stats(sample_features, sample_regimes, return_col="ETH-USD_ret")

    assert "ETH-USD_ret_mean" in stats.columns
    assert "ETH-USD_ret_std" in stats.columns


def test_compute_transition_matrix(sample_regimes):
    trans = compute_transition_matrix(sample_regimes)

    assert isinstance(trans, pd.DataFrame)
    assert trans.shape == (2, 2)
    assert trans.index.name == "from"
    assert trans.columns.name == "to"
    assert (trans >= 0).all().all()
    # Sum of all transitions should be len(regimes) - 1
    assert trans.sum().sum() == len(sample_regimes) - 1


def test_transition_matrix_values():
    # Create deterministic regime series
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    regimes = pd.Series([0, 0, 1, 1, 0, 0, 1, 1, 1, 0], index=dates, name="regime")

    trans = compute_transition_matrix(regimes)

    # 0->0: positions 0->1, 4->5
    assert trans.loc[0, 0] == 2
    # 0->1: positions 1->2, 5->6
    assert trans.loc[0, 1] == 2
    # 1->0: positions 3->4, 8->9
    assert trans.loc[1, 0] == 2
    # 1->1: positions 2->3, 6->7, 7->8
    assert trans.loc[1, 1] == 3


def test_compute_all_regime_stats(sample_features):
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    regime_dict = {
        "Model1": pd.Series(np.random.choice([0, 1], size=100), index=dates, name="regime"),
        "Model2": pd.Series(np.random.choice([0, 1], size=100), index=dates, name="regime"),
    }

    stats = compute_all_regime_stats(sample_features, regime_dict)

    assert isinstance(stats, dict)
    assert "Model1" in stats
    assert "Model2" in stats
    assert isinstance(stats["Model1"], pd.DataFrame)


def test_compute_all_transition_matrices():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    regime_dict = {
        "Model1": pd.Series(np.random.choice([0, 1], size=100), index=dates, name="regime"),
        "Model2": pd.Series(np.random.choice([0, 1], size=100), index=dates, name="regime"),
    }

    trans = compute_all_transition_matrices(regime_dict)

    assert isinstance(trans, dict)
    assert "Model1" in trans
    assert "Model2" in trans
    assert isinstance(trans["Model1"], pd.DataFrame)


def test_plot_regime_timeline(sample_regimes, sample_features):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = pl.Path(tmpdir) / "timeline.png"

        plot_regime_timeline(
            sample_regimes,
            returns=sample_features["BTC-USD_ret"],
            title="Test Timeline",
            save_path=str(save_path),
        )

        assert save_path.exists()


def test_plot_regime_timeline_no_returns(sample_regimes):
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = pl.Path(tmpdir) / "timeline_no_ret.png"

        plot_regime_timeline(
            sample_regimes,
            returns=None,
            title="Test Timeline No Returns",
            save_path=str(save_path),
        )

        assert save_path.exists()


def test_plot_conditional_stats(sample_features, sample_regimes):
    stats_dict = {
        "Model1": compute_regime_stats(sample_features, sample_regimes),
        "Model2": compute_regime_stats(sample_features, sample_regimes),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = pl.Path(tmpdir) / "conditional.png"

        plot_conditional_stats(
            stats_dict,
            metric="BTC-USD_ret_mean",
            title="Test Conditional Stats",
            save_path=str(save_path),
        )

        assert save_path.exists()


def test_plot_transition_matrix(sample_regimes):
    trans = compute_transition_matrix(sample_regimes)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = pl.Path(tmpdir) / "transition.png"

        plot_transition_matrix(
            trans,
            title="Test Transition Matrix",
            save_path=str(save_path),
        )

        assert save_path.exists()


def test_regime_stats_with_missing_data():
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    features = pd.DataFrame({"BTC-USD_ret": np.random.randn(50) * 0.02}, index=dates)

    # Regimes with different index (some overlap)
    regime_dates = pd.date_range("2020-01-10", periods=30, freq="D")
    regimes = pd.Series(np.random.choice([0, 1], size=30), index=regime_dates, name="regime")

    stats = compute_regime_stats(features, regimes)

    # Should only compute stats for overlapping dates
    assert stats["BTC-USD_ret_count"].sum() == 30


def test_transition_matrix_single_regime():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    regimes = pd.Series([0] * 10, index=dates, name="regime")

    trans = compute_transition_matrix(regimes)

    assert trans.shape == (1, 1)
    assert trans.loc[0, 0] == 9
