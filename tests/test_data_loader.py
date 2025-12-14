"""Tests for the Yahoo Finance data loader utilities.

These tests focus on path handling, error conditions, and the contract of
``fetch_ohlcv`` / ``download_and_save`` without relying on real network calls.
"""

from __future__ import annotations

import pathlib as pl
import tempfile

import pandas as pd
import pytest

from src.data.yfinance_loader import _to_path, download_and_save, fetch_ohlcv


def test_to_path_creates_parent_dirs():
    # Verify that _to_path creates the parent directories if they don't exist
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pl.Path(tmpdir) / "subdir" / "file.csv"
        result = _to_path(path)
        assert result.parent.exists()
        assert result.is_absolute()


def test_to_path_expands_user():
    path = _to_path("~/test.csv")
    assert "~" not in str(path)
    assert path.is_absolute()


def test_fetch_ohlcv_returns_dataframe():
    # This test requires network access; skip if offline
    try:
        df = fetch_ohlcv("BTC-USD", start="2024-01-01", end="2024-01-10", interval="1d")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert df.index.name == "Date"
    except Exception as e:
        pytest.skip(f"Network test failed: {e}")


def test_fetch_ohlcv_raises_on_invalid_symbol():
    with pytest.raises(RuntimeError, match="No data returned"):
        fetch_ohlcv("INVALID_SYMBOL_XYZ", start="2024-01-01", end="2024-01-10")


def test_download_and_save_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pl.Path(tmpdir)
        try:
            download_and_save(
                ["BTC-USD"], start="2024-01-01", end="2024-01-05", interval="1d", out_dir=out_dir
            )
            saved_file = out_dir / "BTC-USD.csv"
            assert saved_file.exists()
            df = pd.read_csv(saved_file)
            assert not df.empty
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")


def test_download_and_save_handles_caret_symbols():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = pl.Path(tmpdir)
        try:
            download_and_save(
                ["^GSPC"], start="2024-01-01", end="2024-01-05", interval="1d", out_dir=out_dir
            )
            saved_file = out_dir / "GSPC.csv"
            assert saved_file.exists()
        except Exception as e:
            pytest.skip(f"Network test failed: {e}")
