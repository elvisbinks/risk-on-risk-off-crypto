"""Yahoo Finance data loader for cryptocurrency and traditional market data.

This module provides small helper functions to download price time series
from Yahoo Finance and persist them as CSV files. The CSVs are then used
by the feature engineering pipeline.
"""

from __future__ import annotations

# pathlib gives a convenient, cross-platform way to work with filesystem paths
import pathlib as _pl
from typing import Iterable, Optional

# pandas for tabular time-series data handling
import pandas as pd

# yfinance is a thin wrapper around Yahoo Finance HTTP APIs
import yfinance as yf


def _to_path(pathlike: str | _pl.Path) -> _pl.Path:
    """Convert path-like object to resolved Path and ensure parent directory exists.

    This utility keeps path handling consistent across the project and makes sure
    parent directories exist before we try to write any CSV files.

    Args:
        pathlike: String or Path object representing a file path.

    Returns:
        Resolved absolute Path with parent directory created.
    """

    # Expand user (~), resolve relative segments, and create parent directories
    p = _pl.Path(pathlike).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def fetch_ohlcv(symbol: str, start: str, end: Optional[str], interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

    Args:
        symbol: Ticker symbol (e.g., 'BTC-USD', '^GSPC', '^VIX').
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format. If None, fetches until today.
        interval: Data frequency ('1d' for daily, '1h' for hourly, etc.).

    Returns:
        DataFrame with OHLCV data indexed by Date.

    Raises:
        RuntimeError: If no data is returned for the symbol.

    Example:
        >>> df = fetch_ohlcv('BTC-USD', start='2020-01-01', end='2020-12-31')
        >>> print(df.head())
    """
    # Use yfinance to download OHLCV data for the given symbol and date range.
    # auto_adjust=False keeps original prices, which is important when comparing
    # across assets and with other data sources.
    df = yf.download(
        symbol, start=start, end=end, interval=interval, auto_adjust=False, progress=False
    )
    # Guard against cases where Yahoo returns no data (bad symbol, no history, etc.)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    # Normalise column names so downstream code can rely on canonical labels
    # such as 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    df = df.rename(columns=str.title)

    # Set a consistent index name so joins on 'Date' are easy later
    df.index.name = "Date"
    return df


def download_and_save(
    symbols: Iterable[str],
    start: str,
    end: Optional[str],
    interval: str,
    out_dir: str | _pl.Path,
) -> None:
    """Download OHLCV data for multiple symbols and save to CSV files.

    Args:
        symbols: Iterable of ticker symbols to download.
        start: Start date in 'YYYY-MM-DD' format.
        end: End date in 'YYYY-MM-DD' format. If None, fetches until today.
        interval: Data frequency ('1d' for daily, '1h' for hourly, etc.).
        out_dir: Output directory path where CSV files will be saved.

    Note:
        Symbols starting with '^' (e.g., '^GSPC') will have the caret removed
        in the filename (e.g., 'GSPC.csv').

    Example:
        >>> symbols = ['BTC-USD', 'ETH-USD', '^GSPC', '^VIX']
        >>> download_and_save(symbols, '2020-01-01', None, '1d', 'data/raw')
    """
    # Ensure output directory exists before writing any CSV files
    out = _pl.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        # Download data for each symbol in turn. If a symbol fails, fetch_ohlcv
        # will raise, which is usually preferable to silently skipping it.
        df = fetch_ohlcv(sym, start=start, end=end, interval=interval)

        # Yahoo Finance uses the caret (^) to mark indices, but filesystem
        # paths typically do not, so we drop it from the filename.
        out_path = out / f"{sym.replace('^','')}.csv"

        # Persist the clean OHLCV data for later use by feature engineering.
        df.to_csv(out_path)
        print(f"Saved {sym} -> {out_path}")
