from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def _local_csv_path(symbol: str, out_dir: str | Path) -> Path:
    """
    Build the expected local CSV path for a given symbol.

    Handles Yahoo-style tickers like '^GSPC' by mapping to 'GSPC.csv'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = symbol.replace("^", "")  # ^GSPC -> GSPC, ^VIX -> VIX
    return out_dir / f"{fname}.csv"


def _load_local_csv(path: Path) -> pd.DataFrame:
    """Load a local CSV and ensure a datetime index when possible."""
    df = pd.read_csv(path)

    # Try common date column names
    for col in ("Date", "date", "Datetime", "datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break

    return df


def download_and_save(
    symbols: Iterable[str],
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
    out_dir: str | Path = "data/raw",
) -> None:
    """
    Download market data for symbols and save to CSV.

    Offline-first behavior:
    - If a local CSV already exists for a symbol, it is reused and no Yahoo call is made.
    - If it doesn't exist, we try Yahoo. If Yahoo fails/empty -> raise a clear error.
    """
    for sym in symbols:
        csv_path = _local_csv_path(sym, out_dir)

        # ✅ OFFLINE FIRST: reuse local CSV if present
        if csv_path.exists() and csv_path.stat().st_size > 0:
            print(f"✅ Using existing local CSV for {sym}: {csv_path}")
            # Optional: sanity check that file is readable
            _ = _load_local_csv(csv_path)
            continue

        # Otherwise try to download from Yahoo (optional but kept for completeness)
        print(f"📥 Downloading {sym} from Yahoo Finance...")
        df = yf.download(sym, start=start, end=end, interval=interval, progress=False)

        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for {sym} from Yahoo Finance, and no local CSV found at {csv_path}. "
                f"Please place a CSV file at {csv_path} to run offline."
            )

        df.to_csv(csv_path)
        print(f"✅ Saved {sym} to {csv_path}")