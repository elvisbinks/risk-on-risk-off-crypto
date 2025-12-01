from __future__ import annotations

import pathlib as pl
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Configuration for the feature engineering pipeline.

    These parameters control how raw OHLCV data is transformed into the
    model-ready feature matrix used by the regime detection models.
    """

    # Directory where raw CSV files from Yahoo Finance are stored
    raw_dir: str = "data/raw"
    # Directory where the engineered feature matrix will be written
    processed_dir: str = "data/processed"
    # Symbols to process: crypto assets plus benchmarks like S&P 500 and VIX
    symbols: Tuple[str, ...] = ("BTC-USD", "ETH-USD", "^GSPC", "^VIX")
    # Choice of return definition. Log returns are better for modeling, pct for intuition.
    ret_kind: str = "log"  # "log" or "pct"
    # Rolling window (in trading days) used to estimate volatility
    vol_window: int = 21
    # Whether to scale volatility to an annualized number (vs daily)
    vol_annualize: bool = True
    # Number of trading days per year used for annualization (252 is common for equities)
    vol_trading_days: int = 252
    # Window length for computing a z-score of trading volume
    vol_z_window: int = 60  # window for volume z-score
    # Window length for rolling correlation with SPX / VIX
    corr_window: int = 21


def _read_symbol_csv(raw_dir: pl.Path, symbol: str) -> pd.DataFrame:
    """Load a single symbol's CSV and normalise it into a clean price series.

    The raw Yahoo Finance files can have slightly different formats depending on
    the asset (especially indices). This helper hides that complexity and always
    returns a DataFrame indexed by Date with numeric OHLCV columns.
    """

    # Yahoo Finance removes carets (^) from filenames, so normalise symbol first
    fname = f"{symbol.replace('^','')}.csv"
    path = raw_dir / fname
    if not path.exists():
        # Fail fast so the pipeline clearly reports which input is missing
        raise FileNotFoundError(f"Missing raw CSV for {symbol}: {path}")

    # Read raw CSV first without parsing dates so we can detect and fix odd headers.
    # Some index data (e.g. ^GSPC) comes with two meta-information rows at the top.
    raw = pd.read_csv(path)
    # Standardise column names to avoid subtle bugs (e.g. ' adj close ' vs 'Adj Close').
    raw.columns = [str(c).strip().title() for c in raw.columns]

    # Special handling for Yahoo-style files where the first column is called 'Price'
    # and the first two rows encode the ticker and the word 'Date'. This occurs for
    # some index downloads and breaks direct date parsing.
    if "Price" in raw.columns and len(raw) >= 2:
        first_vals = raw["Price"].astype(str).str.lower().tolist()[:2]
        if any(v.startswith("ticker") for v in first_vals) and any(
            v.startswith("date") for v in first_vals
        ):
            # Drop the two meta rows and treat the 'Price' column as the date field
            raw = raw.iloc[2:].copy()
            # Coerce potential date strings into a proper datetime index
            raw["Date"] = pd.to_datetime(raw["Price"], errors="coerce")
            raw = raw.drop(columns=["Price"])
            raw = raw.set_index("Date").sort_index()
        else:
            # Generic fallback: assume first column contains dates and move it to index
            try:
                raw.index = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
                raw = raw.drop(columns=[raw.columns[0]])
                raw = raw.sort_index()
            except Exception:
                # If this heuristic fails we keep the original frame and try again below
                pass

    # If we still do not have a DatetimeIndex, try a more direct parse using pandas
    if not isinstance(raw.index, pd.DatetimeIndex):
        # This second attempt mirrors the usual "index_col=0, parse_dates=True" pattern
        try:
            df2 = pd.read_csv(path, index_col=0, parse_dates=True)
            df2.columns = [str(c).strip().title() for c in df2.columns]
            raw = df2
        except Exception:
            # In the worst case we keep whatever index we have and rely on downstream checks
            pass

    # Ensure all numeric price/volume columns are converted to floats; this guards
    # against commas or stray characters that pandas may import as objects.
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    # Work with a clean, chronological time series and drop rows with missing closes,
    # since most features rely on a valid closing price.
    raw = raw.sort_index()
    if "Close" in raw.columns:
        raw = raw[raw["Close"].notna()]

    # Name the index consistently so downstream joins can rely on 'Date'.
    raw.index.name = "Date"
    return raw


def _returns(close: pd.Series, kind: str) -> pd.Series:
    """Compute returns from a price series using the chosen convention.

    Log returns are additive over time and often preferred in statistical models,
    while percentage returns are more intuitive but slightly less convenient.
    """

    if kind == "log":
        # Use log(P_t) - log(P_{t-1}) rather than log(P_t / P_{t-1}) to avoid division
        # by zero and to exploit the log-difference identity.
        return np.log(close).diff()
    elif kind == "pct":
        # Classic percentage change (P_t - P_{t-1}) / P_{t-1}
        return close.pct_change()
    else:
        # Explicitly fail on unknown modes so misconfigurations surface early.
        raise ValueError(f"Unknown ret_kind: {kind}")


def _annualize(std: pd.Series, trading_days: int) -> pd.Series:
    """Convert daily (or period) standard deviation into annualised volatility.

    We assume that returns are independent and identically distributed, so the
    standard deviation over N periods scales with sqrt(N).
    """

    return std * np.sqrt(trading_days)


def _rolling_corr_aligned(
    a: pd.Series, b: pd.Series, window: int, min_frac: float = 0.6
) -> pd.Series:
    """Rolling correlation of a with b using the intersection of dates.
    Requires at least ceil(min_frac * window) overlapping points in the window.
    Returns a Series aligned to a's index.
    """
    # Require a minimum number of overlapping observations; this avoids unstable
    # correlations computed on very few points.
    min_periods = max(2, int(np.ceil(min_frac * window)))

    # Align series on the intersection of dates so we only correlate where both
    # assets have data (important for assets with different trading calendars).
    ab = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()

    # Compute rolling correlation on the aligned window.
    corr_on_common = ab["a"].rolling(window, min_periods=min_periods).corr(ab["b"])

    # Reindex back to the original index of 'a'. Dates with no overlapping data
    # will naturally appear as NaN.
    return corr_on_common.reindex(a.index)


def build_features(cfg: Optional[FeatureConfig] = None) -> pd.DataFrame:
    """Build the main feature matrix used by all regime detection models.

    The resulting DataFrame combines benchmark returns (SPX, VIX) with
    crypto-specific features such as returns, volatility, volume z-scores
    and rolling correlations with benchmarks.
    """

    cfg = cfg or FeatureConfig()
    raw_dir = pl.Path(cfg.raw_dir)
    processed_dir = pl.Path(cfg.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load market benchmarks (S&P 500 and VIX). These act as macro proxies
    # for risk-on / risk-off conditions that we correlate crypto with.
    spx = _read_symbol_csv(raw_dir, "^GSPC")
    vix = _read_symbol_csv(raw_dir, "^VIX")
    r_spx = _returns(spx["Close"], cfg.ret_kind).rename("r_spx")
    r_vix = _returns(vix["Close"], cfg.ret_kind).rename("r_vix")

    # Start with benchmark returns; crypto features will be added to this dict.
    out_cols: Dict[str, pd.Series] = {
        "r_spx": r_spx,
        "r_vix": r_vix,
    }

    # Determine crypto symbols (exclude benchmarks beginning with '^').
    # This keeps configuration simple: one list covers both benchmarks and assets.
    crypto_syms: Iterable[str] = [s for s in cfg.symbols if not s.startswith("^")]

    for sym in crypto_syms:
        # Read raw OHLCV for each crypto asset
        df = _read_symbol_csv(raw_dir, sym)
        close = df["Close"]
        vol = df["Volume"]

        # Core feature: asset returns
        r = _returns(close, cfg.ret_kind).rename(f"{sym}_ret")

        # Realised volatility from returns, possibly annualised
        vol_std = r.rolling(cfg.vol_window).std()
        if cfg.vol_annualize:
            vol_std = _annualize(vol_std, cfg.vol_trading_days)
        vol_std = vol_std.rename(f"{sym}_vol{cfg.vol_window}")

        # Volume z-score on log1p(volume) to stabilise scale and highlight extremes
        vol_log = np.log1p(vol)
        vol_mu = vol_log.rolling(cfg.vol_z_window).mean()
        vol_sd = vol_log.rolling(cfg.vol_z_window).std()
        vol_z = ((vol_log - vol_mu) / vol_sd).rename(f"{sym}_vol_z{cfg.vol_z_window}")

        # Rolling correlations with SPX and VIX returns capture how tightly
        # the crypto asset moves with equity and volatility markets.
        corr_spx = _rolling_corr_aligned(r, r_spx, cfg.corr_window).rename(
            f"{sym}_corr_spx{cfg.corr_window}"
        )
        corr_vix = _rolling_corr_aligned(r, r_vix, cfg.corr_window).rename(
            f"{sym}_corr_vix{cfg.corr_window}"
        )

        # Collect all newly created features into the output column dictionary.
        out_cols[r.name] = r
        out_cols[vol_std.name] = vol_std
        out_cols[vol_z.name] = vol_z
        out_cols[corr_spx.name] = corr_spx
        out_cols[corr_vix.name] = corr_vix

    # Align columns; avoid dropping rows due to correlation NaNs. Instead we only
    # require that the core return series are present. Correlations may appear
    # slightly later in the sample and will be NaN at the very beginning.
    features = pd.concat(out_cols.values(), axis=1)
    core = ["r_spx", "r_vix"] + [f"{sym}_ret" for sym in crypto_syms]
    to_drop = [c for c in core if c in features.columns]
    features = features.dropna(subset=to_drop)

    # Ensure index is named consistently so that later joins and plots use 'Date'.
    features.index.name = "Date"

    # Persist the engineered feature set; this is the main input to modelling.
    out_path = processed_dir / "features.csv"
    features.to_csv(out_path, index=True)
    print(f"Saved features -> {out_path}")

    return features
