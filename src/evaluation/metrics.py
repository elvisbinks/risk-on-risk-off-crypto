from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_regime_stats(
    features: pd.DataFrame, regimes: pd.Series, return_col: str = "BTC-USD_ret"
) -> pd.DataFrame:
    """Compute basic return statistics for each regime.

    This function is the main way we evaluate how "good" a regime is: we look at
    the mean and standard deviation of returns when the system is in that regime.

    Args:
        features: DataFrame with features (must include return_col).
        regimes: Series with regime labels (same index as features).
        return_col: Column name for returns used in the evaluation.

    Returns:
        DataFrame with one row per regime and columns for mean, std and count.
    """

    # Align features and regimes on common dates to avoid mismatched indices.
    aligned = features.join(regimes, how="inner")

    # Compute per-regime statistics of the chosen return series.
    stats = aligned.groupby("regime")[return_col].agg(["mean", "std", "count"])

    # Prefix with the return column name so we can safely combine multiple metrics later.
    stats.columns = [f"{return_col}_{c}" for c in stats.columns]
    return stats


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute a transition count matrix from a regime time series.

    Each cell (i, j) stores how many times the system transitioned from
    regime i to regime j in the observed sequence.

    Args:
        regimes: Series with regime labels.

    Returns:
        DataFrame with transition counts (rows = from, columns = to).
    """

    # Sort unique states so the matrix has a stable, readable ordering.
    states = sorted(regimes.unique())
    n = len(states)
    trans = np.zeros((n, n), dtype=int)

    # Iterate over consecutive pairs of regimes to count transitions.
    for i in range(len(regimes) - 1):
        curr = regimes.iloc[i]
        nxt = regimes.iloc[i + 1]
        trans[curr, nxt] += 1

    df = pd.DataFrame(trans, index=states, columns=states)
    df.index.name = "from"
    df.columns.name = "to"
    return df


def compute_all_regime_stats(
    features: pd.DataFrame, regime_dict: Dict[str, pd.Series]
) -> Dict[str, pd.DataFrame]:
    """Compute per-regime statistics for several different models.

    This is a thin convenience wrapper that lets us treat each model's
    regimes in a uniform way when comparing performance.

    Args:
        features: DataFrame with features.
        regime_dict: Dict mapping model name -> regime series.

    Returns:
        Dict mapping model name -> stats DataFrame.
    """

    stats: Dict[str, pd.DataFrame] = {}
    for name, regimes in regime_dict.items():
        stats[name] = compute_regime_stats(features, regimes)
    return stats


def compute_all_transition_matrices(regime_dict: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
    """Compute transition matrices for several different models.

    This mirrors :func:`compute_all_regime_stats` but for transition counts,
    making it easy to compare regime persistence and switching behaviour
    across models.

    Args:
        regime_dict: Dict mapping model name -> regime series.

    Returns:
        Dict mapping model name -> transition matrix DataFrame.
    """

    trans: Dict[str, pd.DataFrame] = {}
    for name, regimes in regime_dict.items():
        trans[name] = compute_transition_matrix(regimes)
    return trans
