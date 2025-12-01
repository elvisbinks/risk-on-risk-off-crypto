from __future__ import annotations

from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_regime_timeline(
    regimes: pd.Series,
    returns: Optional[pd.Series] = None,
    title: str = "Regime Timeline",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
):
    """Plot a time line of regimes, optionally overlaid with returns.

    The coloured background bands show which regime the model thinks the market
    was in on each day. Optionally we overlay a return series to visually
    inspect how regimes line up with price moves.

    Args:
        regimes: Series with regime labels indexed by date.
        returns: Optional series with returns to overlay on a secondary axis.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure (if provided).
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Plot regimes as coloured background bands
    unique_regimes = sorted(regimes.unique())
    colors = sns.color_palette("Set2", len(unique_regimes))

    for i, regime in enumerate(unique_regimes):
        mask = regimes == regime
        dates = regimes.index[mask]
        for date in dates:
            ax.axvspan(
                date,
                date + pd.Timedelta(days=1),
                alpha=0.3,
                color=colors[i],
                # Only label the first span for each regime to avoid legend clutter
                label=f"Regime {regime}" if date == dates[0] else "",
            )

    # Overlay returns if provided
    if returns is not None:
        # Draw returns on a secondary y-axis so scales remain readable
        ax2 = ax.twinx()
        aligned = returns.reindex(regimes.index)
        ax2.plot(
            aligned.index, aligned.values, color="black", alpha=0.6, linewidth=0.8, label="Returns"
        )
        ax2.set_ylabel("Returns", fontsize=10)
        ax2.legend(loc="upper right")

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Regime", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    # Remove duplicate labels so each regime appears only once in the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {save_path}")
    plt.close()


def plot_conditional_stats(
    stats_dict: Dict[str, pd.DataFrame],
    metric: str = "BTC-USD_ret_mean",
    title: str = "Conditional Returns by Regime",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None,
):
    """Compare a chosen metric (e.g. mean return) across models and regimes.

    Each bar group corresponds to a model, and colours correspond to regimes,
    which makes it easy to see which model achieves better separation between
    risk-on and risk-off states.

    Args:
        stats_dict: Dict mapping model name -> stats DataFrame.
        metric: Column name to plot (e.g. "BTC-USD_ret_mean").
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
    """

    fig, ax = plt.subplots(figsize=figsize)

    models = list(stats_dict.keys())
    n_models = len(models)
    x = np.arange(n_models)
    width = 0.35

    # Assume 2 regimes for simplicity; extend if needed.
    regimes = sorted(stats_dict[models[0]].index)
    colors = sns.color_palette("Set2", len(regimes))

    for i, regime in enumerate(regimes):
        # Some models might not have all regimes present; default to zero in that case.
        values = [
            stats_dict[m].loc[regime, metric] if regime in stats_dict[m].index else 0
            for m in models
        ]
        ax.bar(x + i * width, values, width, label=f"Regime {regime}", color=colors[i])

    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(models)
    ax.legend()
    # Horizontal zero-line gives context for positive vs negative mean returns.
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {save_path}")
    plt.close()


def plot_transition_matrix(
    trans_matrix: pd.DataFrame,
    title: str = "Transition Matrix",
    figsize: tuple = (6, 5),
    save_path: Optional[str] = None,
):
    """Visualise a transition matrix as a heatmap.

    Darker cells indicate more frequent transitions between the corresponding
    regimes. Diagonal dominance usually indicates more persistent regimes.

    Args:
        trans_matrix: Transition matrix DataFrame.
        title: Plot title.
        figsize: Figure size.
        save_path: Path to save figure.
    """

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(trans_matrix, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "Count"}, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("To Regime", fontsize=10)
    ax.set_ylabel("From Regime", fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot -> {save_path}")
    plt.close()
