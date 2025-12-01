from __future__ import annotations

import argparse
import pathlib as pl
import sys

import pandas as pd

# Ensure the project root is importable so we can access the src package
ROOT = pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_all_regime_stats,
    compute_all_transition_matrices,
)
from src.evaluation.plots import (  # noqa: E402
    plot_conditional_stats,
    plot_regime_timeline,
    plot_transition_matrix,
)
from src.utils.logging_utils import setup_logging  # noqa: E402


def main():
    # CLI entrypoint that loads regimes from all models, computes statistics
    # and transition matrices, and generates comparison plots.
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--hmm", type=str, default="results/hmm_regimes.csv")
    parser.add_argument("--gmm", type=str, default="results/gmm_regimes.csv")
    parser.add_argument("--autoencoder", type=str, default="results/autoencoder_regimes.csv")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    args = parser.parse_args()

    # Set up structured logging (both console and CI-friendly)
    logger = setup_logging(name=__name__)

    # Load engineered features (returns, vol, correlations, etc.)
    features = pd.read_csv(args.features, parse_dates=[0], index_col=0)

    # Load regimes from each model if the corresponding CSV is present.
    regime_dict = {}
    for name, path in [("HMM", args.hmm), ("GMM", args.gmm), ("Autoencoder", args.autoencoder)]:
        if pl.Path(path).exists():
            # Each CSV contains a single integer regime label per date
            regime_dict[name] = pd.read_csv(path, parse_dates=[0], index_col=0).squeeze()
        else:
            logger.warning("%s regimes not found at %s", name, path)

    if not regime_dict:
        # Without any regimes there is nothing to evaluate, so we exit early.
        logger.warning("No regime files found. Exiting.")
        return

    # Compute both per-regime performance statistics and transition matrices.
    stats = compute_all_regime_stats(features, regime_dict)
    trans = compute_all_transition_matrices(regime_dict)

    # Log stats to the console for quick inspection in addition to generating plots.
    logger.info("=== Regime Statistics ===")
    for name, df in stats.items():
        logger.info("\n%s:\n%s", name, df.to_string())

    logger.info("=== Transition Matrices ===")
    for name, df in trans.items():
        logger.info("\n%s:\n%s", name, df.to_string())

    # Create output directory
    out_dir = pl.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot regime timelines overlaid with BTC returns
    returns = features["BTC-USD_ret"] if "BTC-USD_ret" in features.columns else None
    for name, regimes in regime_dict.items():
        plot_regime_timeline(
            regimes,
            returns=returns,
            title=f"{name} Regime Timeline",
            save_path=str(out_dir / f"{name.lower()}_timeline.png"),
        )

    # Plot conditional stats returns by regime and model
    plot_conditional_stats(
        stats,
        metric="BTC-USD_ret_mean",
        title="Mean BTC Returns by Regime",
        save_path=str(out_dir / "conditional_returns.png"),
    )

    # Plot transition matrices for each model
    for name, tm in trans.items():
        plot_transition_matrix(
            tm,
            title=f"{name} Transition Matrix",
            save_path=str(out_dir / f"{name.lower()}_transition.png"),
        )

    logger.info("All plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
