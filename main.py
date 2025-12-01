"""
Main entry point for Risk-On/Risk-Off regime detection in cryptocurrency markets.

This script runs the complete pipeline:
1. Fetch historical market data
2. Build features (returns, volatility, correlations)
3. Train three models (HMM, GMM, Autoencoder+KMeans)
4. Evaluate and visualize results

Usage:
    python main.py
"""



from __future__ import annotations

import pathlib as pl
import sys

# Ensure project root is on sys.path
ROOT = pl.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_features import main as build_features_main
from scripts.evaluate_models import main as evaluate_models_main
from scripts.fetch_data import main as fetch_data_main
from scripts.run_autoencoder import main as run_autoencoder_main
from scripts.run_gmm import main as run_gmm_main
from scripts.run_hmm import main as run_hmm_main
from src.utils.logging_utils import setup_logging


def main():
    """Run the complete regime detection pipeline."""

    logger = setup_logging(name=__name__)

    print("=" * 60)
    print("Risk-On / Risk-Off Regime Detection")
    print("=" * 60)
    print()

    # Step 1: Fetch data
    logger.info("Step 1/5: Fetching historical market data...")
    print("📊 Fetching historical market data from Yahoo Finance...")
    try:
        # Override sys.argv to pass default config to fetch_data
        original_argv = sys.argv.copy()
        sys.argv = ["fetch_data", "--config", "configs/default.yaml"]
        fetch_data_main()
        sys.argv = original_argv
        print("✅ Data fetched successfully\n")
    except Exception as e:
        logger.error("Failed to fetch data: %s", e)
        print(f"❌ Error fetching data: {e}\n")
        return

    # Step 2: Build features
    logger.info("Step 2/5: Engineering features...")
    print("🔧 Building features (returns, volatility, correlations)...")
    try:
        original_argv = sys.argv.copy()
        sys.argv = ["build_features", "--config", "configs/default.yaml"]
        build_features_main()
        sys.argv = original_argv
        print("✅ Features built successfully\n")
    except Exception as e:
        logger.error("Failed to build features: %s", e)
        print(f"❌ Error building features: {e}\n")
        return

    # Step 3: Train models
    logger.info("Step 3/5: Training models...")
    print("🤖 Training models...")

    # Train HMM
    print("  → Training Hidden Markov Model (HMM)...")
    try:
        original_argv = sys.argv.copy()
        sys.argv = ["run_hmm", "--config", "configs/hmm.yaml"]
        run_hmm_main()
        sys.argv = original_argv
        print("    ✅ HMM trained")
    except Exception as e:
        logger.error("Failed to train HMM: %s", e)
        print(f"    ❌ HMM failed: {e}")

    # Train GMM
    print("  → Training Gaussian Mixture Model (GMM)...")
    try:
        original_argv = sys.argv.copy()
        sys.argv = ["run_gmm", "--config", "configs/gmm.yaml"]
        run_gmm_main()
        sys.argv = original_argv
        print("    ✅ GMM trained")
    except Exception as e:
        logger.error("Failed to train GMM: %s", e)
        print(f"    ❌ GMM failed: {e}")

    # Train Autoencoder
    print("  → Training Autoencoder + K-Means...")
    try:
        original_argv = sys.argv.copy()
        sys.argv = ["run_autoencoder", "--config", "configs/autoencoder.yaml"]
        run_autoencoder_main()
        sys.argv = original_argv
        print("    ✅ Autoencoder trained")
    except Exception as e:
        logger.error("Failed to train Autoencoder: %s", e)
        print(f"    ❌ Autoencoder failed: {e}")

    print()

    # Step 4: Evaluate and visualize
    logger.info("Step 4/5: Evaluating models and generating visualizations...")
    print("📈 Evaluating models and generating visualizations...")
    try:
        original_argv = sys.argv.copy()
        sys.argv = ["evaluate_models"]
        evaluate_models_main()
        sys.argv = original_argv
        print("✅ Evaluation complete\n")
    except Exception as e:
        logger.error("Failed to evaluate models: %s", e)
        print(f"❌ Error during evaluation: {e}\n")
        return

    # Step 5: Summary
    print("=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print()
    print("📁 Results saved to:")
    print("   - results/hmm_regimes.csv")
    print("   - results/gmm_regimes.csv")
    print("   - results/autoencoder_regimes.csv")
    print("   - results/figures/ (timeline plots, transition matrices)")
    print()
    print("📊 Check the results/ directory for detailed outputs.")
    print()
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
