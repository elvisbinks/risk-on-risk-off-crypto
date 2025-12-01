from __future__ import annotations

import argparse
import pathlib as pl
import sys

import pandas as pd
import yaml

# Ensure the project root is importable so we can access the src package
ROOT = pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.hmm import HMMConfig, decode_hmm, fit_hmm  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def main():
    # CLI entrypoint for training an HMM on the engineered features and
    # exporting the inferred regime sequence to a CSV file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--config", type=str, default="configs/hmm.yaml")
    args = parser.parse_args()

    # Load model hyperparameters from the YAML configuration file.
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    logger = setup_logging(name=__name__)

    # Translate the raw YAML dictionary into a typed HMMConfig instance.
    hmm_cfg = HMMConfig(
        n_states=int(cfg_dict.get("n_states", 2)),
        covariance_type=str(cfg_dict.get("covariance_type", "full")),
        n_iter=int(cfg_dict.get("n_iter", 500)),
        tol=float(cfg_dict.get("tol", 1e-3)),
        random_state=int(cfg_dict.get("random_state", 42)),
        features=tuple(cfg_dict.get("features")) if cfg_dict.get("features") else None,
    )

    # Load the feature matrix that will be used for training and decoding.
    df = pd.read_csv(args.features, parse_dates=[0], index_col=0)

    # Fit the HMM and decode the most likely regime sequence over the sample.
    model, idx, scaler, feats = fit_hmm(df, hmm_cfg)
    regimes, _ = decode_hmm(model, df, scaler, feats)

    # Persist regime labels so that other scripts (evaluation, plotting) can use them.
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "hmm_regimes.csv"
    regimes.to_csv(out_csv, header=True)
    logger.info("Saved HMM regimes -> %s", out_csv)


if __name__ == "__main__":
    main()
