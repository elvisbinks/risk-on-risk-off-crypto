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

from src.models.gmm import GMMConfig, fit_gmm, predict_gmm  # noqa: E402
from src.utils.logging_utils import setup_logging  # noqa: E402


def main():
    # CLI entrypoint for fitting a Gaussian Mixture Model (GMM) to the
    # feature matrix and exporting cluster assignments as regimes.
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--config", type=str, default="configs/gmm.yaml")
    args = parser.parse_args()

    # Load GMM hyperparameters from YAML and construct the config object.
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    logger = setup_logging(name=__name__)

    gmm_cfg = GMMConfig(
        n_components=int(cfg_dict.get("n_components", 2)),
        covariance_type=str(cfg_dict.get("covariance_type", "full")),
        n_init=int(cfg_dict.get("n_init", 10)),
        max_iter=int(cfg_dict.get("max_iter", 500)),
        tol=float(cfg_dict.get("tol", 1e-3)),
        random_state=int(cfg_dict.get("random_state", 42)),
        features=tuple(cfg_dict.get("features")) if cfg_dict.get("features") else None,
    )

    # Load the feature matrix and fit the GMM on the selected features.
    df = pd.read_csv(args.features, parse_dates=[0], index_col=0)

    model, idx, scaler, feats = fit_gmm(df, gmm_cfg)
    regimes, _ = predict_gmm(model, df, scaler, feats)

    # Save regime labels and report information criteria for quick model comparison.
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "gmm_regimes.csv"
    regimes.to_csv(out_csv, header=True)
    logger.info("Saved GMM regimes -> %s", out_csv)
    logger.info("BIC: %.2f", model.bic(scaler.transform(df.loc[idx, feats].values)))
    logger.info("AIC: %.2f", model.aic(scaler.transform(df.loc[idx, feats].values)))


if __name__ == "__main__":
    main()
