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

from src.models.autoencoder import (  # noqa: E402
    AutoencoderConfig,
    fit_autoencoder,
    predict_autoencoder,
)
from src.utils.logging_utils import setup_logging  # noqa: E402


def main():
    # CLI entrypoint for training the autoencoder + KMeans model on the
    # feature matrix and exporting both regimes and latent embeddings.

    # Parse command-line arguments (paths to features and config file)
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="data/processed/features.csv")
    parser.add_argument("--config", type=str, default="configs/autoencoder.yaml")
    args = parser.parse_args()

    # Load autoencoder hyperparameters from YAML and construct the config object.
    with open(args.config, "r") as f:
        cfg_dict = yaml.safe_load(f)

    logger = setup_logging(name=__name__)

    # Build the AutoencoderConfig object from the YAML dictionary
    ae_cfg = AutoencoderConfig(
        encoding_dim=int(cfg_dict.get("encoding_dim", 4)),
        hidden_dims=tuple(cfg_dict.get("hidden_dims", [8])),
        n_clusters=int(cfg_dict.get("n_clusters", 2)),
        epochs=int(cfg_dict.get("epochs", 100)),
        batch_size=int(cfg_dict.get("batch_size", 32)),
        learning_rate=float(cfg_dict.get("learning_rate", 1e-3)),
        random_state=int(cfg_dict.get("random_state", 42)),
        features=tuple(cfg_dict.get("features")) if cfg_dict.get("features") else None,
    )

    # Load features, train the autoencoder + KMeans, and obtain regimes
    # together with the compressed embeddings.
    df = pd.read_csv(args.features, parse_dates=[0], index_col=0)

    # Train autoencoder + KMeans on the selected features
    ae_model, kmeans, idx, scaler, feats = fit_autoencoder(df, ae_cfg)
    regimes, embeddings = predict_autoencoder(ae_model, kmeans, df, scaler, feats)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "autoencoder_regimes.csv"
    # Save regime labels and latent embeddings to CSV files in results/
    regimes.to_csv(out_csv, header=True)
    logger.info("Saved Autoencoder regimes -> %s", out_csv)

    # Optionally save embeddings so they can be inspected or used by other
    # models (e.g. dimensionality reduction visualisations).
    emb_csv = out_dir / "autoencoder_embeddings.csv"
    emb_df = pd.DataFrame(
        embeddings, index=idx, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )
    emb_df.to_csv(emb_csv)
    logger.info("Saved embeddings -> %s", emb_csv)


if __name__ == "__main__":
    main()
