from __future__ import annotations

import argparse
import pathlib as pl
import sys

import yaml

# Ensure project root is on sys.path when running this file directly so that
# we can import the feature engineering code from the src package.
# This allows the script to run correctly even when executed from the scripts/ folder.
ROOT = pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the core feature engineering logic (dataclass + processing function)
# from the src package. This script only acts as a thin CLI wrapper.
from src.features.build_features import FeatureConfig, build_features  # noqa: E402


def main():

    # Small CLI wrapper that reads configuration from YAML and then calls
    # the pure Python feature-building function.
    # This script does not perform any feature calculations itself.
    # Instead, it delegates all processing to src/features/build_features.py.

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # Read the selected YAML config file
    cfg_path = pl.Path(args.config)
    with cfg_path.open("r") as f:
        base = yaml.safe_load(f)

    # Convert YAML values into a strongly typed FeatureConfig dataclass.
    # Defaults are used whenever the YAML file does not define a field.
    symbols = tuple(base.get("symbols", ("BTC-USD", "ETH-USD", "^GSPC", "^VIX")))
    feat_cfg = FeatureConfig(
        raw_dir=base.get("output_dir", "data/raw"),
        processed_dir="data/processed",
        symbols=symbols,
        ret_kind=base.get("ret_kind", "log"),
        vol_window=int(base.get("vol_window", 21)),
        vol_annualize=bool(base.get("vol_annualize", True)),
        vol_trading_days=int(base.get("vol_trading_days", 252)),
        vol_z_window=int(base.get("vol_z_window", 60)),
        corr_window=int(base.get("corr_window", 21)),
    )

    # Entry point: allows running the script via
    # python -m scripts.build_features --config configs/default.yaml
    build_features(feat_cfg)


if __name__ == "__main__":
    main()
