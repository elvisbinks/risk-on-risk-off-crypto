from __future__ import annotations

import argparse
import pathlib as pl
import sys

import yaml

# Ensure project root is on sys.path when running as a module/script so that
# ``src`` can be imported even when the script is executed directly.
ROOT = pl.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.yfinance_loader import download_and_save  # noqa: E402


def main():
    # Small CLI that reads a YAML config and downloads all requested symbols.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # Load the configuration that defines symbols and date range to fetch.
    cfg_path = pl.Path(args.config)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    start = cfg.get("start_date")
    end = cfg.get("end_date")
    interval = cfg.get("interval", "1d")
    symbols = cfg.get("symbols", [])
    out_dir = cfg.get("output_dir", "data/raw")

    if not symbols:
        # Failing early is better than silently producing an empty data folder.
        raise ValueError("No symbols specified in config.")

    # Delegate actual downloading and saving logic to the data loader module.
    download_and_save(symbols, start=start, end=end, interval=interval, out_dir=out_dir)


if __name__ == "__main__":
    main()
