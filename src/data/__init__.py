"""Data loading utilities for the project.

This subpackage contains helpers to download and normalise raw market data
into a consistent tabular format used by the feature engineering step.
"""

from .yfinance_loader import download_and_save

__all__ = ["download_and_save"]
