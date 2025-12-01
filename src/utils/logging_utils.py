"""Logging utilities for scripts.

This module centralises logging configuration so that all command-line
scripts in the project produce consistent, timestamped log messages.
"""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: int = logging.INFO, name: Optional[str] = None) -> logging.Logger:
    """Configure basic logging and return a logger for the caller.

    The idea is that each script calls this once at start-up and then uses
    the returned logger instead of raw ``print`` statements.

    Args:
        level: Logging level (e.g. logging.INFO, logging.DEBUG).
        name: Optional logger name. If None, the root logger is returned.

    Returns:
        Configured logger instance.
    """

    # Basic configuration: timestamp, severity level, logger name, and message text.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=level,
    )
    return logging.getLogger(name)
