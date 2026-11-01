"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a clean, timestamped format.

    Args:
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    for noisy in ("matplotlib", "PIL", "xgboost", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)