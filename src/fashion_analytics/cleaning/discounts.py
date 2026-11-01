"""Validate discount_percent vs discount_applied consistency."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def validate_discounts(df: pd.DataFrame, tolerance: float = 0.02) -> pd.DataFrame:
    """Flag rows whose discount differs from the expected percent-based value.

    Args:
        df: Merged analysis DataFrame with ``discount_percent``,
            ``original_price``, ``discount_applied``.
        tolerance: Absolute difference above which a row is flagged.

    Returns:
        The input DataFrame with a new ``discount_difference`` column.
    """
    df = df.copy()

    pct = df["discount_percent"].str.rstrip("%").astype(float)
    expected = df["original_price"] * pct / 100
    df["discount_difference"] = df["discount_applied"] - expected

    bad = int((df["discount_difference"].abs() > tolerance).sum())
    logger.info(
        "Discount validation: %d rows exceed tolerance %.3f", bad, tolerance
    )
    return df