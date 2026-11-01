"""Feature engineering for daily revenue forecasting."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fashion_analytics.config import HOLIDAY_DAYS, LAG_DAYS, ROLLING_WINDOWS

logger = logging.getLogger(__name__)


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the sales table into a daily revenue/transactions series.

    Args:
        df: Merged analysis DataFrame.

    Returns:
        DataFrame with ``sale_date``, ``revenue``, ``transactions``.
    """
    daily = (
        df.set_index("sale_date")
        .resample("D")
        .agg(revenue=("item_total", "sum"), transactions=("item_id", "count"))
        .reset_index()
        .sort_values("sale_date")
        .reset_index(drop=True)
    )
    daily["revenue"] = daily["revenue"].fillna(0.0)
    daily["transactions"] = daily["transactions"].fillna(0).astype(int)
    logger.info(
        "Daily series: %s, %s -> %s",
        daily.shape,
        daily["sale_date"].min(),
        daily["sale_date"].max(),
    )
    return daily


def add_features(daily: pd.DataFrame, campaigns: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, lag, rolling, and campaign features.

    Args:
        daily: Daily series with ``sale_date`` and ``revenue``.
        campaigns: Campaigns table with ``start_date``, ``end_date``.

    Returns:
        DataFrame with engineered features (may contain NaNs from lags).
    """
    df = daily.copy().set_index("sale_date").sort_index()
    idx = df.index

    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["week_of_year"] = idx.isocalendar().week.astype(int)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_month_start"] = idx.is_month_start.astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    df["is_holiday"] = idx.day.isin(HOLIDAY_DAYS).astype(int)

    for lag in LAG_DAYS:
        df[f"revenue_lag_{lag}"] = df["revenue"].shift(lag)

    for w in ROLLING_WINDOWS:
        df[f"revenue_rolling_mean_{w}"] = df["revenue"].shift(1).rolling(w).mean()
        df[f"revenue_rolling_std_{w}"] = df["revenue"].shift(1).rolling(w).std()

    campaign_active = np.zeros(len(df), dtype=int)
    for _, c in campaigns.iterrows():
        mask = (idx >= c["start_date"]) & (idx <= c["end_date"])
        campaign_active[mask] = 1
    df["campaign_active"] = campaign_active

    return df.reset_index()