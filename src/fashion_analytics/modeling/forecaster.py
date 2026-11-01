"""Recursive multi-step forecasting using a fitted pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any

from fashion_analytics.config import (
    FEATURE_COLUMNS,
    HOLIDAY_DAYS,
    LAG_DAYS,
    ROLLING_WINDOWS,
)


def _is_campaign_active(date: pd.Timestamp, campaigns: pd.DataFrame) -> int:
    for _, row in campaigns.iterrows():
        if row["start_date"] <= date <= row["end_date"]:
            return 1
    return 0


def recursive_forecast(
    model: Any,
    history_df: pd.DataFrame,
    future_df: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> np.ndarray:
    """Forecast revenue one day at a time, feeding predictions back in.

    Args:
        model: Fitted scikit-learn Pipeline (preprocessor + regressor).
        history_df: Training DataFrame with ``sale_date`` and ``revenue``.
        future_df: DataFrame with ``sale_date`` (only dates are used).
        campaigns: Campaigns table.

    Returns:
        Array of predicted revenues with length ``len(future_df)``.
    """
    history = (
        history_df[["sale_date", "revenue"]]
        .sort_values("sale_date")
        .reset_index(drop=True)
    )
    preds: list[float] = []

    for date in future_df["sale_date"]:
        revenues = history["revenue"].tolist()
        weekday = date.dayofweek

        row: dict[str, float | int] = {
            "day_of_week": weekday,
            "month": date.month,
            "quarter": date.quarter,
            "week_of_year": int(date.isocalendar().week),
            "is_weekend": int(weekday >= 5),
            "is_month_start": int(date.is_month_start),
            "is_month_end": int(date.is_month_end),
            "is_holiday": int(date.day in HOLIDAY_DAYS),
            "campaign_active": _is_campaign_active(date, campaigns),
        }
        for lag in LAG_DAYS:
            row[f"revenue_lag_{lag}"] = revenues[-lag]
        for w in ROLLING_WINDOWS:
            recent = revenues[-w:]
            row[f"revenue_rolling_mean_{w}"] = float(np.mean(recent))
            row[f"revenue_rolling_std_{w}"] = (
                float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0
            )

        x_row = pd.DataFrame([row])[FEATURE_COLUMNS]
        pred = float(model.predict(x_row)[0])
        preds.append(pred)

        history = pd.concat(
            [history, pd.DataFrame({"sale_date": [date], "revenue": [pred]})],
            ignore_index=True,
        )

    return np.array(preds)