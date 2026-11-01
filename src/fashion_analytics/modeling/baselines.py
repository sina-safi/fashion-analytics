"""Naive and seasonal baselines for revenue forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Repeat the last observed value for every future step."""
    history = train["revenue"].tolist()
    preds: list[float] = []
    for _ in range(len(test)):
        preds.append(float(history[-1]))
        history.append(history[-1])
    return np.array(preds)


def seasonal_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Use the same weekday from the previous week; fallback to naive."""
    naive = naive_forecast(train, test)
    preds: list[float] = []
    for i, date in enumerate(test["sale_date"]):
        week_ago = date - pd.Timedelta(days=7)
        match = train.loc[train["sale_date"] == week_ago, "revenue"]
        preds.append(float(match.iloc[0]) if len(match) else float(naive[i]))
    return np.array(preds)