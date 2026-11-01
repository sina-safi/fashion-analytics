"""Metric helpers for regression."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from typing import Any

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return MAE, RMSE, and R2 as floats.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        Dict with keys ``mae``, ``rmse``, ``r2``.
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }