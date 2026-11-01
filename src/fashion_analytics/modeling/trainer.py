"""Train and evaluate revenue forecasting models."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fashion_analytics.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    RANDOM_STATE,
)
from fashion_analytics.modeling.baselines import (
    naive_forecast,
    seasonal_forecast,
)
from fashion_analytics.modeling.evaluation import compute_metrics
from fashion_analytics.modeling.forecaster import recursive_forecast

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Holds metrics and predictions for one model."""

    name: str
    mae: float
    rmse: float
    r2: float
    predictions: np.ndarray


def build_preprocessor() -> ColumnTransformer:
    """Standard-scale numerics; one-hot-encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_models() -> dict[str, object]:
    """Instantiate the three candidate models."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    campaigns: pd.DataFrame,
) -> tuple[dict[str, Pipeline], list[ModelResult], dict[str, float | str]]:
    """Train all models, evaluate on test, and compute the best baseline.

    Args:
        train: Training DataFrame (features + ``revenue``).
        test: Test DataFrame (``sale_date`` and ``revenue``).
        campaigns: Campaigns table.

    Returns:
        Tuple ``(trained_pipelines, results, baseline_summary)``.
    """
    train_clean = train.dropna().reset_index(drop=True)
    x_train = train_clean[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_clean["revenue"]

    trained: dict[str, Pipeline] = {}
    results: list[ModelResult] = []

    for name, model in get_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        pipe.fit(x_train, y_train)
        trained[name] = pipe

        preds = recursive_forecast(pipe, train, test, campaigns)
        metrics = compute_metrics(test["revenue"], preds)
        results.append(
            ModelResult(
                name=name,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                predictions=preds,
            )
        )
        logger.info(
            "%-18s MAE=%.2f RMSE=%.2f R2=%.4f",
            name,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    # Baselines
    naive = naive_forecast(train, test)
    seasonal = seasonal_forecast(train, test)
    naive_mae = compute_metrics(test["revenue"], naive)["mae"]
    seasonal_mae = compute_metrics(test["revenue"], seasonal)["mae"]

    if seasonal_mae < naive_mae:
        baseline_name = "Seasonal"
        baseline_mae = seasonal_mae
    else:
        baseline_name = "Naive"
        baseline_mae = naive_mae

    logger.info("Baseline (%s) MAE=%.2f", baseline_name, baseline_mae)

    baseline_summary = {
        "name": baseline_name,
        "mae": baseline_mae,
    }

    return trained, results, baseline_summary