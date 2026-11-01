"""Generate summary reports and save them as CSV files."""
from __future__ import annotations

from typing import Any

import logging

import pandas as pd

from fashion_analytics.config import REPORTS_DIR

logger = logging.getLogger(__name__)


def save_eda_report(eda: dict[str, Any]) -> None:
    """Persist key EDA tables as CSVs in outputs/reports/."""
    tables_to_save = {
        "country": "eda_country.csv",
        "category": "eda_category.csv",
        "channel": "eda_channel.csv",
        "monthly": "eda_monthly.csv",
        "top_products": "eda_top_products.csv",
        "age_range": "eda_age_range.csv",
    }
    for key, filename in tables_to_save.items():
        if key in eda and isinstance(eda[key], pd.DataFrame):
            path = REPORTS_DIR / filename
            eda[key].to_csv(path)
            logger.info("Saved report: %s", path)


def save_rfm_report(rfm: pd.DataFrame) -> None:
    """Save RFM customer segments to CSV."""
    path = REPORTS_DIR / "rfm_segments.csv"
    rfm.to_csv(path, index=False)
    logger.info("Saved report: %s", path)


def save_model_results(results: list[Any]) -> None:
    """Save model metrics to CSV."""
    rows = [
        {"model": r.name, "mae": r.mae, "rmse": r.rmse, "r2": r.r2}
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    path = REPORTS_DIR / "model_results.csv"
    df.to_csv(path, index=False)
    logger.info("Saved report: %s", path)