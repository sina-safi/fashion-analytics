"""Load and validate raw CSV datasets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fashion_analytics.config import DATA_DIR, DATASET_FILES

logger = logging.getLogger(__name__)


def load_datasets(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all seven CSVs into a dict of DataFrames.

    Args:
        data_dir: Optional override for the data directory.

    Returns:
        Mapping of table_name -> DataFrame.

    Raises:
        FileNotFoundError: If any expected CSV is missing.
    """
    base = data_dir or DATA_DIR
    tables: dict[str, pd.DataFrame] = {}

    for name, filename in DATASET_FILES.items():
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset '{name}': {path}\n"
                f"Place all CSVs in {base} - see README.md."
            )
        tables[name] = pd.read_csv(path)
        logger.info("Loaded %-11s %s", name, tables[name].shape)

    return tables


def quality_report(tables: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """Compute missing/duplicate stats for each table."""
    report: dict[str, dict[str, Any]] = {}
    for name, df in tables.items():
        report[name] = {
            "shape": df.shape,
            "missing": int(df.isna().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
        }
        logger.info(
            "Quality %-11s shape=%s missing=%d duplicates=%d",
            name,
            df.shape,
            report[name]["missing"],
            report[name]["duplicates"],
        )
    return report