"""Date parsing and fixing swapped day/month values."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def parse_dates(tables: dict[str, pd.DataFrame]) -> None:
    """Convert date-like columns to datetime, in place.

    Args:
        tables: Mapping of table name -> DataFrame (mutated in place).
    """
    tables["customers"]["signup_date"] = pd.to_datetime(
        tables["customers"]["signup_date"]
    )
    for key in ("sales", "sale_items"):
        tables[key]["sale_date"] = pd.to_datetime(tables[key]["sale_date"])
    for col in ("start_date", "end_date"):
        tables["campaigns"][col] = pd.to_datetime(tables["campaigns"][col])
    logger.info("Parsed date columns: customers, sales, sale_items, campaigns")


def fix_swapped_dates(
    sale_items: pd.DataFrame, sales: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """Fix sale_item dates whose day and month were swapped during input.

    For each sale_item, if swapping day and month makes its date match the
    parent sale's date, we consider it fixable and correct it.

    Args:
        sale_items: Table with ``item_id``, ``sale_id``, ``sale_date``.
        sales: Table with ``sale_id``, ``sale_date``.

    Returns:
        Tuple ``(fixed_sale_items, n_fixed)``.
    """
    sale_items = sale_items.copy()

    check = sale_items.merge(
        sales[["sale_id", "sale_date"]],
        on="sale_id",
        how="left",
        suffixes=("_item", "_sale"),
    )

    mismatched = check[check["sale_date_item"] != check["sale_date_sale"]].copy()
    mismatched["swapped"] = pd.to_datetime(
        mismatched["sale_date_item"].dt.strftime("%Y-%d-%m"),
        errors="coerce",
    )
    fixable = mismatched[mismatched["swapped"] == mismatched["sale_date_sale"]]

    sale_items.loc[
        sale_items["item_id"].isin(fixable["item_id"]), "sale_date"
    ] = fixable["swapped"].values

    logger.info(
        "Date fix: %d mismatched, %d fixable", len(mismatched), len(fixable)
    )
    return sale_items, len(fixable)