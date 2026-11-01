"""Merge all raw tables into one analysis-ready DataFrame."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from fashion_analytics.config import SUPPLIER_COUNTRIES

logger = logging.getLogger(__name__)


def build_analysis_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge sale_items, sales, customers, products, and stock.

    Also derives ``profit``, ``sale_date``, and ``stock_status``.

    Args:
        tables: Mapping of table name -> DataFrame.

    Returns:
        Merged analysis DataFrame.
    """
    logger.info("Merging analysis table...")

    df = tables["sale_items"].merge(
        tables["sales"],
        on="sale_id",
        how="left",
        suffixes=("_item", "_sale"),
    )
    df = df.merge(
        tables["customers"],
        on="customer_id",
        how="left",
        suffixes=("", "_customer"),
    )
    df = df.merge(
        tables["products"],
        on="product_id",
        how="left",
        suffixes=("", "_product"),
    )
    df = df.merge(
        tables["stock"][["product_id", "country", "stock_quantity"]],
        on=["product_id", "country"],
        how="left",
    )

    df["stock_status"] = np.where(
        df["country"].isin(SUPPLIER_COUNTRIES),
        np.where(
            df["stock_quantity"].isna(),
            "Missing Supplier Stock",
            "Supplier Stock Available",
        ),
        "Sales Market - No Local Stock",
    )

    df["sale_date"] = df["sale_date_sale"]
    df["profit"] = df["item_total"] - df["quantity"] * df["cost_price"]

    logger.info("Merged table shape: %s", df.shape)
    return df