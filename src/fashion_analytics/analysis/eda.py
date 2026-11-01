"""Exploratory data analysis on the merged sales table."""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def overall_summary(df: pd.DataFrame) -> dict[str, float]:
    """Return headline totals for the sales table.

    Args:
        df: Merged analysis DataFrame.

    Returns:
        Dict with total sales, quantity, revenue, profit, and averages.
    """
    summary = {
        "total_sales": int(len(df)),
        "total_quantity": int(df["quantity"].sum()),
        "total_revenue": float(df["item_total"].sum()),
        "total_profit": float(df["profit"].sum()),
        "avg_item_total": float(df["item_total"].mean()),
        "avg_profit": float(df["profit"].mean()),
    }
    logger.info("Overall summary: %s", summary)
    return summary


def _group_agg(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregate revenue and profit by a single column."""
    out = (
        df.groupby(by)
        .agg(
            sales_count=("item_id", "count"),
            quantity_sold=("quantity", "sum"),
            revenue=("item_total", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )
    out["profit_margin"] = out["profit"] / out["revenue"] * 100
    return out


def by_country(df: pd.DataFrame) -> pd.DataFrame:
    return _group_agg(df, "country")


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    return _group_agg(df, "category")


def by_channel(df: pd.DataFrame) -> pd.DataFrame:
    return _group_agg(df, "channel_sale")


def by_month(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly revenue and profit, sorted by year-month."""
    df = df.copy()
    df["year"] = df["sale_date"].dt.year
    df["month"] = df["sale_date"].dt.month

    out = (
        df.groupby(["year", "month"])
        .agg(
            sales_count=("item_id", "count"),
            revenue=("item_total", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    out["year_month"] = pd.to_datetime(
        out["year"].astype(str) + "-" + out["month"].astype(str) + "-01"
    )
    return out.sort_values("year_month").reset_index(drop=True)


def top_products(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-n products by revenue."""
    return (
        df.groupby(["product_id", "product_name"])
        .agg(
            quantity_sold=("quantity", "sum"),
            revenue=("item_total", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(n)
    )


def by_age_range(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("age_range")
        .agg(
            customers=("customer_id", "nunique"),
            sales=("item_id", "count"),
            revenue=("item_total", "sum"),
            profit=("profit", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )


def full_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Run the full EDA suite and return all aggregated tables."""
    return {
        "summary": overall_summary(df),
        "country": by_country(df),
        "category": by_category(df),
        "channel": by_channel(df),
        "monthly": by_month(df),
        "top_products": top_products(df),
        "age_range": by_age_range(df),
    }