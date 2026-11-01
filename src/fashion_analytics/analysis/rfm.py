"""RFM customer segmentation."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _segment(score: int) -> str:
    if score >= 10:
        return "Champions"
    if score >= 8:
        return "Loyal Customers"
    if score >= 6:
        return "Potential Loyalists"
    if score >= 4:
        return "At Risk"
    return "Lost Customers"


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency / Frequency / Monetary and 4-quantile scores.

    Args:
        df: Merged analysis table with ``customer_id``, ``sale_id``,
            ``sale_date``, and ``item_total``.

    Returns:
        Per-customer DataFrame with ``R_score``, ``F_score``, ``M_score``,
        ``RFM_score``, and ``segment``.
    """
    reference = df["sale_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency=("sale_date", lambda x: (reference - x.max()).days),
            Frequency=("sale_id", "nunique"),
            Monetary=("item_total", "sum"),
        )
        .reset_index()
    )

    rfm["R_score"] = pd.qcut(
        rfm["Recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]
    )
    rfm["F_score"] = pd.qcut(
        rfm["Frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
    )
    rfm["M_score"] = pd.qcut(
        rfm["Monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]
    )

    rfm["RFM_score"] = (
        rfm["R_score"].astype(int)
        + rfm["F_score"].astype(int)
        + rfm["M_score"].astype(int)
    )
    rfm["segment"] = rfm["RFM_score"].apply(_segment)

    logger.info("RFM segments: %s", rfm["segment"].value_counts().to_dict())
    return rfm