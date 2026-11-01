"""Command-line entry point for the Fashion Analytics pipeline."""
from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # headless backend for saving figures

from fashion_analytics.analysis import eda, rfm
from fashion_analytics.cleaning.dates import fix_swapped_dates, parse_dates
from fashion_analytics.cleaning.discounts import validate_discounts
from fashion_analytics.cleaning.merge import build_analysis_table
from fashion_analytics.config import TRAIN_SPLIT_RATIO
from fashion_analytics.io.loader import load_datasets, quality_report
from fashion_analytics.logging_setup import setup_logging
from fashion_analytics.modeling.features import add_features, build_daily
from fashion_analytics.modeling.trainer import train_and_evaluate
from fashion_analytics.reporting.figures import save_bar, save_line
from fashion_analytics.reporting.report_builder import (
    save_eda_report,
    save_model_results,
    save_rfm_report,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full data pipeline end-to-end."""
    setup_logging("INFO")
    logger.info("=" * 60)
    logger.info("FASHION ANALYTICS PIPELINE")
    logger.info("=" * 60)

    # 1. Load
    tables = load_datasets()
    quality_report(tables)

    # 2. Clean
    parse_dates(tables)
    tables["sale_items"], n_fixed = fix_swapped_dates(
        tables["sale_items"], tables["sales"]
    )
    logger.info("Fixed %d swapped dates", n_fixed)

    df = build_analysis_table(tables)
    df = validate_discounts(df)

    # 3. EDA
    eda_results = eda.full_eda(df)
    save_eda_report(eda_results)

    save_bar(
        eda_results["country"],
        "revenue",
        "Revenue by Country",
        "country_revenue.png",
        rotate=0,
    )
    save_bar(
        eda_results["category"],
        "revenue",
        "Revenue by Category",
        "category_revenue.png",
    )
    save_line(
        eda_results["monthly"],
        "year_month",
        "revenue",
        "Monthly Revenue Trend",
        "monthly_revenue.png",
    )

    # 4. RFM
    rfm_df = rfm.compute_rfm(df)
    save_rfm_report(rfm_df)
    save_bar(
        rfm_df["segment"].value_counts().to_frame("count"),
        "count",
        "Customer Segments",
        "rfm_segments.png",
        rotate=30,
    )

    # 5. Features
    daily = build_daily(df)
    daily_feat = add_features(daily, tables["campaigns"])

    split_idx = int(len(daily_feat) * TRAIN_SPLIT_RATIO)
    train = daily_feat.iloc[:split_idx].copy().reset_index(drop=True)
    test = daily_feat.iloc[split_idx:].copy().reset_index(drop=True)

    # 6. Model
    trained, results, baseline = train_and_evaluate(
        train, test, tables["campaigns"]
    )
    save_model_results(results)

    best = min(results, key=lambda r: r.mae)
    logger.info("BEST MODEL: %s (MAE=%.2f, R2=%.4f)", best.name, best.mae, best.r2)
    logger.info("BASELINE: %s (MAE=%.2f)", baseline["name"], baseline["mae"])

    # 7. Final forecast plot
    import matplotlib.pyplot as plt
    
    from fashion_analytics.config import FIGURES_DIR

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test["sale_date"], test["revenue"], label="Actual", marker="o", markersize=3)
    ax.plot(
        test["sale_date"],
        best.predictions,
        label=f"Forecast ({best.name})",
        marker="x",
        markersize=3,
    )
    ax.set_title("Actual vs Forecasted Revenue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()