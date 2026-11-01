"""Unit tests for data cleaning functions."""
import pandas as pd
import pytest
from fashion_analytics.cleaning.dates import fix_swapped_dates
from fashion_analytics.cleaning.discounts import validate_discounts


def test_fix_swapped_dates_finds_and_fixes_swap():
    """Test that day/month swapped dates are correctly identified and fixed."""
    sale_items = pd.DataFrame({
        "item_id": [1, 2],
        "sale_id": [10, 11],
        "sale_date": pd.to_datetime(["2024-03-01", "2024-05-12"]),
    })
    sales = pd.DataFrame({
        "sale_id": [10, 11],
        "sale_date": pd.to_datetime(["2024-01-03", "2024-05-12"]),
    })

    fixed_sale_items, n_fixed = fix_swapped_dates(sale_items, sales)

    assert n_fixed == 1
    assert fixed_sale_items.loc[0, "sale_date"] == pd.Timestamp("2024-01-03")
    assert fixed_sale_items.loc[1, "sale_date"] == pd.Timestamp("2024-05-12")


def test_validate_discounts_calculates_difference():
    """Test that discount difference is calculated correctly."""
    df = pd.DataFrame({
        "discount_percent": ["10%", "20%"],
        "original_price": [100.0, 50.0],
        "discount_applied": [10.0, 12.0],  # 20% of 50 is 10, so 12 is wrong
    })

    result_df = validate_discounts(df, tolerance=0.02)

    assert "discount_difference" in result_df.columns
    assert result_df.loc[0, "discount_difference"] == pytest.approx(0.0, abs=0.01)
    assert result_df.loc[1, "discount_difference"] == pytest.approx(2.0, abs=0.01)
