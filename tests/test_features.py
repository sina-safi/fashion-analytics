"""Unit tests for feature engineering."""
import pandas as pd
import pytest
from fashion_analytics.modeling.features import add_features


def test_add_features_calendar_and_lags():
    """Test that calendar and lag features are generated correctly."""
    # ساخت یک سری زمانی روزانه ساختگی
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    daily = pd.DataFrame({
        "sale_date": dates,
        "revenue": [100.0] * 10,
        "transactions": [5] * 10,
    })
    
    # دیتای کمپین خالی (هیچ کمپینی فعال نیست)
    campaigns = pd.DataFrame(columns=["start_date", "end_date"])

    df_feat = add_features(daily, campaigns)

    # بررسی فیچرهای تقویمی
    assert "day_of_week" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    assert "is_month_start" in df_feat.columns
    
    # بررسی فیچرهای Lag (تأخیر)
    assert "revenue_lag_1" in df_feat.columns
    assert "revenue_lag_7" in df_feat.columns
    
    # بررسی درستی مقادیر Lag
    # روز اول باید NaN باشه چون lag_1 نداره
    assert pd.isna(df_feat.loc[0, "revenue_lag_1"])
    # روز دوم باید مقدار روز اول (100) رو داشته باشه
    assert df_feat.loc[1, "revenue_lag_1"] == 100.0
    # روز هشتم باید مقدار روز اول (100) رو به عنوان lag_7 داشته باشه
    assert df_feat.loc[7, "revenue_lag_7"] == 100.0
