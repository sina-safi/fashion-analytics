"""Unit tests for feature engineering."""
import pandas as pd

from fashion_analytics.modeling.features import add_features


def test_add_features_calendar_and_lags():
    """Test that calendar and lag features are generated correctly."""
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    daily = pd.DataFrame({
        "sale_date": dates,
        "revenue": [100.0] * 10,
        "transactions": [5] * 10,
    })
    
    campaigns = pd.DataFrame(columns=["start_date", "end_date"])

    df_feat = add_features(daily, campaigns)

    assert "day_of_week" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    assert "is_month_start" in df_feat.columns
    
    assert "revenue_lag_1" in df_feat.columns
    assert "revenue_lag_7" in df_feat.columns
    
    assert pd.isna(df_feat.loc[0, "revenue_lag_1"])
    assert df_feat.loc[1, "revenue_lag_1"] == 100.0
    assert df_feat.loc[7, "revenue_lag_7"] == 100.0
