"""Unit tests for forecasting logic."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from fashion_analytics.modeling.forecaster import recursive_forecast


def test_recursive_forecast_returns_correct_shape_and_type():
    """Test that recursive forecast returns an array of correct length."""
    # ساخت مدل ساختگی که همیشه مقدار ثابت 100 رو پیش‌بینی می‌کنه
    model = DummyRegressor(strategy="constant", constant=100)
    
    # ساخت دیتای آموزش (تاریخچه)
    history_dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    history_df = pd.DataFrame({
        "sale_date": history_dates,
        "revenue": np.random.rand(30) * 100,
    })
    
    # ساخت دیتای آینده (که باید پیش‌بینی بشه)
    future_dates = pd.date_range(start="2024-01-31", periods=7, freq="D")
    future_df = pd.DataFrame({"sale_date": future_dates})
    
    # دیتای کمپین خالی
    campaigns = pd.DataFrame(columns=["start_date", "end_date"])
    
    # آموزش مدل ساختگی (چون DummyRegressor هست، صرفا fit میشه)
    # نکته: برای تست منطق، ما نیازی به فیچرهای واقعی نداریم چون مدل Dummy است
    X_dummy = np.zeros((30, 5))
    y_dummy = history_df["revenue"].values
    model.fit(X_dummy, y_dummy)
    
    # تابع پیش‌بینی رو صدا می‌زنیم
    preds = recursive_forecast(model, history_df, future_df, campaigns)
    
    # چک می‌کنیم که خروجی از نوع numpy array باشه و طولش ۷ باشه
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 7
    # چک می‌کنیم که همه پیش‌بینی‌ها عدد ۱۰۰ باشن (چون مدل Dummy)
    assert all(pred == 100 for pred in preds)
