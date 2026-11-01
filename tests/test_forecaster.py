"""Unit tests for forecasting logic."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from fashion_analytics.modeling.forecaster import recursive_forecast


def test_recursive_forecast_returns_correct_shape_and_type():
    """Test that recursive forecast returns an array of correct length."""
    model = DummyRegressor(strategy="constant", constant=100)
    
    history_dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    history_df = pd.DataFrame({
        "sale_date": history_dates,
        "revenue": np.random.rand(30) * 100,
    })
    
    future_dates = pd.date_range(start="2024-01-31", periods=7, freq="D")
    future_df = pd.DataFrame({"sale_date": future_dates})
    
    campaigns = pd.DataFrame(columns=["start_date", "end_date"])
    
    x_dummy = np.zeros((30, 5))
    y_dummy = history_df["revenue"].values
    model.fit(x_dummy, y_dummy)
    
    preds = recursive_forecast(model, history_df, future_df, campaigns)
    
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 7
    assert all(pred == 100 for pred in preds)
