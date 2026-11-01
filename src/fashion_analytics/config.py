"""Central configuration: paths, business constants, and ML feature spec."""
from __future__ import annotations

from pathlib import Path

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for _dir in (FIGURES_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DATASET_FILES: dict[str, str] = {
    "customers": "dataset_fashion_store_customers.csv",
    "products": "dataset_fashion_store_products.csv",
    "sales": "dataset_fashion_store_sales.csv",
    "sale_items": "dataset_fashion_store_salesitems.csv",
    "stock": "dataset_fashion_store_stock.csv",
    "campaigns": "dataset_fashion_store_campaigns.csv",
    "channels": "dataset_fashion_store_channels.csv",
}

# ============================================================
# BUSINESS CONSTANTS
# ============================================================
SUPPLIER_COUNTRIES: list[str] = ["France", "Germany"]
SALES_ONLY_COUNTRIES: list[str] = ["Italy", "Netherlands", "Spain", "Portugal"]
HOLIDAY_DAYS: list[int] = [1, 2, 3, 7, 8, 9, 10, 11, 12]

# ============================================================
# FEATURE ENGINEERING
# ============================================================
LAG_DAYS: list[int] = [1, 2, 3, 7]
ROLLING_WINDOWS: list[int] = [3, 7]

LAG_FEATURES = [f"revenue_lag_{lag}" for lag in LAG_DAYS]
ROLLING_MEAN_FEATURES = [f"revenue_rolling_mean_{w}" for w in ROLLING_WINDOWS]
ROLLING_STD_FEATURES = [f"revenue_rolling_std_{w}" for w in ROLLING_WINDOWS]

NUMERICAL_FEATURES = (
    ["week_of_year"]
    + LAG_FEATURES
    + ROLLING_MEAN_FEATURES
    + ROLLING_STD_FEATURES
    + [
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_holiday",
        "campaign_active",
    ]
)

CATEGORICAL_FEATURES: list[str] = ["day_of_week", "month", "quarter"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# MODELING
# ============================================================
TRAIN_SPLIT_RATIO = 0.70
RANDOM_STATE = 42
CV_SPLITS = 3