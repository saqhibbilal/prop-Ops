"""
Training configuration for property price prediction model.
"""

from pathlib import Path

# Project root (parent of src/) for resolving paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# XGBoost model parameters (used by train.py)
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Multiple models for comparison (Phase 2)
MULTIPLE_MODEL_CONFIGS = {
    "LightGBM": {
        "flavor": "lightgbm",
        "params": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
    },
    "Random Forest": {
        "flavor": "sklearn",
        "params": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
    },
    "Linear Regression": {
        "flavor": "sklearn",
        "params": {},
    },
    "Ridge": {
        "flavor": "sklearn",
        "params": {"alpha": 1.0, "random_state": 42},
    },
    "Lasso": {
        "flavor": "sklearn",
        "params": {"alpha": 1.0, "random_state": 42},
    },
}

# Data paths (absolute so dashboard/training find data from any cwd)
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_PATH = str(DATA_DIR / "train.csv")
VAL_DATA_PATH = str(DATA_DIR / "val.csv")
TEST_DATA_PATH = str(DATA_DIR / "test.csv")

# MLflow settings (absolute path so dashboard finds mlruns when run from any cwd)
MLRUNS_DIR = _PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = MLRUNS_DIR.resolve().as_uri()
EXPERIMENT_NAME = 'property_price_prediction'
EXPERIMENT_NAME_COMPARISON = 'property_price_model_comparison'
