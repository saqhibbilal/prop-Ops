"""
Training configuration for property price prediction model.
"""

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

# Data paths
TRAIN_DATA_PATH = 'data/processed/train.csv'
VAL_DATA_PATH = 'data/processed/val.csv'
TEST_DATA_PATH = 'data/processed/test.csv'

# MLflow settings
MLFLOW_TRACKING_URI = 'file:./mlruns'
EXPERIMENT_NAME = 'property_price_prediction'
EXPERIMENT_NAME_COMPARISON = 'property_price_model_comparison'
