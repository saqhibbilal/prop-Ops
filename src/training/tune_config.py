"""
Configuration for hyperparameter tuning with Optuna (XGBoost and LightGBM).
"""

# Number of Optuna trials per model
N_TRIALS_XGBOOST = 15
N_TRIALS_LIGHTGBM = 15

# Search spaces (param name -> (low, high) or list of choices)
XGBOOST_SEARCH_SPACE = {
    "n_estimators": (50, 200),
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 7),
}

LIGHTGBM_SEARCH_SPACE = {
    "n_estimators": (50, 200),
    "max_depth": (3, 12),
    "learning_rate": (0.01, 0.3),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_samples": (5, 50),
}

# Fixed params (not tuned)
RANDOM_STATE = 42
