"""
Training configuration for property price prediction model.
"""

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42
}

# Data paths
TRAIN_DATA_PATH = 'data/processed/train.csv'
VAL_DATA_PATH = 'data/processed/val.csv'
TEST_DATA_PATH = 'data/processed/test.csv'

# MLflow settings
MLFLOW_TRACKING_URI = 'file:./mlruns'
EXPERIMENT_NAME = 'property_price_prediction'
