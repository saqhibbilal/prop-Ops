"""
API configuration.
"""

import os

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "property_price_prediction")
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", None)
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID", None)

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
