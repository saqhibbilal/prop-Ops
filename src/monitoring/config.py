"""
Monitoring configuration.
"""

import os
from pathlib import Path

# Monitoring data store
MONITORING_DIR = Path(os.getenv("MONITORING_DIR", "monitoring"))
MONITORING_DB = MONITORING_DIR / "monitoring.db"
MONITORING_DATA_DIR = MONITORING_DIR / "data"

# Reference dataset (training data)
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/processed/train.csv")

# Drift detection thresholds
DATA_DRIFT_THRESHOLD = float(os.getenv("DATA_DRIFT_THRESHOLD", "0.3"))  # PSI threshold
PREDICTION_DRIFT_THRESHOLD = float(os.getenv("PREDICTION_DRIFT_THRESHOLD", "0.2"))  # KS test p-value

# Monitoring window (number of recent predictions to analyze)
MONITORING_WINDOW = int(os.getenv("MONITORING_WINDOW", "100"))

# Create directories
MONITORING_DIR.mkdir(exist_ok=True)
MONITORING_DATA_DIR.mkdir(exist_ok=True)
