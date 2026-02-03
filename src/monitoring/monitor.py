"""
Monitoring service to log predictions and track data.
"""

import pandas as pd
from typing import Dict, Optional
from datetime import datetime

from .data_store import MonitoringStore
from .config import MONITORING_DB


class PredictionMonitor:
    """Monitor and log predictions from the API."""
    
    def __init__(self, db_path: str = None):
        self.store = MonitoringStore(db_path)
    
    def log_prediction(
        self,
        features: Dict,
        prediction: float,
        ground_truth: Optional[float] = None,
        model_version: Optional[str] = None
    ):
        """
        Log a prediction.
        
        Args:
            features: Input features dictionary
            prediction: Model prediction
            ground_truth: Actual value (if available)
            model_version: Model version/run_id
        """
        self.store.log_prediction(
            features=features,
            prediction=prediction,
            ground_truth=ground_truth,
            model_version=model_version
        )
    
    def get_recent_data(self, limit: int = 100) -> pd.DataFrame:
        """Get recent prediction data."""
        return self.store.get_recent_predictions(limit=limit)
    
    def get_features_dataframe(self, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions as features DataFrame."""
        predictions_df = self.get_recent_data(limit=limit)
        
        if predictions_df.empty:
            return pd.DataFrame()
        
        # Extract features from JSON
        features_list = []
        for _, row in predictions_df.iterrows():
            features = row['features']
            features['prediction'] = row['prediction']
            features['timestamp'] = row['timestamp']
            if pd.notna(row['ground_truth']):
                features['ground_truth'] = row['ground_truth']
            features_list.append(features)
        
        return pd.DataFrame(features_list)
