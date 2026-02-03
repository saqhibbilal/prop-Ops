"""
Monitoring data store for logging predictions and features.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json

from .config import MONITORING_DB


class MonitoringStore:
    """SQLite-based monitoring data store."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(MONITORING_DB)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                features TEXT NOT NULL,
                prediction REAL NOT NULL,
                ground_truth REAL,
                model_version TEXT
            )
        """)
        
        # Drift metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drift_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                threshold REAL,
                status TEXT NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
    
    def log_prediction(
        self,
        features: Dict,
        prediction: float,
        ground_truth: Optional[float] = None,
        model_version: Optional[str] = None
    ):
        """Log a prediction to the store."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (timestamp, features, prediction, ground_truth, model_version)
            VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            json.dumps(features),
            prediction,
            ground_truth,
            model_version
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_predictions(self, limit: int = 100) -> pd.DataFrame:
        """Get recent predictions."""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT * FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=(limit,))
        
        conn.close()
        
        # Parse features JSON
        if not df.empty and 'features' in df.columns:
            df['features'] = df['features'].apply(json.loads)
        
        return df
    
    def log_drift_metric(
        self,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        status: str
    ):
        """Log drift detection metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO drift_metrics (timestamp, metric_type, metric_name, metric_value, threshold, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            metric_type,
            metric_name,
            metric_value,
            threshold,
            status
        ))
        
        conn.commit()
        conn.close()
    
    def get_drift_metrics(self, limit: int = 50) -> pd.DataFrame:
        """Get recent drift metrics."""
        conn = sqlite3.connect(self.db_path)
        
        df = pd.read_sql_query("""
            SELECT * FROM drift_metrics
            ORDER BY timestamp DESC
            LIMIT ?
        """, conn, params=(limit,))
        
        conn.close()
        return df
