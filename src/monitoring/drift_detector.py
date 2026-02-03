"""
Drift detection using Evidently AI.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple

# Lazy import flag - will be set when Evidently is actually imported
EVIDENTLY_AVAILABLE = None

def _check_evidently():
    """Lazy check for Evidently availability."""
    global EVIDENTLY_AVAILABLE
    if EVIDENTLY_AVAILABLE is None:
        try:
            # Try to import Evidently - will fail if asyncio issues occur
            from evidently import ColumnMapping
            from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
            from evidently.report import Report
            
            EVIDENTLY_AVAILABLE = True
        except (ImportError, RuntimeError) as e:
            EVIDENTLY_AVAILABLE = False
            # Don't print in Streamlit context - let caller handle it
    
    return EVIDENTLY_AVAILABLE

from .data_store import MonitoringStore
from .config import (
    REFERENCE_DATA_PATH,
    DATA_DRIFT_THRESHOLD,
    PREDICTION_DRIFT_THRESHOLD,
    MONITORING_WINDOW
)


class DriftDetector:
    """Detect data and prediction drift using Evidently AI."""
    
    def __init__(self, reference_data_path: str = None):
        self.reference_data_path = reference_data_path or REFERENCE_DATA_PATH
        self.reference_data = self._load_reference_data()
        self.store = MonitoringStore()
    
    def _load_reference_data(self) -> pd.DataFrame:
        """Load reference (training) dataset."""
        try:
            df = pd.read_csv(self.reference_data_path)
            # Drop target if present
            if 'price' in df.columns:
                df = df.drop('price', axis=1)
            return df
        except Exception as e:
            print(f"Warning: Could not load reference data: {e}")
            return pd.DataFrame()
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = None
    ) -> Dict:
        """
        Detect data drift in features.
        
        Args:
            current_data: Current production data
            threshold: Drift threshold (default from config)
        
        Returns:
            Dictionary with drift detection results
        """
        if not _check_evidently():
            return {
                "drift_detected": False,
                "message": "Evidently AI not available. Install with: pip install evidently"
            }
        
        # Import here to avoid asyncio issues
        from evidently import ColumnMapping
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report
        
        if self.reference_data.empty or current_data.empty:
            return {
                "drift_detected": False,
                "message": "Reference or current data is empty"
            }
        
        threshold = threshold or DATA_DRIFT_THRESHOLD
        
        # Ensure columns match
        common_cols = set(self.reference_data.columns) & set(current_data.columns)
        if not common_cols:
            return {
                "drift_detected": False,
                "message": "No common columns between reference and current data"
            }
        
        ref_subset = self.reference_data[list(common_cols)]
        current_subset = current_data[list(common_cols)]
        
        try:
            # Create column mapping
            column_mapping = ColumnMapping()
            column_mapping.numerical_features = [
                col for col in common_cols 
                if ref_subset[col].dtype in ['int64', 'float64']
            ]
            column_mapping.categorical_features = [
                col for col in common_cols 
                if col not in column_mapping.numerical_features
            ]
            
            # Generate drift report
            data_drift_report = Report(metrics=[DataDriftPreset()])
            data_drift_report.run(
                reference_data=ref_subset,
                current_data=current_subset,
                column_mapping=column_mapping
            )
            
            # Extract drift metrics
            report_dict = data_drift_report.as_dict()
            
            # Check if drift detected
            drift_detected = False
            drift_score = 0.0
            
            if 'metrics' in report_dict:
                for metric in report_dict['metrics']:
                    if 'dataset_drift' in metric:
                        drift_detected = metric['dataset_drift']
                    if 'drift_score' in metric:
                        drift_score = metric.get('drift_score', 0.0)
            
            result = {
                "drift_detected": drift_detected or drift_score > threshold,
                "drift_score": drift_score,
                "threshold": threshold,
                "status": "drift" if (drift_detected or drift_score > threshold) else "no_drift",
                "report": report_dict
            }
            
            # Log metric
            self.store.log_drift_metric(
                metric_type="data_drift",
                metric_name="dataset_drift_score",
                metric_value=drift_score,
                threshold=threshold,
                status=result["status"]
            )
            
            return result
            
        except Exception as e:
            return {
                "drift_detected": False,
                "error": str(e),
                "message": f"Error detecting drift: {e}"
            }
    
    def detect_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
        threshold: float = None
    ) -> Dict:
        """
        Detect prediction drift.
        
        Args:
            reference_predictions: Reference prediction distribution
            current_predictions: Current prediction distribution
            threshold: Drift threshold
        
        Returns:
            Dictionary with prediction drift results
        """
        if not _check_evidently():
            return {
                "drift_detected": False,
                "message": "Evidently AI not available. Install with: pip install evidently"
            }
        
        # Import here to avoid asyncio issues
        from evidently.metric_preset import TargetDriftPreset
        from evidently.report import Report
        
        threshold = threshold or PREDICTION_DRIFT_THRESHOLD
        
        if reference_predictions.empty or current_predictions.empty:
            return {
                "drift_detected": False,
                "message": "Reference or current predictions are empty"
            }
        
        try:
            # Create DataFrames for Evidently
            ref_df = pd.DataFrame({'prediction': reference_predictions})
            current_df = pd.DataFrame({'prediction': current_predictions})
            
            # Generate drift report
            target_drift_report = Report(metrics=[TargetDriftPreset()])
            target_drift_report.run(
                reference_data=ref_df,
                current_data=current_df
            )
            
            report_dict = target_drift_report.as_dict()
            
            # Extract drift metrics
            drift_detected = False
            drift_score = 0.0
            
            if 'metrics' in report_dict:
                for metric in report_dict['metrics']:
                    if 'target_drift' in metric:
                        drift_detected = metric['target_drift']
                    if 'drift_score' in metric:
                        drift_score = metric.get('drift_score', 0.0)
            
            result = {
                "drift_detected": drift_detected or drift_score > threshold,
                "drift_score": drift_score,
                "threshold": threshold,
                "status": "drift" if (drift_detected or drift_score > threshold) else "no_drift",
                "report": report_dict
            }
            
            # Log metric
            self.store.log_drift_metric(
                metric_type="prediction_drift",
                metric_name="prediction_drift_score",
                metric_value=drift_score,
                threshold=threshold,
                status=result["status"]
            )
            
            return result
            
        except Exception as e:
            return {
                "drift_detected": False,
                "error": str(e),
                "message": f"Error detecting prediction drift: {e}"
            }
    
    def check_drift(
        self,
        current_data: Optional[pd.DataFrame] = None,
        window_size: int = None
    ) -> Dict:
        """
        Check for both data and prediction drift.
        
        Args:
            current_data: Current production data (if None, uses recent predictions)
            window_size: Number of recent predictions to analyze
        
        Returns:
            Dictionary with drift check results
        """
        window_size = window_size or MONITORING_WINDOW
        
        # Get recent predictions if current_data not provided
        if current_data is None:
            monitor = __import__('monitoring.monitor', fromlist=['PredictionMonitor']).PredictionMonitor()
            current_data = monitor.get_features_dataframe(limit=window_size)
            
            if current_data.empty:
                return {
                    "drift_detected": False,
                    "message": "No recent data available for drift detection"
                }
            
            # Extract predictions
            current_predictions = current_data['prediction'] if 'prediction' in current_data.columns else pd.Series()
            current_features = current_data.drop(['prediction', 'timestamp'], axis=1, errors='ignore')
        else:
            current_features = current_data
            current_predictions = pd.Series()
        
        results = {
            "data_drift": {},
            "prediction_drift": {}
        }
        
        # Check data drift
        if not current_features.empty:
            results["data_drift"] = self.detect_data_drift(current_features)
        
        # Check prediction drift if we have predictions
        if not current_predictions.empty and len(current_predictions) >= 10:
            # Use reference predictions from training (if available)
            # For now, use a sample from reference data predictions
            # In production, you'd store reference predictions separately
            reference_predictions = pd.Series()  # Would be loaded from stored reference predictions
            
            if not reference_predictions.empty:
                results["prediction_drift"] = self.detect_prediction_drift(
                    reference_predictions,
                    current_predictions
                )
        
        # Overall drift status
        data_drift = results["data_drift"].get("drift_detected", False)
        pred_drift = results["prediction_drift"].get("drift_detected", False)
        
        results["overall_drift_detected"] = data_drift or pred_drift
        results["status"] = "drift" if results["overall_drift_detected"] else "no_drift"
        
        return results
