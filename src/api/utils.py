"""
Utility functions for model loading and prediction.
"""

import mlflow
import pandas as pd
from typing import Optional
import os


def load_model(model_uri: Optional[str] = None, run_id: Optional[str] = None):
    """
    Load model from MLflow.
    
    Args:
        model_uri: MLflow model URI (e.g., "runs:/<run_id>/model")
        run_id: MLflow run ID (alternative to model_uri)
    
    Returns:
        Loaded XGBoost model
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    
    if run_id and not model_uri:
        model_uri = f"runs:/{run_id}/model"
    
    if not model_uri:
        # Try to get latest run from experiment
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "property_price_prediction")
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
                if not runs.empty:
                    latest_run_id = runs.iloc[0]["run_id"]
                    model_uri = f"runs:/{latest_run_id}/model"
        except Exception as e:
            print(f"Warning: Could not find latest run: {e}")
    
    if not model_uri:
        raise ValueError("Model URI or run_id must be provided, or set MLFLOW_MODEL_URI environment variable")
    
    print(f"Loading model from {model_uri}...")
    model = mlflow.xgboost.load_model(model_uri)
    print("Model loaded successfully!")
    return model


def prepare_features(property_data: dict) -> pd.DataFrame:
    """
    Prepare features from property input for model prediction.
    
    Args:
        property_data: Dictionary with property features
    
    Returns:
        DataFrame with prepared features
    """
    # Calculate derived features if not provided
    if property_data.get('price_per_sqft') is None:
        # This will be calculated, but for prediction we need a placeholder
        # In real scenario, this would be estimated or provided
        property_data['price_per_sqft'] = 0  # Will be ignored if model doesn't use it
    
    if property_data.get('total_rooms') is None:
        property_data['total_rooms'] = property_data.get('bedrooms', 0) + property_data.get('bathrooms', 0)
    
    # Ensure all required columns are present
    feature_order = [
        'area_sqft', 'bedrooms', 'bathrooms', 'age',
        'has_parking', 'has_gym', 'has_pool',
        'price_per_sqft', 'total_rooms',
        'property_type_Apartment', 'property_type_Condo', 
        'property_type_House', 'property_type_Townhouse',
        'location_Downtown', 'location_Rural', 
        'location_Suburbs', 'location_Urban'
    ]
    
    # Create DataFrame with proper column order
    df = pd.DataFrame([property_data])
    
    # Ensure all columns exist
    for col in feature_order:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[feature_order]
    
    return df
