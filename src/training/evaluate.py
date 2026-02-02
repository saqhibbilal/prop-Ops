"""
Evaluation script for property price prediction model.
"""

import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from training.config import TEST_DATA_PATH, MLFLOW_TRACKING_URI


def evaluate_model(model_uri: str = None, run_id: str = None):
    """
    Evaluate model on test set.
    
    Args:
        model_uri: MLflow model URI (e.g., "runs:/<run_id>/model")
        run_id: MLflow run ID (alternative to model_uri)
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop('price', axis=1)
    y_test = test_df['price']
    
    print(f"Test samples: {len(X_test)}")
    
    # Load model
    if run_id and not model_uri:
        model_uri = f"runs:/{run_id}/model"
    
    if not model_uri:
        raise ValueError("Either model_uri or run_id must be provided")
    
    print(f"Loading model from {model_uri}...")
    model = mlflow.xgboost.load_model(model_uri)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    import numpy as np
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print("\n" + "="*50)
    print("Test Set Evaluation Results:")
    print("="*50)
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")
    print("="*50)
    
    # Calculate percentage errors
    mape = (abs(y_test - y_pred) / y_test).mean() * 100
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--run-id', type=str, help='MLflow run ID')
    parser.add_argument('--model-uri', type=str, help='MLflow model URI')
    
    args = parser.parse_args()
    evaluate_model(model_uri=args.model_uri, run_id=args.run_id)
