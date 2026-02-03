"""
Retraining script for property price prediction model.
"""

import sys
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    MODEL_PARAMS, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME
)
from training.train import train_model


def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def compare_models(current_run_id: str, new_run_id: str) -> dict:
    """
    Compare current production model with newly trained model.
    
    Args:
        current_run_id: MLflow run ID of current production model
        new_run_id: MLflow run ID of newly trained model
    
    Returns:
        Dictionary with comparison results
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load test data
    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df.drop('price', axis=1)
    y_test = test_df['price']
    
    # Load models
    current_model_uri = f"runs:/{current_run_id}/model"
    new_model_uri = f"runs:/{new_run_id}/model"
    
    current_model = mlflow.xgboost.load_model(current_model_uri)
    new_model = mlflow.xgboost.load_model(new_model_uri)
    
    # Evaluate both models
    current_metrics = evaluate_model_performance(current_model, X_test, y_test)
    new_metrics = evaluate_model_performance(new_model, X_test, y_test)
    
    # Compare
    rmse_improvement = current_metrics['rmse'] - new_metrics['rmse']
    mae_improvement = current_metrics['mae'] - new_metrics['mae']
    r2_improvement = new_metrics['r2'] - current_metrics['r2']
    
    # Determine if new model is better
    # New model is better if RMSE/MAE decrease OR R2 increases
    is_better = (
        rmse_improvement > 0 or  # Lower RMSE is better
        mae_improvement > 0 or   # Lower MAE is better
        r2_improvement > 0.01    # R2 improvement > 1%
    )
    
    return {
        'current_metrics': current_metrics,
        'new_metrics': new_metrics,
        'rmse_improvement': rmse_improvement,
        'mae_improvement': mae_improvement,
        'r2_improvement': r2_improvement,
        'is_better': is_better,
        'should_promote': is_better
    }


def retrain_model() -> str:
    """
    Retrain the model with latest data.
    
    Returns:
        MLflow run ID of the new model
    """
    print("Starting model retraining...")
    
    # Train new model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run() as run:
        # Load data
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VAL_DATA_PATH)
        
        X_train = train_df.drop('price', axis=1)
        y_train = train_df['price']
        X_val = val_df.drop('price', axis=1)
        y_val = val_df['price']
        
        # Train model
        import xgboost as xgb
        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        y_val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Log metrics
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_r2", val_r2)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        run_id = run.info.run_id
        print(f"Model retrained successfully. Run ID: {run_id}")
        print(f"Validation RMSE: ${val_rmse:,.2f}")
        print(f"Validation MAE: ${val_mae:,.2f}")
        print(f"Validation R²: {val_r2:.4f}")
        
        return run_id


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain property price prediction model')
    parser.add_argument('--compare-with', type=str, help='MLflow run ID of current production model')
    
    args = parser.parse_args()
    
    # Retrain
    new_run_id = retrain_model()
    
    # Compare if current model provided
    if args.compare_with:
        comparison = compare_models(args.compare_with, new_run_id)
        print("\nModel Comparison:")
        print(f"Current RMSE: ${comparison['current_metrics']['rmse']:,.2f}")
        print(f"New RMSE: ${comparison['new_metrics']['rmse']:,.2f}")
        print(f"RMSE Improvement: ${comparison['rmse_improvement']:,.2f}")
        print(f"\nShould Promote: {comparison['should_promote']}")
        
        if comparison['should_promote']:
            print("✅ New model performs better. Ready for promotion.")
        else:
            print("❌ New model does not improve performance. Keep current model.")
