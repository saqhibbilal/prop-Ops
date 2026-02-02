"""
Training script for property price prediction model with MLflow tracking.
"""

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from training.config import MODEL_PARAMS, TRAIN_DATA_PATH, VAL_DATA_PATH, MLFLOW_TRACKING_URI, EXPERIMENT_NAME


def train_model():
    """Train XGBoost model with MLflow tracking."""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    
    # Separate features and target
    X_train = train_df.drop('price', axis=1)
    y_train = train_df['price']
    X_val = val_df.drop('price', axis=1)
    y_val = val_df['price']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(MODEL_PARAMS)
        
        # Train model
        print("Training model...")
        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # Log metrics
        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("train_r2", train_r2)
        mlflow.log_metric("val_r2", val_r2)
        
        # Log feature importance as artifact
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        print("\nTraining Results:")
        print(f"Train RMSE: ${train_rmse:,.2f}")
        print(f"Val RMSE: ${val_rmse:,.2f}")
        print(f"Train MAE: ${train_mae:,.2f}")
        print(f"Val MAE: ${val_mae:,.2f}")
        print(f"Train R²: {train_r2:.4f}")
        print(f"Val R²: {val_r2:.4f}")
        
        print(f"\nMLflow run ID: {mlflow.active_run().info.run_id}")
        print(f"Model logged to MLflow")


if __name__ == '__main__':
    train_model()
