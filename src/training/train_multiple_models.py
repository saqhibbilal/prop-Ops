"""
Train multiple models (LightGBM, Random Forest, Linear Regression, Ridge, Lasso)
on the same data and log all to MLflow for comparison.
"""

import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME_COMPARISON,
    MULTIPLE_MODEL_CONFIGS,
)

# Model class mapping for sklearn models
SKLEARN_MODEL_CLASSES = {
    "Random Forest": RandomForestRegressor,
    "Linear Regression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
}


def train_all_models():
    """Train all configured models and log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME_COMPARISON)

    print("Loading data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    X_train = train_df.drop("price", axis=1)
    y_train = train_df["price"]
    X_val = val_df.drop("price", axis=1)
    y_val = val_df["price"]

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    results = []

    for model_name, cfg in MULTIPLE_MODEL_CONFIGS.items():
        flavor = cfg["flavor"]
        params = dict(cfg["params"])

        with mlflow.start_run(tags={"model_type": model_name}):
            mlflow.log_param("model_type", model_name)

            if flavor == "lightgbm":
                import lightgbm as lgb

                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train)
                mlflow.log_params(params)
                mlflow.lightgbm.log_model(model, "model")
            else:
                model_class = SKLEARN_MODEL_CLASSES[model_name]
                model = model_class(**params)
                model.fit(X_train, y_train)
                mlflow.log_params(params)
                mlflow.sklearn.log_model(model, "model")

            y_val_pred = model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            mlflow.log_metric("val_rmse", val_rmse)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("val_r2", val_r2)

            run_id = mlflow.active_run().info.run_id
            results.append(
                {
                    "model_type": model_name,
                    "run_id": run_id,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_r2": val_r2,
                }
            )
            print(f"  {model_name}: val_rmse=${val_rmse:,.2f}, val_r2={val_r2:.4f}")

    return results


if __name__ == "__main__":
    train_all_models()
