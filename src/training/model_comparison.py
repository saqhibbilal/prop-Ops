"""
Load and compare multiple models from MLflow (metrics and predictions).
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    TEST_DATA_PATH,
    VAL_DATA_PATH,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME_COMPARISON,
)


def get_comparison_runs(experiment_name=None, limit_per_model=1):
    """
    Get latest run per model_type from the comparison experiment.
    Returns list of (run_id, model_type) for the best run per type (by val_r2).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp_name = experiment_name or EXPERIMENT_NAME_COMPARISON
    try:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            return []
    except Exception:
        return []

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.val_r2 DESC"],
    )

    if runs.empty:
        return []

    # One run per model_type (best by val_r2)
    seen = set()
    selected = []
    for _, row in runs.iterrows():
        mt = row.get("tags.model_type") or row.get("model_type")
        if pd.isna(mt):
            continue
        if mt not in seen:
            seen.add(mt)
            selected.append({"run_id": row["run_id"], "model_type": mt})
        if len(seen) >= 5:
            break

    return selected


def load_model_by_run_id(run_id):
    """Load model from MLflow run. Supports sklearn and lightgbm flavors."""
    model_uri = f"runs:/{run_id}/model"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        pass
    try:
        return mlflow.lightgbm.load_model(model_uri)
    except Exception:
        pass
    return None


def get_metrics_table(use_test_data=True):
    """
    Get a DataFrame of metrics for each model (from MLflow run metrics).
    If use_test_data=True, also evaluate on test set and include test metrics.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs_info = get_comparison_runs()

    if not runs_info:
        return pd.DataFrame()

    rows = []
    for r in runs_info:
        run_id = r["run_id"]
        model_type = r["model_type"]
        run = mlflow.get_run(run_id)
        metrics = run.data.metrics
        rows.append(
            {
                "model_type": model_type,
                "run_id": run_id,
                "val_rmse": metrics.get("val_rmse"),
                "val_mae": metrics.get("val_mae"),
                "val_r2": metrics.get("val_r2"),
            }
        )

    df = pd.DataFrame(rows)

    if use_test_data and not df.empty:
        try:
            test_df = pd.read_csv(TEST_DATA_PATH)
            X_test = test_df.drop("price", axis=1)
            y_test = test_df["price"]
        except Exception:
            return df

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        test_metrics = []
        for r in runs_info:
            model = load_model_by_run_id(r["run_id"])
            if model is None:
                test_metrics.append({"test_rmse": None, "test_mae": None, "test_r2": None})
                continue
            pred = model.predict(X_test)
            test_metrics.append(
                {
                    "test_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                    "test_mae": mean_absolute_error(y_test, pred),
                    "test_r2": r2_score(y_test, pred),
                }
            )
        for i, col in enumerate(["test_rmse", "test_mae", "test_r2"]):
            df[col] = [m[col] for m in test_metrics]

    return df


def get_predictions_comparison(sample_size=200):
    """
    For each model, get predictions on the same subset of data.
    Returns (X_sample, y_true, predictions_df) where predictions_df has columns model_type and prediction.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    runs_info = get_comparison_runs()
    if not runs_info:
        return None, None, pd.DataFrame()

    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
    except Exception:
        try:
            test_df = pd.read_csv(VAL_DATA_PATH)
        except Exception:
            return None, None, pd.DataFrame()

    X = test_df.drop("price", axis=1)
    y = test_df["price"]
    n = min(sample_size, len(X))
    X_sample = X.sample(n=n, random_state=42)
    y_sample = y.loc[X_sample.index]

    preds = []
    for r in runs_info:
        model = load_model_by_run_id(r["run_id"])
        if model is None:
            continue
        p = model.predict(X_sample)
        preds.append(pd.DataFrame({"model_type": r["model_type"], "prediction": p}))

    if not preds:
        return X_sample, y_sample, pd.DataFrame()

    predictions_df = pd.concat(preds, ignore_index=True)
    return X_sample, y_sample, predictions_df
