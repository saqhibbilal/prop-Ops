"""
Hyperparameter tuning for XGBoost and LightGBM using Optuna.
Logs each trial to MLflow and returns best params and study history.
"""

import pandas as pd
import numpy as np
import mlflow
import optuna
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME_TUNING,
)
from training.tune_config import (
    N_TRIALS_XGBOOST,
    N_TRIALS_LIGHTGBM,
    XGBOOST_SEARCH_SPACE,
    LIGHTGBM_SEARCH_SPACE,
    RANDOM_STATE,
)

def _suggest_params(trial, space):
    """Suggest hyperparameters from a space dict."""
    params = {}
    for name, bounds in space.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            a, b = bounds
            if isinstance(a, int) and isinstance(b, int):
                params[name] = trial.suggest_int(name, a, b)
            else:
                params[name] = trial.suggest_float(name, float(a), float(b))
        else:
            params[name] = trial.suggest_categorical(name, list(bounds))
    return params


def _run_xgboost_trial(trial, X_train, y_train, X_val, y_val):
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error, r2_score

    params = _suggest_params(trial, XGBOOST_SEARCH_SPACE)
    params["random_state"] = RANDOM_STATE
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    return rmse, r2, params, "xgboost", model


def _run_lightgbm_trial(trial, X_train, y_train, X_val, y_val):
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error, r2_score

    params = _suggest_params(trial, LIGHTGBM_SEARCH_SPACE)
    params["random_state"] = RANDOM_STATE
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)
    return rmse, r2, params, "lightgbm", model


def run_tuning(model_name="both", n_trials_xgb=None, n_trials_lgb=None):
    """
    Run Optuna tuning for XGBoost and/or LightGBM.
    Logs each trial to MLflow under EXPERIMENT_NAME_TUNING.

    Args:
        model_name: "xgboost", "lightgbm", or "both"
        n_trials_xgb: override trials for XGBoost
        n_trials_lgb: override trials for LightGBM

    Returns:
        dict with keys: best_xgb (or None), best_lgb (or None), trials_xgb, trials_lgb,
        each trials_* is list of {number, value, params}.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME_TUNING)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    X_train = train_df.drop("price", axis=1)
    y_train = train_df["price"]
    X_val = val_df.drop("price", axis=1)
    y_val = val_df["price"]

    n_xgb = n_trials_xgb if n_trials_xgb is not None else N_TRIALS_XGBOOST
    n_lgb = n_trials_lgb if n_trials_lgb is not None else N_TRIALS_LIGHTGBM

    result = {"best_xgb": None, "best_lgb": None, "trials_xgb": [], "trials_lgb": []}

    def xgb_objective(trial):
        rmse, r2, params, _, _ = _run_xgboost_trial(trial, X_train, y_train, X_val, y_val)
        with mlflow.start_run(nested=True, tags={"model_type": "XGBoost"}):
            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_r2", r2)
        return rmse

    def lgb_objective(trial):
        rmse, r2, params, _, _ = _run_lightgbm_trial(trial, X_train, y_train, X_val, y_val)
        with mlflow.start_run(nested=True, tags={"model_type": "LightGBM"}):
            mlflow.log_params(params)
            mlflow.log_metric("val_rmse", rmse)
            mlflow.log_metric("val_r2", r2)
        return rmse

    if model_name in ("xgboost", "both"):
        with mlflow.start_run(tags={"tuning": "xgboost"}):
            study_xgb = optuna.create_study(direction="minimize", study_name="xgboost_tuning")
            study_xgb.optimize(xgb_objective, n_trials=n_xgb, show_progress_bar=True)
            result["best_xgb"] = study_xgb.best_params
            result["study_xgb"] = study_xgb
            result["trials_xgb"] = [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study_xgb.trials
            ]

    if model_name in ("lightgbm", "both"):
        with mlflow.start_run(tags={"tuning": "lightgbm"}):
            study_lgb = optuna.create_study(direction="minimize", study_name="lightgbm_tuning")
            study_lgb.optimize(lgb_objective, n_trials=n_lgb, show_progress_bar=True)
            result["best_lgb"] = study_lgb.best_params
            result["study_lgb"] = study_lgb
            result["trials_lgb"] = [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study_lgb.trials
            ]

    return result


def get_tuning_results():
    """
    Load latest tuning runs from MLflow (trials and best params) for display in dashboard.
    Returns dict with tuning_trials (DataFrame), best_xgb, best_lgb.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME_TUNING)
        if exp is None:
            return None
    except Exception:
        return None

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=500,
    )
    if runs.empty:
        return None

    tag_col = "tags.model_type"
    if tag_col not in runs.columns:
        return None

    # Only runs that have trial metrics (nested runs with model_type)
    runs = runs[runs["metrics.val_rmse"].notna()].copy()
    if runs.empty:
        return None

    trials = []
    for _, row in runs.iterrows():
        mt = row.get(tag_col)
        if pd.isna(mt) or str(mt) not in ("XGBoost", "LightGBM"):
            continue
        trials.append({
            "run_id": row["run_id"],
            "model_type": str(mt),
            "val_rmse": row["metrics.val_rmse"],
            "val_r2": row.get("metrics.val_r2"),
        })
    tuning_trials = pd.DataFrame(trials) if trials else pd.DataFrame()

    best_xgb = None
    best_lgb = None
    xgb_runs = runs[runs[tag_col] == "XGBoost"]
    lgb_runs = runs[runs[tag_col] == "LightGBM"]
    if not xgb_runs.empty:
        best_run_id = xgb_runs.loc[xgb_runs["metrics.val_rmse"].idxmin(), "run_id"]
        best_xgb = dict(mlflow.get_run(best_run_id).data.params)
    if not lgb_runs.empty:
        best_run_id = lgb_runs.loc[lgb_runs["metrics.val_rmse"].idxmin(), "run_id"]
        best_lgb = dict(mlflow.get_run(best_run_id).data.params)

    return {
        "tuning_trials": tuning_trials,
        "best_xgb": best_xgb,
        "best_lgb": best_lgb,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "lightgbm", "both"], default="both")
    parser.add_argument("--n-trials-xgb", type=int, default=None)
    parser.add_argument("--n-trials-lgb", type=int, default=None)
    args = parser.parse_args()
    run_tuning(model_name=args.model, n_trials_xgb=args.n_trials_xgb, n_trials_lgb=args.n_trials_lgb)
