# Automated Retraining Pipeline

This directory contains the automated retraining pipeline for the PropTech ML model.

## Components

### 1. Retraining Script (`src/pipelines/retrain.py`)

- Retrains the model with latest data
- Compares new model with production model
- Returns metrics and promotion recommendation

### 2. Model Promotion (`src/pipelines/promote_model.py`)

- Promotes models to production in MLflow registry
- Manages model versions

### 3. Simple Pipeline Runner (`src/pipelines/run_retraining.py`)

- Standalone script that runs the complete pipeline
- Can be scheduled with cron (Linux) or Task Scheduler (Windows)
- No Airflow required

### 4. Airflow DAG (`pipelines/retraining_dag.py`)

- Full Airflow DAG for orchestration
- Requires Airflow setup

## Usage

### Simple Pipeline (Recommended for Learning)

Run the standalone pipeline:

```bash
python src/pipelines/run_retraining.py
```

This will:

1. Check for data drift
2. Retrain the model
3. Compare with production model
4. Promote if better

### Schedule with Task Scheduler (Windows)

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger (e.g., weekly)
4. Action: Start a program
5. Program: `python`
6. Arguments: `C:\path\to\prop-ops\src\pipelines\run_retraining.py`
7. Start in: `C:\path\to\prop-ops`

### Airflow Setup (Advanced)

1. Initialize Airflow:

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
```

2. Create admin user:

```bash
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com
```

3. Copy DAG:

```bash
cp pipelines/retraining_dag.py $AIRFLOW_HOME/dags/
```

4. Start Airflow:

```bash
airflow webserver --port 8080
airflow scheduler
```

5. Access UI at http://localhost:8080

## Pipeline Steps

1. **Drift Detection**: Checks if data drift is detected
2. **Retraining**: Trains new model with latest data
3. **Comparison**: Compares new model with production
4. **Promotion**: Promotes if new model is better

## Model Comparison Criteria

New model is promoted if:

- RMSE decreases, OR
- MAE decreases, OR
- R² increases by >1%

## Notes

- The pipeline logs all runs to MLflow
- Models are versioned in MLflow Model Registry
- Production model can be loaded via MLflow API
