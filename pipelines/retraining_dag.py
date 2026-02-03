"""
Airflow DAG for automated model retraining pipeline.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipelines.retrain import retrain_model, compare_models
from pipelines.promote_model import promote_model, get_production_model_version
from monitoring.drift_detector import DriftDetector


def check_drift_task():
    """Check for data drift."""
    detector = DriftDetector()
    monitor = __import__('monitoring.monitor', fromlist=['PredictionMonitor']).PredictionMonitor()
    
    current_data = monitor.get_features_dataframe(limit=100)
    if not current_data.empty:
        feature_cols = [col for col in current_data.columns 
                       if col not in ['prediction', 'timestamp', 'ground_truth']]
        current_features = current_data[feature_cols] if feature_cols else None
        
        if current_features is not None and not current_features.empty:
            drift_results = detector.check_drift(current_data=current_features)
            drift_detected = drift_results.get('overall_drift_detected', False)
            
            if drift_detected:
                print("⚠️ Drift detected! Proceeding with retraining...")
                return True
            else:
                print("✅ No drift detected. Skipping retraining.")
                return False
    
    print("⚠️ Not enough data for drift detection. Proceeding with retraining.")
    return True


def retrain_task(**context):
    """Retrain the model."""
    run_id = retrain_model()
    context['ti'].xcom_push(key='new_run_id', value=run_id)
    return run_id


def compare_and_promote_task(**context):
    """Compare models and promote if better."""
    new_run_id = context['ti'].xcom_pull(key='new_run_id')
    
    # Get current production model
    prod_model = get_production_model_version()
    current_run_id = None
    
    if prod_model:
        # Extract run_id from model version
        # In production, you'd store run_id in model tags
        print(f"Current production model: {prod_model.name} v{prod_model.version}")
        # For now, we'll need to pass current_run_id differently
        # This is a simplified version
    
    # If we have a current run_id, compare
    if current_run_id:
        comparison = compare_models(current_run_id, new_run_id)
        
        if comparison['should_promote']:
            print("✅ New model is better. Promoting to production...")
            promote_model(new_run_id)
        else:
            print("❌ New model is not better. Keeping current model.")
    else:
        # First time - promote anyway
        print("No existing production model. Promoting new model...")
        promote_model(new_run_id)


# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    description='Automated model retraining pipeline with drift detection',
    schedule_interval=timedelta(days=7),  # Run weekly
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'retraining', 'drift-detection'],
)

# Define tasks
check_drift = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift_task,
    dag=dag,
)

retrain = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_task,
    dag=dag,
)

compare_and_promote = PythonOperator(
    task_id='compare_and_promote',
    python_callable=compare_and_promote_task,
    dag=dag,
)

# Define task dependencies
check_drift >> retrain >> compare_and_promote
