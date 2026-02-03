"""
Simple script to run retraining pipeline without Airflow.
Can be scheduled with cron (Linux) or Task Scheduler (Windows).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.retrain import retrain_model, compare_models
from pipelines.promote_model import promote_model, get_production_model_version
from monitoring.drift_detector import DriftDetector
from monitoring.monitor import PredictionMonitor


def run_retraining_pipeline():
    """Run the complete retraining pipeline."""
    print("="*60)
    print("Starting Model Retraining Pipeline")
    print("="*60)
    
    # Step 1: Check for drift
    print("\n[Step 1] Checking for data drift...")
    detector = DriftDetector()
    monitor = PredictionMonitor()
    
    current_data = monitor.get_features_dataframe(limit=100)
    drift_detected = False
    
    if not current_data.empty:
        feature_cols = [col for col in current_data.columns 
                       if col not in ['prediction', 'timestamp', 'ground_truth']]
        current_features = current_data[feature_cols] if feature_cols else None
        
        if current_features is not None and not current_features.empty:
            drift_results = detector.check_drift(current_data=current_features)
            drift_detected = drift_results.get('overall_drift_detected', False)
            
            if drift_detected:
                print("[WARNING] Drift detected! Proceeding with retraining...")
            else:
                print("[OK] No drift detected.")
                print("Pipeline will continue anyway (can be modified to skip retraining)")
    else:
        print("[WARNING] Not enough data for drift detection. Proceeding with retraining...")
        drift_detected = True
    
    # Step 2: Retrain model
    print("\n[Step 2] Retraining model...")
    new_run_id = retrain_model()
    
    # Step 3: Compare with current production model
    print("\n[Step 3] Comparing with production model...")
    prod_model = get_production_model_version()
    
    if prod_model:
        print(f"Current production model: {prod_model.name} v{prod_model.version}")
        # In a real scenario, you'd retrieve the run_id from model tags/metadata
        # For now, we'll skip comparison if we can't get the run_id
        print("Note: Run ID comparison requires model metadata. Promoting new model.")
        should_promote = True
    else:
        print("No existing production model found. Promoting new model...")
        should_promote = True
    
    # Step 4: Promote if better
    if should_promote:
        print("\n[Step 4] Promoting model to production...")
        version = promote_model(new_run_id)
        if version:
            print(f"[SUCCESS] Model successfully promoted to production (version {version})")
        else:
            print(f"[INFO] Model run_id saved: {new_run_id} (manual promotion may be needed)")
    else:
        print("\n[Step 4] New model does not improve performance. Keeping current model.")
    
    print("\n" + "="*60)
    print("Retraining Pipeline Complete")
    print("="*60)


if __name__ == '__main__':
    run_retraining_pipeline()
