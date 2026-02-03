"""
Model promotion logic for updating production model.
"""

import os
import mlflow
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.config import MLFLOW_TRACKING_URI


def promote_model(run_id: str, model_name: str = "property_price_model"):
    """
    Promote a model to production by registering it in MLflow model registry.
    
    Args:
        run_id: MLflow run ID of the model to promote
        model_name: Name for the model in registry
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    model_uri = f"runs:/{run_id}/model"
    
    try:
        # Register model in MLflow Model Registry
        result = mlflow.register_model(model_uri, model_name)
        print(f"Model registered: {result.name} version {result.version}")
        
        # Transition to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        
        print(f"Model {model_name} version {result.version} promoted to Production")
        return result.version
        
    except Exception as e:
        print(f"Error promoting model: {e}")
        # If model registry not set up, just log the run_id for manual promotion
        print(f"Model run_id for manual promotion: {run_id}")
        return None


def get_production_model_version(model_name: str = "property_price_model"):
    """
    Get the current production model version.
    
    Returns:
        Model version info or None
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if latest_versions:
            return latest_versions[0]
        return None
    except Exception as e:
        print(f"Error getting production model: {e}")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Promote model to production')
    parser.add_argument('run_id', type=str, help='MLflow run ID to promote')
    parser.add_argument('--model-name', type=str, default='property_price_model',
                       help='Model name in registry')
    
    args = parser.parse_args()
    promote_model(args.run_id, args.model_name)
