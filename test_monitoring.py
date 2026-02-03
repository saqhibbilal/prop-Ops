"""
Test script for monitoring and drift detection.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from monitoring.monitor import PredictionMonitor
from monitoring.drift_detector import DriftDetector

# Test monitoring store
print("Testing Monitoring Store...")
monitor = PredictionMonitor()

# Log some test predictions
print("\nLogging test predictions...")
test_features = [
    {
        "area_sqft": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "age": 10,
        "has_parking": 1,
        "has_gym": 1,
        "has_pool": 0,
        "property_type_House": 1,
        "location_Suburbs": 1
    },
    {
        "area_sqft": 2000,
        "bedrooms": 4,
        "bathrooms": 3,
        "age": 5,
        "has_parking": 1,
        "has_gym": 1,
        "has_pool": 1,
        "property_type_Condo": 1,
        "location_Downtown": 1
    }
]

for i, features in enumerate(test_features):
    prediction = 300000 + i * 50000  # Mock prediction
    monitor.log_prediction(
        features=features,
        prediction=prediction,
        model_version="test_run_001"
    )
    print(f"Logged prediction {i+1}: ${prediction:,.2f}")

# Retrieve recent predictions
print("\nRetrieving recent predictions...")
recent = monitor.get_recent_data(limit=10)
print(f"Retrieved {len(recent)} predictions")
if not recent.empty:
    print(recent[['timestamp', 'prediction', 'model_version']].head())

# Test drift detection
print("\nTesting Drift Detection...")
try:
    detector = DriftDetector()
    
    # Get current data from monitoring
    current_data = monitor.get_features_dataframe(limit=50)
    
    if not current_data.empty:
        # Remove non-feature columns
        feature_cols = [col for col in current_data.columns 
                       if col not in ['prediction', 'timestamp', 'ground_truth']]
        current_features = current_data[feature_cols] if feature_cols else pd.DataFrame()
        
        if not current_features.empty:
            print(f"Checking drift with {len(current_features)} samples...")
            drift_results = detector.check_drift(current_data=current_features)
            print(f"Overall drift detected: {drift_results.get('overall_drift_detected', False)}")
            print(f"Status: {drift_results.get('status', 'unknown')}")
            
            if 'data_drift' in drift_results:
                dd = drift_results['data_drift']
                print(f"Data drift detected: {dd.get('drift_detected', False)}")
                print(f"Drift score: {dd.get('drift_score', 0.0):.4f}")
        else:
            print("No feature data available for drift detection")
    else:
        print("No recent data available for drift detection")
        
except Exception as e:
    print(f"Error testing drift detection: {e}")
    import traceback
    traceback.print_exc()

print("\nMonitoring test complete!")
