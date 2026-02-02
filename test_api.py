"""
Simple script to test the API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Test health endpoint
print("Testing /health endpoint...")
response = requests.get(f"{BASE_URL}/health")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test single prediction
print("Testing /predict endpoint...")
property_data = {
    "area_sqft": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "age": 10,
    "has_parking": 1,
    "has_gym": 1,
    "has_pool": 0,
    "property_type_House": 1,
    "location_Suburbs": 1
}

response = requests.post(f"{BASE_URL}/predict", json=property_data)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}\n")

# Test batch prediction
print("Testing /predict/batch endpoint...")
batch_data = {
    "properties": [
        property_data,
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
}

response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
