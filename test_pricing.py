"""
Integration test for pricing API endpoints.
Run this after starting the API: uvicorn src.api.app:app
"""

import requests
import json

BASE_URL = "http://localhost:8000"

# Sample property for pricing tests
sample_property = {
    "area_sqft": 1500.0,
    "bedrooms": 3,
    "bathrooms": 2.0,
    "age": 10,
    "has_parking": 1,
    "has_gym": 1,
    "has_pool": 0,
    "property_type_House": 1,
    "property_type_Apartment": 0,
    "property_type_Condo": 0,
    "property_type_Townhouse": 0,
    "location_Downtown": 0,
    "location_Rural": 0,
    "location_Suburbs": 1,
    "location_Urban": 0,
}

print("=" * 60)
print("Testing Pricing API Endpoints")
print("=" * 60)

# Test 1: Owner Recommendation
print("\n1. Testing /pricing/recommend (Owner)...")
try:
    response = requests.post(
        f"{BASE_URL}/pricing/recommend",
        json={
            "property": sample_property,
            "market_position": "market"
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Recommended Price: ${data['recommended_price']:,.0f}")
        print(f"   Base Price: ${data['base_price']:,.0f}")
        print(f"   Demand Level: {data['demand_level']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Renter Alert
print("\n2. Testing /pricing/alert (Renter)...")
try:
    response = requests.post(
        f"{BASE_URL}/pricing/alert",
        json={
            "property": sample_property,
            "asking_price": 400000.0
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Is Fair: {data['is_fair']}")
        print(f"   Fair Range: ${data['fair_low']:,.0f} - ${data['fair_high']:,.0f}")
        print(f"   Message: {data['message']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Investor Opportunity
print("\n3. Testing /pricing/opportunity (Investor)...")
try:
    response = requests.post(
        f"{BASE_URL}/pricing/opportunity",
        json={
            "property": sample_property,
            "min_roi_pct": 8.0
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Score: {data['score']}/100")
        print(f"   Suggested Bid: ${data['suggested_bid']:,.0f}")
        print(f"   Expected Value: ${data['expected_value']:,.0f}")
        print(f"   Meets ROI: {data['meets_roi']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Current Dynamic Price
print("\n4. Testing /pricing/current (Dynamic Price)...")
try:
    response = requests.post(
        f"{BASE_URL}/pricing/current",
        json={
            "property": sample_property
        }
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Current Price: ${data['current_price']:,.0f}")
        print(f"   Base Price: ${data['base_price']:,.0f}")
        print(f"   Demand Multiplier: {data['demand_multiplier']:.2f}x")
        print(f"   Demand Level: {data['demand_level']}")
    else:
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Pricing API Tests Complete")
print("=" * 60)
