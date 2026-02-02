"""
Synthetic property data generator for PropTech MLOps project.
Generates realistic property listings with configurable drift simulation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


def generate_property_data(
    n_samples: int = 1000,
    start_date: Optional[str] = None,
    drift_factor: float = 0.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic property listing data.
    
    Args:
        n_samples: Number of property listings to generate
        start_date: Start date for listings (YYYY-MM-DD). If None, uses today.
        drift_factor: Market drift factor (0.0 = no drift, 1.0 = significant drift)
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with property features and target price
    """
    np.random.seed(random_seed)
    
    if start_date is None:
        start_date = datetime.now().strftime('%Y-%m-%d')
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Property types
    property_types = ['Apartment', 'House', 'Condo', 'Townhouse']
    
    # Locations (simplified)
    locations = ['Downtown', 'Suburbs', 'Urban', 'Rural']
    
    # Generate base features
    data = {
        'listing_date': [start + timedelta(days=np.random.randint(0, 365)) 
                        for _ in range(n_samples)],
        'property_type': np.random.choice(property_types, n_samples),
        'location': np.random.choice(locations, n_samples),
        'area_sqft': np.random.normal(1500, 500, n_samples).clip(500, 5000),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3, 4], n_samples, 
                                     p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'has_parking': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'has_gym': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'has_pool': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate price based on features with drift
    base_price = (
        df['area_sqft'] * 150 +  # Base price per sqft
        df['bedrooms'] * 30000 +
        df['bathrooms'] * 20000 +
        (2024 - df['year_built']) * -500 +  # Newer = more expensive
        df['has_parking'] * 15000 +
        df['has_gym'] * 5000 +
        df['has_pool'] * 25000
    )
    
    # Location premium
    location_premium = df['location'].map({
        'Downtown': 50000,
        'Urban': 30000,
        'Suburbs': 10000,
        'Rural': -10000
    })
    
    # Property type premium
    type_premium = df['property_type'].map({
        'House': 40000,
        'Townhouse': 20000,
        'Condo': 10000,
        'Apartment': 0
    })
    
    # Apply drift (market trend over time)
    days_since_start = (df['listing_date'] - start).dt.days
    drift_effect = base_price * (drift_factor * days_since_start / 365)
    
    # Final price with noise
    price = base_price + location_premium + type_premium + drift_effect
    price = price + np.random.normal(0, price * 0.1, n_samples)  # 10% noise
    price = price.clip(50000, 2000000)  # Reasonable bounds
    
    df['price'] = price.round(2)
    
    # Sort by listing date
    df = df.sort_values('listing_date').reset_index(drop=True)
    
    return df


if __name__ == '__main__':
    # Generate baseline dataset (no drift)
    print("Generating baseline dataset...")
    baseline_df = generate_property_data(n_samples=5000, drift_factor=0.0)
    baseline_df.to_csv('data/raw/baseline_properties.csv', index=False)
    print(f"Generated {len(baseline_df)} property listings")
    print(f"Price range: ${baseline_df['price'].min():,.0f} - ${baseline_df['price'].max():,.0f}")
    print(f"Date range: {baseline_df['listing_date'].min()} to {baseline_df['listing_date'].max()}")
