"""
Data processing pipeline for property data.
Performs feature engineering and train/val/test splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def process_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42
):
    """
    Process raw property data and create train/val/test splits.
    
    Args:
        input_path: Path to raw CSV file
        output_dir: Directory to save processed datasets
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set (from remaining after test)
        random_seed: Random seed for reproducibility
    """
    # Load raw data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Convert listing_date to datetime
    df['listing_date'] = pd.to_datetime(df['listing_date'])
    
    # Feature engineering
    df['age'] = 2024 - df['year_built']
    df['price_per_sqft'] = df['price'] / df['area_sqft']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    
    # One-hot encode categorical features
    property_type_dummies = pd.get_dummies(df['property_type'], prefix='property_type')
    location_dummies = pd.get_dummies(df['location'], prefix='location')
    
    # Combine features
    feature_cols = [
        'area_sqft', 'bedrooms', 'bathrooms', 'age',
        'has_parking', 'has_gym', 'has_pool',
        'price_per_sqft', 'total_rooms'
    ]
    
    # Create processed dataframe
    processed_df = pd.concat([
        df[feature_cols + ['price', 'listing_date']],
        property_type_dummies,
        location_dummies
    ], axis=1)
    
    # Remove listing_date for model training (keep for reference)
    X = processed_df.drop(['price', 'listing_date'], axis=1)
    y = processed_df['price']
    
    # Split data: first test, then train/val from remaining
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    
    # Split remaining into train and val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_seed
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    print(f"Saving processed data to {output_dir}...")
    
    # Combine X and y for each split
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print("Processing complete!")


if __name__ == '__main__':
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/baseline_properties.csv'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    
    process_data(input_file, output_dir)
