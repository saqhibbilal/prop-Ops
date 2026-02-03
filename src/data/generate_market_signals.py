"""
Generate synthetic market signals for dynamic pricing: demand, competition, seasonality.
Designed to mimic real-world real estate market patterns (listing activity, seasonal demand).
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.market_data_store import (
    create_tables,
    insert_signals,
    LOCATIONS,
    DEFAULT_DB_PATH,
)

# Real-estate style seasonality: spring/summer peak, winter trough (US-style calendar)
MONTH_SEASONALITY = {
    1: 0.90,   # Jan - post-holiday lull
    2: 0.92,   # Feb
    3: 1.00,   # Mar - pickup
    4: 1.06,   # Apr
    5: 1.10,   # May - peak season start
    6: 1.12,   # Jun
    7: 1.08,   # Jul
    8: 1.05,   # Aug
    9: 1.02,   # Sep
    10: 0.98,  # Oct
    11: 0.92,  # Nov - holiday slowdown
    12: 0.88,  # Dec - year-end lull
}

# Base demand by location (urban cores hotter, rural slower)
LOCATION_DEMAND_BASE = {
    "Downtown": 1.15,
    "Urban": 1.08,
    "Suburbs": 1.00,
    "Rural": 0.88,
}

# Base competition (active listings) by location - more inventory in dense areas
LOCATION_COMPETITION_BASE = {
    "Downtown": 180,
    "Urban": 220,
    "Suburbs": 320,
    "Rural": 90,
}


def generate_market_signals(
    start_date: date,
    end_date: date,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate daily market signals per location for the given date range.
    Each row: one (date, location) with demand_level, demand_multiplier,
    competition_listings, seasonality_factor, day_of_week, is_weekend.
    """
    np.random.seed(random_seed)
    rows = []
    d = start_date
    while d <= end_date:
        month = d.month
        year = d.year
        day_of_week = d.isoweekday()  # 1=Mon, 7=Sun
        is_weekend = 1 if day_of_week >= 6 else 0
        seasonality = MONTH_SEASONALITY[month]

        for location in LOCATIONS:
            # Demand: base by location * seasonality * weekend lift
            loc_base = LOCATION_DEMAND_BASE[location]
            weekend_lift = 1.06 if is_weekend else 1.0  # more viewings on weekends
            demand_mult = loc_base * seasonality * weekend_lift
            # Add small daily noise
            demand_mult *= np.clip(1.0 + np.random.normal(0, 0.03), 0.92, 1.08)

            # Demand level label for readability
            if demand_mult >= 1.10:
                demand_level = "high"
            elif demand_mult <= 0.95:
                demand_level = "low"
            else:
                demand_level = "medium"

            # Competition: base by location, seasonal trend (more listings in spring/summer)
            comp_base = LOCATION_COMPETITION_BASE[location]
            comp = int(comp_base * seasonality * (1.0 + np.random.uniform(-0.08, 0.08)))
            comp = max(20, comp)

            rows.append({
                "signal_date": d,
                "location": location,
                "demand_level": demand_level,
                "demand_multiplier": round(float(demand_mult), 4),
                "competition_listings": comp,
                "seasonality_factor": round(seasonality, 4),
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "month": month,
                "year": year,
            })
        d += timedelta(days=1)

    return pd.DataFrame(rows)


def populate_and_store(
    days_back: int = 730,
    end_date: Optional[date] = None,
    db_path: Optional[Path] = None,
    random_seed: int = 42,
) -> int:
    """
    Generate market signals for the last `days_back` days (or up to end_date)
    and store them in the SQLite database. Run once to populate.
    """
    from datetime import date as date_type
    end = end_date or date_type.today()
    start = end - timedelta(days=days_back)
    df = generate_market_signals(start, end, random_seed=random_seed)
    create_tables(db_path)
    count = insert_signals(df, db_path)
    return count


if __name__ == "__main__":
    from datetime import date as date_type
    import argparse

    parser = argparse.ArgumentParser(description="Generate and store market signals")
    parser.add_argument("--days", type=int, default=730, help="Number of days of history (default 730 = 2 years)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Generate but do not write to DB")
    args = parser.parse_args()

    end = date_type.today()
    start = end - timedelta(days=args.days)
    print(f"Generating market signals from {start} to {end} ({args.days} days) for {len(LOCATIONS)} locations...")
    df = generate_market_signals(start, end, random_seed=args.seed)
    print(f"Generated {len(df)} rows.")

    if args.dry_run:
        print("Dry run: not writing to database.")
        print(df.head(10))
    else:
        count = insert_signals(df, DEFAULT_DB_PATH)
        print(f"Stored {count} rows in {DEFAULT_DB_PATH}")
