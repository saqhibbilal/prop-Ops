"""
Market analysis: current conditions, demand multipliers, competition, optional price trend.
Uses pre-generated market signals from the data store.
"""

import sys
from pathlib import Path
from datetime import date
from typing import Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.market_data_store import (
    get_current_conditions,
    get_signals_for_date_range,
    get_seasonality_by_month,
    get_latest_date,
    LOCATIONS,
)


class MarketAnalyzer:
    """
    Analyzes current market conditions from the market signals database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path

    def get_conditions(
        self,
        as_of_date: Optional[date] = None,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get market conditions for a date (default: latest) and optional location.
        Returns dict with demand_multiplier, competition_listings, seasonality_factor,
        demand_level, and optionally location-specific row.
        """
        df = get_current_conditions(as_of_date=as_of_date, location=location, db_path=self.db_path)
        if df.empty:
            return {
                "demand_multiplier": 1.0,
                "competition_listings": 100,
                "seasonality_factor": 1.0,
                "demand_level": "medium",
                "as_of_date": as_of_date,
                "location": location,
            }
        row = df.iloc[0]
        return {
            "demand_multiplier": float(row["demand_multiplier"]),
            "competition_listings": int(row["competition_listings"]),
            "seasonality_factor": float(row["seasonality_factor"]),
            "demand_level": str(row["demand_level"]),
            "as_of_date": str(row["signal_date"]),
            "location": str(row["location"]),
        }

    def get_demand_multiplier(self, location: str, as_of_date: Optional[date] = None) -> float:
        """Demand multiplier for a location (1.0 = baseline)."""
        c = self.get_conditions(as_of_date=as_of_date, location=location)
        return c["demand_multiplier"]

    def get_competition_effect(self, location: str, as_of_date: Optional[date] = None) -> float:
        """
        Competition-based price adjustment. High competition -> slight discount factor (< 1.0).
        Returns multiplier: e.g. 0.97 when many listings, 1.02 when few.
        """
        c = self.get_conditions(as_of_date=as_of_date, location=location)
        comp = c["competition_listings"]
        # Simple rule: normalize around 200 listings; more -> discount, fewer -> premium
        if comp >= 300:
            return 0.96
        if comp >= 220:
            return 0.98
        if comp <= 80:
            return 1.03
        if comp <= 120:
            return 1.01
        return 1.0

    def get_history(
        self,
        start_date: date,
        end_date: date,
        location: Optional[str] = None,
    ):
        """Return DataFrame of signals in range for charts/trends."""
        return get_signals_for_date_range(start_date, end_date, location=location, db_path=self.db_path)

    def get_seasonality(self):
        """Return DataFrame of average seasonality by month (1-12)."""
        return get_seasonality_by_month(db_path=self.db_path)

    def get_latest_signal_date(self) -> Optional[date]:
        """Latest date we have signals for."""
        return get_latest_date(db_path=self.db_path)
