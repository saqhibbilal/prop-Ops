"""
SQLite store for pre-generated market signals (demand, competition, seasonality).
Used by pricing logic to query current market conditions.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Tuple

# Project root: parent of src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "market_signals.db"

# Locations aligned with generate_data.py
LOCATIONS = ("Downtown", "Suburbs", "Urban", "Rural")

# Schema: one row per (date, location) with all signal types
TABLE_NAME = "market_signals"
SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    signal_date DATE NOT NULL,
    location TEXT NOT NULL,
    demand_level TEXT NOT NULL,
    demand_multiplier REAL NOT NULL,
    competition_listings INTEGER NOT NULL,
    seasonality_factor REAL NOT NULL,
    day_of_week INTEGER NOT NULL,
    is_weekend INTEGER NOT NULL,
    month INTEGER NOT NULL,
    year INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (signal_date, location)
);
CREATE INDEX IF NOT EXISTS idx_market_signals_date ON {TABLE_NAME}(signal_date);
CREATE INDEX IF NOT EXISTS idx_market_signals_location ON {TABLE_NAME}(location);
CREATE INDEX IF NOT EXISTS idx_market_signals_date_location ON {TABLE_NAME}(signal_date, location);
"""


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the market signals database."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def create_tables(db_path: Optional[Path] = None) -> None:
    """Create market_signals table and indexes if they do not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def insert_signals(df: pd.DataFrame, db_path: Optional[Path] = None) -> int:
    """
    Insert or replace market signals. DataFrame must have columns:
    signal_date, location, demand_level, demand_multiplier, competition_listings,
    seasonality_factor, day_of_week, is_weekend, month, year.
    """
    create_tables(db_path)
    conn = get_connection(db_path)
    try:
        df = df.copy()
        if "created_at" not in df.columns:
            df["created_at"] = datetime.utcnow().isoformat()
        df["signal_date"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
        cols = [
            "signal_date", "location", "demand_level", "demand_multiplier",
            "competition_listings", "seasonality_factor", "day_of_week",
            "is_weekend", "month", "year", "created_at",
        ]
        df = df[cols]
        count = 0
        cur = conn.cursor()
        for _, row in df.iterrows():
            cur.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_NAME}
                (signal_date, location, demand_level, demand_multiplier,
                 competition_listings, seasonality_factor, day_of_week,
                 is_weekend, month, year, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["signal_date"], row["location"], row["demand_level"],
                    float(row["demand_multiplier"]), int(row["competition_listings"]),
                    float(row["seasonality_factor"]), int(row["day_of_week"]),
                    int(row["is_weekend"]), int(row["month"]), int(row["year"]),
                    row["created_at"],
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def get_current_conditions(
    as_of_date: Optional[date] = None,
    location: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Get market conditions for a given date (default: latest date in DB).
    If location is None, returns all locations.
    """
    conn = get_connection(db_path)
    try:
        if as_of_date is None:
            cur = conn.execute(
                f"SELECT signal_date FROM {TABLE_NAME} ORDER BY signal_date DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row is None:
                return pd.DataFrame()
            as_of_str = row[0]
        else:
            # Convert to date object if needed, then to string
            if isinstance(as_of_date, date):
                as_of_str = as_of_date.strftime("%Y-%m-%d")
            elif hasattr(as_of_date, "strftime"):
                as_of_str = as_of_date.strftime("%Y-%m-%d")
            else:
                as_of_str = str(as_of_date)

        query = f"SELECT * FROM {TABLE_NAME} WHERE signal_date = ?"
        params: List = [as_of_str]
        if location is not None:
            query += " AND location = ?"
            params.append(location)
        query += " ORDER BY location"
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def get_signals_for_date_range(
    start_date: date,
    end_date: date,
    location: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Get all signals between start_date and end_date, optionally for one location."""
    conn = get_connection(db_path)
    try:
        start_str = start_date.strftime("%Y-%m-%d") if hasattr(start_date, "strftime") else str(start_date)
        end_str = end_date.strftime("%Y-%m-%d") if hasattr(end_date, "strftime") else str(end_date)
        query = f"SELECT * FROM {TABLE_NAME} WHERE signal_date >= ? AND signal_date <= ?"
        params: List = [start_str, end_str]
        if location is not None:
            query += " AND location = ?"
            params.append(location)
        query += " ORDER BY signal_date, location"
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def get_seasonality_by_month(db_path: Optional[Path] = None) -> pd.DataFrame:
    """Return average seasonality_factor by month (1-12) across all stored data."""
    conn = get_connection(db_path)
    try:
        query = f"""
        SELECT month, AVG(seasonality_factor) AS seasonality_factor
        FROM {TABLE_NAME}
        GROUP BY month
        ORDER BY month
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def get_latest_date(db_path: Optional[Path] = None) -> Optional[date]:
    """Return the latest signal_date in the database, or None if empty."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute(f"SELECT MAX(signal_date) FROM {TABLE_NAME}")
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        return datetime.strptime(row[0], "%Y-%m-%d").date()
    finally:
        conn.close()


def enrich_with_market_signals(
    df: pd.DataFrame,
    date_col: str = "listing_date",
    location_col: str = "location",
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Left-join market signals onto a DataFrame that has date and location columns.
    Adds: demand_multiplier, competition_listings, seasonality_factor, demand_level.
    """
    if df.empty or date_col not in df.columns or location_col not in df.columns:
        return df
    create_tables(db_path)
    out = df.copy()
    out["_date_str"] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m-%d")
    conn = get_connection(db_path)
    try:
        signals = pd.read_sql_query(
            f"SELECT signal_date, location, demand_multiplier, competition_listings, "
            f"seasonality_factor, demand_level FROM {TABLE_NAME}",
            conn,
        )
    finally:
        conn.close()
    if signals.empty:
        out = out.drop(columns=["_date_str"], errors="ignore")
        return out
    merged = out.merge(
        signals,
        left_on=("_date_str", location_col),
        right_on=("signal_date", "location"),
        how="left",
    )
    drop_cols = ["_date_str", "signal_date", "location"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns], errors="ignore")
    return merged
