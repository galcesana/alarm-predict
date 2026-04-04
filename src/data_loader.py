"""
Historical data loader — downloads and processes Israel alert history.

Data source: dleshem/israel-alerts-data on GitHub
Contains a CSV with all Pikud HaOref alerts since 2014+.

We process this data to:
1. Group temporally-close alerts into "attack events"
2. For each event, determine which cities were alarmed
3. Label whether Tel Aviv zones were alarmed or not
4. Build a training dataset for the prediction model
"""

import io
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests

from src.tel_aviv_zones import (
    TEL_AVIV_ZONE_NAMES,
    GUSH_DAN_CITIES,
    is_tel_aviv_zone,
    warning_includes_tel_aviv_region,
    count_gush_dan_cities,
)

logger = logging.getLogger(__name__)

# URL for the raw CSV from GitHub
ALERTS_CSV_URL = (
    "https://raw.githubusercontent.com/dleshem/israel-alerts-data/"
    "main/israel-alerts.csv"
)

DATA_DIR = Path(__file__).parent.parent / "data"


def download_historical_csv(force: bool = False) -> Path:
    """
    Download the historical alerts CSV from GitHub.
    Caches locally in data/ directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "israel-alerts.csv"

    if csv_path.exists() and not force:
        # Check age — re-download if older than 24 hours
        age = datetime.now().timestamp() - csv_path.stat().st_mtime
        if age < 86400:
            logger.info(f"Using cached data: {csv_path}")
            return csv_path

    logger.info("Downloading historical alerts data from GitHub...")
    try:
        response = requests.get(ALERTS_CSV_URL, timeout=60)
        response.raise_for_status()
        csv_path.write_bytes(response.content)
        logger.info(f"Downloaded {len(response.content)} bytes → {csv_path}")
    except requests.RequestException as e:
        if csv_path.exists():
            logger.warning(f"Download failed ({e}), using cached version")
        else:
            raise RuntimeError(
                f"Cannot download historical data and no cache exists: {e}"
            )

    return csv_path


def load_raw_alerts(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the raw alerts CSV into a DataFrame.

    The CSV typically has columns like:
    - timestamp / date
    - alerts (comma-separated city names)
    - category / type

    The exact format may vary — we handle common variants.
    """
    if csv_path is None:
        csv_path = download_historical_csv()

    logger.info(f"Loading alerts from {csv_path}...")

    # Try reading with different encodings
    for encoding in ["utf-8-sig", "utf-8", "cp1255"]:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise RuntimeError(f"Cannot read CSV with any supported encoding: {csv_path}")

    logger.info(f"Loaded {len(df)} raw alert rows")
    logger.info(f"Columns: {list(df.columns)}")

    return df


def parse_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the raw alerts DataFrame into a standardized format.
    Handles different column naming conventions.
    """
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find the timestamp column
    time_col = None
    for candidate in ["timestamp", "date", "time", "datetime", "alertdate"]:
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        # Use first column as timestamp
        time_col = df.columns[0]
        logger.warning(f"No known timestamp column, using '{time_col}'")

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Find the cities/alerts column
    cities_col = None
    for candidate in ["data", "alerts", "cities", "locations", "areas"]:
        if candidate in df.columns:
            cities_col = candidate
            break

    if cities_col is None:
        # Use second column
        remaining = [c for c in df.columns if c != time_col and c != "timestamp"]
        if remaining:
            cities_col = remaining[0]
        else:
            raise RuntimeError("Cannot identify cities column in CSV")

    df["cities_raw"] = df[cities_col].astype(str)

    # Find category column if it exists
    cat_col = None
    for candidate in ["cat", "category", "type"]:
        if candidate in df.columns:
            cat_col = candidate
            break

    if cat_col:
        df["category"] = pd.to_numeric(df[cat_col], errors="coerce").fillna(1).astype(int)
    else:
        df["category"] = 1  # Assume missiles if no category

    # Find title column if it exists
    title_col = None
    for candidate in ["title", "alerttype"]:
        if candidate in df.columns:
            title_col = candidate
            break

    if title_col:
        df["title"] = df[title_col].astype(str)
    else:
        df["title"] = ""

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "cities_raw", "category", "title"]]


def explode_cities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row may contain multiple cities (comma-separated).
    Explode into one row per city per alert.
    """
    df = df.copy()
    df["cities_list"] = df["cities_raw"].apply(
        lambda x: [c.strip() for c in str(x).split(",") if c.strip()]
    )
    df = df.explode("cities_list")
    df = df.rename(columns={"cities_list": "city"})
    df = df.dropna(subset=["city"])
    df = df[df["city"] != "nan"]
    return df.reset_index(drop=True)


def group_into_events(
    df: pd.DataFrame,
    time_gap_seconds: int = 120,
) -> pd.DataFrame:
    """
    Group temporally-close alerts into "attack events".

    Alerts within `time_gap_seconds` of each other are considered
    part of the same attack event. This reconstructs the warning-to-alarm
    sequence from raw alarm data.

    Returns a DataFrame with one row per event containing:
    - event_id: unique event identifier
    - start_time, end_time: event time bounds
    - all_cities: list of all cities alarmed in this event
    - city_count: number of cities
    - categories: set of alert categories in this event
    - involves_tel_aviv: whether any TLV zone was alarmed
    - tel_aviv_zones_alarmed: which TLV zones were alarmed
    - gush_dan_cities_count: how many Gush Dan cities were alarmed
    - involves_gush_dan: whether any Gush Dan city was alarmed
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure sorted by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Assign event IDs based on time gaps
    time_diff = df["timestamp"].diff()
    gap = timedelta(seconds=time_gap_seconds)
    event_boundaries = time_diff > gap
    df["event_id"] = event_boundaries.cumsum()

    # Aggregate per event
    events = []
    for event_id, group in df.groupby("event_id"):
        all_cities = list(set(group["city"].tolist()))
        categories = list(set(group["category"].tolist()))
        titles = list(set(group["title"].tolist()))

        tlv_zones_alarmed = [c for c in all_cities if is_tel_aviv_zone(c)]
        gush_dan_count = count_gush_dan_cities(all_cities)

        events.append({
            "event_id": int(event_id),
            "start_time": group["timestamp"].min(),
            "end_time": group["timestamp"].max(),
            "duration_seconds": (
                group["timestamp"].max() - group["timestamp"].min()
            ).total_seconds(),
            "all_cities": all_cities,
            "city_count": len(all_cities),
            "categories": categories,
            "primary_category": categories[0] if categories else 1,
            "titles": titles,
            "involves_tel_aviv": len(tlv_zones_alarmed) > 0,
            "tel_aviv_zones_alarmed": tlv_zones_alarmed,
            "tlv_zones_alarmed_count": len(tlv_zones_alarmed),
            "gush_dan_cities_count": gush_dan_count,
            "involves_gush_dan": gush_dan_count > 0,
        })

    events_df = pd.DataFrame(events)
    logger.info(
        f"Grouped {len(df)} alerts into {len(events_df)} events. "
        f"{events_df['involves_tel_aviv'].sum()} involved Tel Aviv, "
        f"{events_df['involves_gush_dan'].sum()} involved Gush Dan."
    )

    return events_df


def build_training_data(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the training dataset for the Tel Aviv alarm prediction model.

    For each event that involved the Gush Dan region, create a training sample.
    The target variable is whether each Tel Aviv zone was alarmed.

    We focus on events that had Gush Dan involvement because those are the
    scenarios where "will TLV be alarmed?" is a meaningful question.
    """
    # Filter to events that involved Gush Dan (potential TLV warnings)
    gush_dan_events = events_df[events_df["involves_gush_dan"]].copy()

    if gush_dan_events.empty:
        logger.warning("No events involving Gush Dan found in data")
        return pd.DataFrame()

    logger.info(
        f"Building training data from {len(gush_dan_events)} "
        f"Gush Dan events"
    )

    # For each TLV zone, create a binary label per event
    for zone_name in TEL_AVIV_ZONE_NAMES:
        safe_name = zone_name.replace(" ", "_").replace("-", "_")
        gush_dan_events[f"alarmed_{safe_name}"] = gush_dan_events[
            "tel_aviv_zones_alarmed"
        ].apply(lambda zones: zone_name in zones)

    # Create an "any TLV alarmed" column
    gush_dan_events["any_tlv_alarmed"] = gush_dan_events["involves_tel_aviv"]

    return gush_dan_events


def load_and_process(force_download: bool = False) -> dict:
    """
    Full pipeline: download → parse → explode → group → build training data.

    Returns a dict with:
    - raw_df: raw parsed alerts
    - events_df: grouped events
    - training_df: training data for TLV prediction
    - stats: summary statistics
    """
    csv_path = download_historical_csv(force=force_download)
    raw_df = load_raw_alerts(csv_path)
    parsed_df = parse_alerts(raw_df)
    exploded_df = explode_cities(parsed_df)
    events_df = group_into_events(exploded_df)
    training_df = build_training_data(events_df)

    stats = {
        "total_raw_rows": len(raw_df),
        "total_alerts": len(exploded_df),
        "total_events": len(events_df),
        "gush_dan_events": len(training_df),
        "tlv_events": int(training_df["any_tlv_alarmed"].sum()) if len(training_df) > 0 else 0,
        "date_range": (
            f"{parsed_df['timestamp'].min()} — {parsed_df['timestamp'].max()}"
            if len(parsed_df) > 0 else "N/A"
        ),
    }

    logger.info(f"Data pipeline complete: {stats}")

    return {
        "raw_df": raw_df,
        "parsed_df": parsed_df,
        "events_df": events_df,
        "training_df": training_df,
        "stats": stats,
    }


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = load_and_process()

    print("\n📊 Data Summary:")
    for k, v in result["stats"].items():
        print(f"   {k}: {v}")

    if len(result["training_df"]) > 0:
        df = result["training_df"]
        print(f"\n📍 Tel Aviv alarm rate in Gush Dan events:")
        print(f"   Any TLV zone alarmed: {df['any_tlv_alarmed'].mean():.1%}")
        print(f"   Sample events: {len(df)}")
