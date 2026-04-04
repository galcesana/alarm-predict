"""
Feature engineering for alarm prediction.

Extracts features from alert events for both:
1. Historical training — from grouped event data
2. Real-time prediction — from a live oref API alert

Features capture:
- Warning size & geographic spread
- Threat type (missile vs drone)
- Whether this is an Iran-scale attack
- Temporal patterns
- Relationship to Tel Aviv specifically
"""

import math
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from src.tel_aviv_zones import (
    TEL_AVIV_ZONE_NAMES,
    GUSH_DAN_CITIES,
    GUSH_DAN_COORDS,
    TEL_AVIV_CENTER,
    count_gush_dan_cities,
    count_tel_aviv_zones_in_warning,
)

logger = logging.getLogger(__name__)

# Feature names in the order expected by the model
FEATURE_NAMES = [
    "city_count",
    "gush_dan_count",
    "gush_dan_ratio",
    "tlv_zones_in_warning",
    "is_missiles",
    "is_uav",
    "is_large_scale",          # >50 cities = likely Iran attack
    "is_massive_scale",        # >100 cities = definitely Iran attack
    "hour_of_day",
    "is_night",                # 22:00 - 06:00
    "is_weekend",              # Friday/Saturday in Israel
    "duration_seconds",
    "warning_spread_km",       # Diameter of warning area
    "distance_to_tlv_center",  # Warning centroid to TLV
    "neighbor_density",        # Fraction of known Gush Dan neighbors warned
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two lat/lng points."""
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _compute_warning_spread(cities: list[str]) -> float:
    """
    Estimate the geographic spread (diameter) of the warning area in km.
    Uses known coordinates of Gush Dan cities.
    """
    coords = []
    for city in cities:
        if city in GUSH_DAN_COORDS:
            coords.append(GUSH_DAN_COORDS[city])

    if len(coords) < 2:
        return 0.0

    # Maximum pairwise distance
    max_dist = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine_km(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            max_dist = max(max_dist, d)

    return max_dist


def _compute_centroid(cities: list[str]) -> tuple[float, float]:
    """Compute the centroid of warned cities (using known coords)."""
    coords = []
    for city in cities:
        if city in GUSH_DAN_COORDS:
            coords.append(GUSH_DAN_COORDS[city])

    if not coords:
        return TEL_AVIV_CENTER  # Default to TLV if no known coords

    avg_lat = sum(c[0] for c in coords) / len(coords)
    avg_lng = sum(c[1] for c in coords) / len(coords)
    return (avg_lat, avg_lng)


def extract_features_from_event(event_row: dict) -> dict:
    """
    Extract features from a historical event (a row from events_df).

    Args:
        event_row: dict with keys from data_loader.group_into_events()

    Returns:
        dict of feature_name → value
    """
    cities = event_row.get("all_cities", [])
    categories = event_row.get("categories", [1])
    primary_cat = event_row.get("primary_category", 1)
    start_time = event_row.get("start_time", datetime.now())
    duration = event_row.get("duration_seconds", 0)

    if isinstance(start_time, str):
        start_time = pd.to_datetime(start_time)

    city_count = len(cities)
    gush_dan_count = count_gush_dan_cities(cities)
    tlv_zones = count_tel_aviv_zones_in_warning(cities)

    # Geographic features
    warning_spread = _compute_warning_spread(cities)
    centroid = _compute_centroid(cities)
    dist_to_tlv = haversine_km(
        centroid[0], centroid[1],
        TEL_AVIV_CENTER[0], TEL_AVIV_CENTER[1]
    )

    # Neighbor density: what fraction of known Gush Dan cities are warned
    neighbor_density = gush_dan_count / len(GUSH_DAN_CITIES) if GUSH_DAN_CITIES else 0

    # Temporal features
    hour = start_time.hour if hasattr(start_time, "hour") else 12
    is_night = 1 if (hour >= 22 or hour < 6) else 0
    day_of_week = start_time.weekday() if hasattr(start_time, "weekday") else 0
    is_weekend = 1 if day_of_week in [4, 5] else 0  # Fri=4, Sat=5 in Python

    return {
        "city_count": city_count,
        "gush_dan_count": gush_dan_count,
        "gush_dan_ratio": gush_dan_count / max(city_count, 1),
        "tlv_zones_in_warning": tlv_zones,
        "is_missiles": 1 if primary_cat == 1 else 0,
        "is_uav": 1 if primary_cat == 2 else 0,
        "is_large_scale": 1 if city_count > 50 else 0,
        "is_massive_scale": 1 if city_count > 100 else 0,
        "hour_of_day": hour,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "duration_seconds": min(duration, 3600),  # Cap at 1 hour
        "warning_spread_km": warning_spread,
        "distance_to_tlv_center": dist_to_tlv,
        "neighbor_density": neighbor_density,
    }


def extract_features_from_live_alert(
    cities: list[str],
    category: int = 1,
    timestamp: Optional[datetime] = None,
    concurrent_warnings: int = 1,
) -> dict:
    """
    Extract features from a live oref API alert for real-time prediction.

    Args:
        cities: list of city names from the alert data[] field
        category: alert category (1=missiles, 2=UAV, etc.)
        timestamp: when the alert was received
        concurrent_warnings: number of simultaneous active warnings

    Returns:
        dict of feature_name → value
    """
    if timestamp is None:
        timestamp = datetime.now()

    city_count = len(cities)
    gush_dan_count = count_gush_dan_cities(cities)
    tlv_zones = count_tel_aviv_zones_in_warning(cities)

    warning_spread = _compute_warning_spread(cities)
    centroid = _compute_centroid(cities)
    dist_to_tlv = haversine_km(
        centroid[0], centroid[1],
        TEL_AVIV_CENTER[0], TEL_AVIV_CENTER[1]
    )

    neighbor_density = gush_dan_count / len(GUSH_DAN_CITIES) if GUSH_DAN_CITIES else 0

    hour = timestamp.hour
    is_night = 1 if (hour >= 22 or hour < 6) else 0
    is_weekend = 1 if timestamp.weekday() in [4, 5] else 0

    return {
        "city_count": city_count,
        "gush_dan_count": gush_dan_count,
        "gush_dan_ratio": gush_dan_count / max(city_count, 1),
        "tlv_zones_in_warning": tlv_zones,
        "is_missiles": 1 if category == 1 else 0,
        "is_uav": 1 if category == 2 else 0,
        "is_large_scale": 1 if city_count > 50 else 0,
        "is_massive_scale": 1 if city_count > 100 else 0,
        "hour_of_day": hour,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "duration_seconds": 0,  # Unknown for live alerts
        "warning_spread_km": warning_spread,
        "distance_to_tlv_center": dist_to_tlv,
        "neighbor_density": neighbor_density,
    }


def build_feature_matrix(events_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix X and target matrix y from training events.

    Returns:
        X: DataFrame with features (one row per event)
        y: DataFrame with per-zone binary labels
    """
    feature_rows = []
    for _, row in events_df.iterrows():
        features = extract_features_from_event(row.to_dict())
        feature_rows.append(features)

    X = pd.DataFrame(feature_rows)

    # Ensure column order matches FEATURE_NAMES
    for col in FEATURE_NAMES:
        if col not in X.columns:
            X[col] = 0
    X = X[FEATURE_NAMES]

    # Build target: "any TLV zone alarmed"
    y = events_df[["any_tlv_alarmed"]].copy()
    y["any_tlv_alarmed"] = y["any_tlv_alarmed"].astype(int)

    # Also include per-zone targets if available
    zone_cols = [c for c in events_df.columns if c.startswith("alarmed_")]
    for col in zone_cols:
        y[col] = events_df[col].astype(int)

    logger.info(f"Built feature matrix: X={X.shape}, y={y.shape}")

    return X, y
