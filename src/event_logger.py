"""
Event logger — records all observed events for future model updates.

Stores events in an append-only JSONL file (one JSON object per line).
Each entry records:
- The warning (timestamp, cities, category)
- Our prediction (per-zone probabilities)
- The actual outcome (which zones were actually alarmed) — added later
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
EVENTS_LOG = DATA_DIR / "events.jsonl"


def log_prediction(
    event_id: str,
    timestamp: datetime,
    cities: list[str],
    category: str,
    predictions: dict[str, float],
    features: dict,
):
    """
    Log a prediction event to the JSONL file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "type": "prediction",
        "event_id": event_id,
        "timestamp": timestamp.isoformat(),
        "logged_at": datetime.now().isoformat(),
        "cities": cities,
        "city_count": len(cities),
        "category": category,
        "predictions": predictions,
        "features": features,
    }

    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.debug(f"Logged prediction for event {event_id}")


def log_outcome(
    event_id: str,
    zones_alarmed: list[str],
):
    """
    Log the actual outcome — which TLV zones were actually alarmed.
    Called after the event is resolved.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "type": "outcome",
        "event_id": event_id,
        "timestamp": datetime.now().isoformat(),
        "zones_alarmed": zones_alarmed,
    }

    with open(EVENTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.debug(f"Logged outcome for event {event_id}")


def load_event_log() -> list[dict]:
    """Load all logged events."""
    if not EVENTS_LOG.exists():
        return []

    events = []
    with open(EVENTS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed log line: {line[:100]}")

    return events


def get_prediction_count() -> int:
    """Get the total number of predictions logged."""
    events = load_event_log()
    return sum(1 for e in events if e.get("type") == "prediction")
