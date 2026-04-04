"""
Real-time predictor — runs predictions on live oref alerts.

Takes a live AlertEvent from the oref client, extracts features,
runs the model, and formats the output.
"""

import logging
from datetime import datetime
from typing import Optional

from src.model import AlarmPredictor
from src.features import extract_features_from_live_alert
from src.tel_aviv_zones import (
    TEL_AVIV_ZONES,
    TEL_AVIV_ZONE_NAMES,
    warning_includes_tel_aviv_region,
    count_gush_dan_cities,
    count_tel_aviv_zones_in_warning,
)
from src.oref_client import AlertEvent

logger = logging.getLogger(__name__)


def format_prediction(
    predictions: dict[str, float],
    event: AlertEvent,
    features: dict,
) -> str:
    """
    Format prediction results as a pretty console table.

    Returns a formatted string ready to print.
    """
    lines = []
    lines.append("")
    lines.append("🚨 WARNING DETECTED — {} — {}".format(
        event.timestamp.strftime("%H:%M:%S"),
        event.title or "Alert",
    ))
    lines.append("   Warning area: {} cities | Category: {} | Gush Dan: {}".format(
        event.city_count,
        event.category_name,
        count_gush_dan_cities(event.cities),
    ))

    # Show which Gush Dan neighbors are in the warning
    from src.tel_aviv_zones import GUSH_DAN_CITIES
    warned_gd = sorted(set(event.cities) & GUSH_DAN_CITIES)
    if warned_gd:
        lines.append("   Gush Dan cities warned: {}".format(
            ", ".join(warned_gd[:8])
        ))
        if len(warned_gd) > 8:
            lines.append(f"   ... and {len(warned_gd) - 8} more")

    h_line = "─" * 38
    h_short = "─" * 10

    lines.append("")
    lines.append(f"   ┌{h_line}┬{h_short}┐")
    lines.append("   │ {:<36} │ {:<8} │".format("Tel Aviv Zone", "Alarm %"))
    lines.append(f"   ├{h_line}┼{h_short}┤")

    # Sort by probability descending
    sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
    for zone_name, prob in sorted_preds:
        pct = f"{prob:.0%}"
        if prob > 0.7:
            emoji = "🔴"
        elif prob > 0.5:
            emoji = "🟠"
        elif prob > 0.3:
            emoji = "🟡"
        else:
            emoji = "🟢"

        # Truncate zone name for display
        display_name = zone_name[:34]
        lines.append(f"   │ {display_name:<36} │ {pct:>4} {emoji}  │")

    lines.append(f"   └{h_line}┴{h_short}┘")

    # Feature summary
    lines.append("")
    lines.append("   Features: cities={}, gush_dan={}, missiles={}, large_scale={}".format(
        features.get("city_count", "?"),
        features.get("gush_dan_count", "?"),
        "yes" if features.get("is_missiles") else "no",
        "yes" if features.get("is_large_scale") else "no",
    ))
    lines.append("")

    return "\n".join(lines)


def predict_for_event(
    event: AlertEvent,
    model: AlarmPredictor,
) -> Optional[dict]:
    """
    Run prediction for a live alert event.

    Returns prediction dict or None if event is not relevant to TLV.
    """
    # Check if this warning is relevant to Tel Aviv region
    if not warning_includes_tel_aviv_region(event.cities):
        logger.debug(
            f"Warning not relevant to TLV region "
            f"({event.city_count} cities, 0 in Gush Dan)"
        )
        return None

    # Extract features
    features = extract_features_from_live_alert(
        cities=event.cities,
        category=event.category,
        timestamp=event.timestamp,
    )

    # Run model
    predictions = model.predict(features)

    # Format and print
    output = format_prediction(predictions, event, features)
    print(output)
    
    # Send to Telegram if configured
    from src.telegram_utils import send_alert_message
    send_alert_message(output)

    return {
        "predictions": predictions,
        "features": features,
        "event_id": event.alert_id,
        "timestamp": event.timestamp.isoformat(),
        "cities": event.cities,
        "category": event.category_name,
    }
