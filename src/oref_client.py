"""
Oref Client — polls the Pikud HaOref alert API in real-time.

Uses the pikudhaoref library's HTTP session to bypass Akamai WAF,
but polls the raw oref API directly to get Hebrew city names
(the library translates to English which breaks our matching).
"""

import json
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import pikudhaoref

logger = logging.getLogger(__name__)

OREF_ALERTS_URL = "https://www.oref.org.il/WarningMessages/alert/alerts.json"

ALERT_CATEGORIES = {
    1: "missiles",
    2: "uav",
    3: "earthquake",
    4: "tsunami",
    5: "radiological",
    6: "terrorist",
    7: "general",
}


@dataclass
class AlertEvent:
    """Represents a single alert event from the oref API."""
    alert_id: str
    category: int
    category_name: str
    title: str
    description: str
    cities: list[str]
    timestamp: datetime
    raw_data: dict = field(default_factory=dict)

    @property
    def is_missile(self) -> bool:
        return self.category == 1

    @property
    def is_uav(self) -> bool:
        return self.category == 2

    @property
    def city_count(self) -> int:
        return len(self.cities)


class OrefClient:
    """
    Polls the Pikud HaOref alert API using pikudhaoref's WAF-bypassing
    HTTP session, but parses the raw JSON ourselves to keep Hebrew names.
    """

    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval

        # Initialize pikudhaoref just to get its WAF-bypassing HTTP session
        self._piku_client = pikudhaoref.SyncClient(update_interval=poll_interval)
        self._session = self._piku_client.http.session

        # Callback fired when a new alert is detected
        self.on_alert: Optional[Callable[[AlertEvent], None]] = None

        # Track the last alert ID to avoid duplicate processing
        self._last_alert_id: Optional[str] = None
        self._running = False

    def fetch_alerts(self) -> Optional[AlertEvent]:
        """
        Fetch current alerts from the oref API using the WAF-bypassing session.
        Returns an AlertEvent with Hebrew city names, or None if no active alert.
        """
        try:
            response = self._session.get(OREF_ALERTS_URL, timeout=5)
            response.raise_for_status()

            text = response.text.strip()
            if not text or text == "null":
                return None

            # Strip BOM if present
            text = text.lstrip("\ufeff")

            # Akamai HTML challenge — skip silently
            if text.startswith("<html") or text.startswith("<!DOC"):
                logger.debug("Received HTML challenge, skipping")
                return None

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                logger.debug(f"Non-JSON response, skipping")
                return None

            # Handle both single object and array responses
            if isinstance(data, list):
                if not data:
                    return None
                data = data[0]

            if not isinstance(data, dict):
                return None

            # Extract fields — these come in Hebrew from the raw API
            alert_id = str(data.get("id", ""))
            category = data.get("cat", 0)
            title = data.get("title", "")
            desc = data.get("desc", "")
            cities_raw = data.get("data", [])

            if isinstance(cities_raw, str):
                cities = [c.strip() for c in cities_raw.split(",") if c.strip()]
            elif isinstance(cities_raw, list):
                cities = [str(c).strip() for c in cities_raw if c]
            else:
                cities = []

            if not cities:
                return None

            return AlertEvent(
                alert_id=alert_id,
                category=category,
                category_name=ALERT_CATEGORIES.get(category, f"unknown-{category}"),
                title=title,
                description=desc,
                cities=cities,
                timestamp=datetime.now(),
                raw_data=data,
            )

        except Exception as e:
            logger.debug(f"Fetch error: {e}")
            return None

    def poll_once(self) -> Optional[AlertEvent]:
        """
        Poll once and fire callback if new alert detected.
        Returns the alert event if new, None otherwise.
        """
        event = self.fetch_alerts()

        if event is None:
            if self._last_alert_id is not None:
                logger.debug("Alert cleared")
                self._last_alert_id = None
            return None

        if event.alert_id == self._last_alert_id:
            return None  # Same alert, already processed

        # New alert!
        self._last_alert_id = event.alert_id
        logger.info(
            f"🚨 NEW ALERT: {event.title} | "
            f"{event.city_count} cities | "
            f"cat={event.category_name}"
        )

        if self.on_alert:
            self.on_alert(event)

        return event

    def start(self):
        """Start the polling loop (blocking)."""
        self._running = True
        logger.info(
            f"Starting oref alert monitor — "
            f"polling every {self.poll_interval}s"
        )
        print(f"🔍 Monitoring oref.org.il for alerts (every {self.poll_interval}s)...")
        print("   Press Ctrl+C to stop.\n")

        try:
            while self._running:
                self.poll_once()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("\n⏹ Monitoring stopped.")
            self._running = False

    def stop(self):
        """Stop the polling loop."""
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = OrefClient(poll_interval=3.0)

    def on_alert(event: AlertEvent):
        print(f"\n🚨 ALERT: {event.title}")
        print(f"   Category: {event.category_name}")
        print(f"   Cities ({event.city_count}): {', '.join(event.cities[:10])}")
        print()

    client.on_alert = on_alert

    print("Testing single poll...")
    result = client.fetch_alerts()
    if result:
        print(f"Active alert: {result.title} — {result.city_count} cities")
    else:
        print("No active alert (this is normal in peacetime)")
