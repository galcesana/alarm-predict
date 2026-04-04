"""
Oref Client — polls the Pikud HaOref alert API in real-time.

This module now wraps the official community `pikudhaoref.py` library
to automatically handle Akamai WAF bypasses, cookies, and HTTP errors 
that frequently block raw requests.
"""

import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable

import pikudhaoref

logger = logging.getLogger(__name__)


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
    Polls the Pikud HaOref alert API using the pikudhaoref library.
    """

    def __init__(self, poll_interval: float = 2.0):
        self.poll_interval = poll_interval
        # Initialize the underlying pikudhaoref client
        self.client = pikudhaoref.SyncClient(update_interval=poll_interval)
        
        # Callback fired when a new alert is detected
        self.on_alert: Optional[Callable[[AlertEvent], None]] = None

        # Track the last alert ID to avoid duplicate processing
        self._last_alert_id: Optional[str] = None
        self._running = False
        
        # Register the callback in pikudhaoref
        @self.client.event
        def on_siren(sirens):
            if not sirens:
                return
            
            # Group the sirens into an AlertEvent
            # The library gives us a list of Siren objects, we group them into a single event
            cities = [siren.city.name for siren in sirens]
            first_siren = sirens[0]
            
            # Approximate the category and title based on the library's data
            category_mapping = {
                "Missiles": 1,
                "UAV": 2,
                "Earthquake": 3,
                "Tsunami": 4,
                "Radiological Event": 5,
                "Terrorist Infiltration": 6,
                "General": 7
            }
            cat_name = getattr(first_siren, "category", "missiles").lower()
            cat_id = 1
            for k, v in category_mapping.items():
                if k.lower() in cat_name:
                    cat_id = v
                    break
                    
            event = AlertEvent(
                alert_id=str(int(time.time())),  # Generate a unique ID
                category=cat_id,
                category_name=cat_name,
                title="התרעת פיקוד העורף", 
                description=cat_name,
                cities=cities,
                timestamp=datetime.now(),
            )
            
            logger.info(
                f"🚨 NEW ALERT: {event.title} | "
                f"{event.city_count} cities | "
                f"cat={event.category_name}"
            )
            
            if self.on_alert:
                self.on_alert(event)

    def fetch_alerts(self) -> Optional[AlertEvent]:
        """
        Fetch current alerts. 
        """
        # The library caches the current sirens, we just return the active ones
        sirens = self.client.current_sirens
        if not sirens:
            return None
            
        cities = [siren.city.name for siren in sirens]
        cat_name = getattr(sirens[0], "category", "missiles").lower()
        
        return AlertEvent(
            alert_id=str(int(time.time())),
            category=1,
            category_name=cat_name,
            title="התרעת פיקוד העורף",
            description=cat_name,
            cities=cities,
            timestamp=datetime.now(),
        )

    def poll_once(self) -> Optional[AlertEvent]:
        """Support standard polling API for backwards compat."""
        # The pikudhaoref client automatically polls in a background thread
        # when we instantiate it/add events, but here we manually fetch
        return self.fetch_alerts()

    def start(self):
        """Start the polling loop (blocking)."""
        self._running = True
        logger.info(
            f"Starting oref alert monitor (via pikudhaoref) — "
            f"polling every {self.poll_interval}s"
        )
        print(f"🔍 Monitoring oref.org.il for alerts via proxy (every {self.poll_interval}s)...")
        print("   Press Ctrl+C to stop.\n")

        try:
            # The client's background thread is already running, so we just block the main thread
            while self._running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n⏹ Monitoring stopped.")
            self._running = False

    def stop(self):
        """Stop the polling loop."""
        self._running = False
        try:
            self.client.closed = True
        except:
            pass


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = OrefClient(poll_interval=3.0)

    def on_alert(event: AlertEvent):
        print(f"\n🚨 ALERT: {event.title}")
        print(f"   Category: {event.category_name}")
        print(f"   Cities ({event.city_count}): {', '.join(event.cities[:10])}...")
        print()

    client.on_alert = on_alert

    # Single poll test
    print("Testing single poll...")
    result = client.fetch_alerts()
    if result:
        print(f"Active alert: {result.title} — {result.city_count} cities")
    else:
        print("No active alert (this is normal in peacetime)")

    print("\nStarting continuous monitoring (press ctrl+c to exit)...")
    client.start()
