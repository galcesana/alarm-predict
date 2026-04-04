"""
Alarm Predict — Tel Aviv Missile Alarm Prediction System

Main entry point. Monitors the Pikud HaOref alert API in real-time
and predicts whether Tel Aviv zones will receive an alarm when a
warning is issued.

Usage:
    python main.py              # Start real-time monitoring
    python main.py --train      # Train/retrain the model first
    python main.py --test       # Run a test prediction (simulated)
    python main.py --poll-once  # Single poll of the API
"""

import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.oref_client import OrefClient, AlertEvent
from src.model import AlarmPredictor
from src.predictor import predict_for_event
from src.event_logger import log_prediction
from src.tel_aviv_zones import (
    TEL_AVIV_ZONES,
    GUSH_DAN_CITIES,
    warning_includes_tel_aviv_region,
)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    """Print startup banner."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         🚀 ALARM PREDICT — Tel Aviv Edition         ║")
    print("║    Missile alarm probability prediction system      ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Monitoring: Pikud HaOref (oref.org.il)             ║")
    print("║  Focus: Tel Aviv 4 alert zones                      ║")
    print("║  Model: Bayesian + XGBoost ensemble                 ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def load_model() -> AlarmPredictor:
    """Load the trained model, or train one if it doesn't exist."""
    model_path = Path("models/alarm_model.pkl")

    if model_path.exists():
        print("📦 Loading trained model...")
        model = AlarmPredictor.load(model_path)
        if model.is_trained:
            stats = model.training_stats
            print(f"   Model type: {stats.get('model_type', 'unknown')}")
            print(f"   Trained on: {stats.get('n_samples', '?')} events")
            if "cv_auc" in stats:
                print(f"   CV AUC: {stats['cv_auc']:.3f}")
            return model
        else:
            print("   ⚠ Model found but not trained, training now...")
    else:
        print("📦 No trained model found, training now...")

    from src.train import train_model
    model = train_model()
    return model


def run_test_prediction(model: AlarmPredictor):
    """Run a simulated test prediction."""
    print("\n🧪 TEST MODE — Simulating an Iran-scale missile warning\n")

    # Simulate a large-scale warning covering Gush Dan
    simulated_cities = list(GUSH_DAN_CITIES)[:20]

    simulated_event = AlertEvent(
        alert_id="test-001",
        category=1,
        category_name="missiles",
        title="ירי רקטות וטילים",
        description="היכנסו למרחב המוגן",
        cities=simulated_cities,
        timestamp=datetime.now(),
    )

    result = predict_for_event(simulated_event, model)

    if result is None:
        print("❌ Prediction returned None — check the model")
    else:
        print("✅ Test prediction completed successfully")


def run_live_monitor(model: AlarmPredictor, poll_interval: float = 2.0):
    """Start live monitoring of the oref API."""
    client = OrefClient(poll_interval=poll_interval)

    non_tlv_count = 0

    def on_alert(event: AlertEvent):
        nonlocal non_tlv_count

        # Check if this warning is relevant to TLV
        if not warning_includes_tel_aviv_region(event.cities):
            non_tlv_count += 1
            print(
                f"   ℹ Alert #{event.alert_id}: {event.city_count} cities "
                f"({event.category_name}) — not in TLV region "
                f"[{non_tlv_count} non-TLV alerts so far]"
            )
            return

        # This warning involves the TLV region — predict!
        result = predict_for_event(event, model)

        if result:
            # Log the prediction
            log_prediction(
                event_id=result["event_id"],
                timestamp=event.timestamp,
                cities=result["cities"],
                category=result["category"],
                predictions=result["predictions"],
                features=result["features"],
            )

    client.on_alert = on_alert

    print(f"\n🔍 Starting live monitoring...")
    print(f"   Polling interval: {poll_interval}s")
    print(f"   Watching for: Tel Aviv / Gush Dan region alerts")
    print(f"   Tel Aviv zones monitored:")
    for zone in TEL_AVIV_ZONES:
        print(f"     • {zone.name_he} ({zone.name_en})")
    print(f"\n   Press Ctrl+C to stop.\n")

    client.start()


def main():
    parser = argparse.ArgumentParser(
        description="Alarm Predict — Tel Aviv missile alarm prediction"
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Force retrain the model before starting"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a test prediction with simulated data"
    )
    parser.add_argument(
        "--poll-once", action="store_true",
        help="Poll the API once and exit"
    )
    parser.add_argument(
        "--interval", type=float, default=2.0,
        help="Polling interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    print_banner()

    # Train if requested
    if args.train:
        from src.train import train_model
        model = train_model(force_download=True)
    else:
        model = load_model()

    # Test mode
    if args.test:
        run_test_prediction(model)
        return

    # Single poll
    if args.poll_once:
        print("🔍 Polling oref API once...")
        client = OrefClient()
        event = client.fetch_alerts()
        if event:
            print(f"   Active alert: {event.title} — {event.city_count} cities")
            predict_for_event(event, model)
        else:
            print("   No active alert (this is normal in peacetime)")
        return

    # Default: live monitoring
    from src.telegram_utils import start_bot_polling
    start_bot_polling()
    run_live_monitor(model, poll_interval=args.interval)


if __name__ == "__main__":
    main()
