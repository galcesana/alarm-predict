"""
Training script — downloads data, engineers features, trains model.

Run this to create/update the prediction model:
    python -m src.train
"""

import logging
import sys
from pathlib import Path

from src.data_loader import load_and_process
from src.features import build_feature_matrix
from src.model import AlarmPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def train_model(force_download: bool = False) -> AlarmPredictor:
    """
    Full training pipeline:
    1. Download & process historical data
    2. Build feature matrix
    3. Train model
    4. Save to disk
    """
    print("=" * 60)
    print("  ALARM PREDICT — Model Training")
    print("=" * 60)

    # Step 1: Load data
    print("\n📥 Step 1: Loading historical data...")
    data = load_and_process(force_download=force_download)

    print(f"\n📊 Data Summary:")
    for k, v in data["stats"].items():
        print(f"   {k}: {v}")

    training_df = data["training_df"]
    events_df = data["events_df"]

    if training_df.empty:
        print("\n❌ No training data available. Cannot train model.")
        print("   This might mean the historical CSV format has changed.")
        print("   Check the data manually with: python -m src.data_loader")
        return AlarmPredictor()

    # Step 2: Build features
    print(f"\n🔧 Step 2: Engineering features for {len(training_df)} events...")
    X, y = build_feature_matrix(training_df)

    print(f"   Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"   Target distribution:")
    print(f"     TLV alarmed: {y['any_tlv_alarmed'].sum()} events")
    print(f"     TLV not alarmed: {(1 - y['any_tlv_alarmed']).sum()} events")

    # Step 3: Train
    print("\n🤖 Step 3: Training model...")
    model = AlarmPredictor()
    model.train(X, y, events_df)

    print(f"\n📈 Training Results:")
    for k, v in model.training_stats.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    # Step 4: Save
    print("\n💾 Step 4: Saving model...")
    model.save()

    # Step 5: Test prediction
    print("\n🧪 Step 5: Test prediction (using last event features)...")
    from src.features import extract_features_from_event
    last_event = training_df.iloc[-1].to_dict()
    test_features = extract_features_from_event(last_event)
    predictions = model.predict(test_features)

    print("\n   Test prediction (last historical event):")
    for zone, prob in sorted(predictions.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        emoji = "🔴" if prob > 0.7 else "🟠" if prob > 0.5 else "🟡" if prob > 0.3 else "🟢"
        print(f"   {emoji} {prob:5.1%} {bar} {zone}")

    print("\n✅ Training complete!")
    print(f"   Model saved to: models/alarm_model.pkl")

    return model


if __name__ == "__main__":
    force = "--force" in sys.argv
    train_model(force_download=force)
