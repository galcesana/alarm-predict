"""
Prediction microservice — FastAPI HTTP server for the alarm-predict model.

Exposes a POST /predict endpoint that accepts a list of warned cities
and returns Tel Aviv zone alarm probabilities.

Usage:
    uvicorn src.server:app --host 0.0.0.0 --port 8000
"""

import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import AlarmPredictor
from src.features import extract_features_from_live_alert
from src.tel_aviv_zones import (
    TEL_AVIV_ZONES,
    GUSH_DAN_CITIES,
    warning_includes_tel_aviv_region,
    count_gush_dan_cities,
    count_tel_aviv_zones_in_warning,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Alarm Predict API", version="1.0.0")

# Global model instance — loaded once at startup
_model: Optional[AlarmPredictor] = None


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""
    cities: list[str]
    category: int = 1  # 1=missiles, 2=UAV
    timestamp: Optional[str] = None  # ISO format, defaults to now


class ZonePrediction(BaseModel):
    """Prediction for a single Tel Aviv zone."""
    zone_he: str
    zone_en: str
    probability: float
    severity: str  # "critical", "high", "medium", "low"


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""
    predictions: dict[str, float]  # zone_he -> probability
    zones: list[ZonePrediction]
    features: dict
    is_relevant: bool  # whether the warning involves Gush Dan
    gush_dan_count: int
    city_count: int
    model_type: str


def _get_severity(prob: float) -> str:
    """Map probability to severity level."""
    if prob > 0.7:
        return "critical"
    elif prob > 0.5:
        return "high"
    elif prob > 0.3:
        return "medium"
    return "low"


def _load_model() -> AlarmPredictor:
    """Load model from disk."""
    global _model
    if _model is None:
        from pathlib import Path
        model_path = Path(__file__).parent.parent / "models" / "alarm_model.pkl"
        if model_path.exists():
            _model = AlarmPredictor.load(model_path)
            logger.info(f"Model loaded: trained={_model.is_trained}, stats={_model.training_stats}")
        else:
            logger.warning("No trained model found, creating untrained model")
            _model = AlarmPredictor()
    return _model


@app.on_event("startup")
async def startup():
    """Load model on server startup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    model = _load_model()
    if model.is_trained:
        stats = model.training_stats
        logger.info(f"Model ready: {stats.get('model_type', '?')}, "
                     f"trained on {stats.get('n_samples', '?')} events")
    else:
        logger.warning("Model is NOT trained — predictions will use priors only")


@app.get("/health")
async def health():
    """Health check endpoint."""
    model = _load_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_trained": model.is_trained if model else False,
        "model_stats": model.training_stats if model else {},
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run ML prediction for a set of warned cities.

    Returns per-zone alarm probabilities for Tel Aviv's 4 alert zones.
    """
    model = _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Parse timestamp
    if req.timestamp:
        try:
            ts = datetime.fromisoformat(req.timestamp)
        except ValueError:
            ts = datetime.now()
    else:
        ts = datetime.now()

    # Check relevance
    is_relevant = warning_includes_tel_aviv_region(req.cities)
    gd_count = count_gush_dan_cities(req.cities)

    if not is_relevant:
        # Return zero predictions for non-Gush-Dan warnings
        zones = []
        predictions = {}
        for z in TEL_AVIV_ZONES:
            predictions[z.name_he] = 0.0
            zones.append(ZonePrediction(
                zone_he=z.name_he,
                zone_en=z.name_en,
                probability=0.0,
                severity="low",
            ))
        return PredictResponse(
            predictions=predictions,
            zones=zones,
            features={},
            is_relevant=False,
            gush_dan_count=0,
            city_count=len(req.cities),
            model_type=model.training_stats.get("model_type", "unknown"),
        )

    # Extract features
    features = extract_features_from_live_alert(
        cities=req.cities,
        category=req.category,
        timestamp=ts,
    )

    # Run model
    raw_predictions = model.predict(features)

    # Build structured response
    zones = []
    for z in TEL_AVIV_ZONES:
        prob = raw_predictions.get(z.name_he, 0.0)
        zones.append(ZonePrediction(
            zone_he=z.name_he,
            zone_en=z.name_en,
            probability=round(prob, 4),
            severity=_get_severity(prob),
        ))

    # Sort by probability descending
    zones.sort(key=lambda z: -z.probability)

    return PredictResponse(
        predictions={z.zone_he: round(raw_predictions.get(z.zone_he, 0.0), 4)
                     for z in TEL_AVIV_ZONES},
        zones=zones,
        features=features,
        is_relevant=True,
        gush_dan_count=gd_count,
        city_count=len(req.cities),
        model_type=model.training_stats.get("model_type", "unknown"),
    )
