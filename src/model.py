"""
Alarm prediction model — Bayesian-enhanced gradient boosting.

Two-stage approach:
1. Bayesian baseline: P(alarm | warned) per zone from historical rates
2. XGBoost classifier: uses event features for refined prediction
3. Calibrated output: well-calibrated probabilities via Platt scaling

The model predicts P(Tel Aviv alarm | warning event features).
"""

import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    classification_report,
    log_loss,
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.features import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"


@dataclass
class BayesianPrior:
    """
    Beta-Binomial Bayesian prior for per-zone alarm rates.

    Given a city/zone, stores how many times it was:
    - warned (appeared in warning region)
    - alarmed (actually received siren)

    Uses a Beta(alpha, beta) prior for smoothing, so even with
    few observations, we get a reasonable estimate.
    """
    # Counts per zone: {zone_name: {"warned": N, "alarmed": N}}
    zone_counts: dict = field(default_factory=dict)

    # Prior hyperparameters (weakly informative)
    alpha: float = 1.0  # pseudo-successes
    beta: float = 2.0   # pseudo-failures (prior toward ~33% alarm rate)

    def update(self, zone_name: str, was_alarmed: bool):
        """Update counts for a zone."""
        if zone_name not in self.zone_counts:
            self.zone_counts[zone_name] = {"warned": 0, "alarmed": 0}
        self.zone_counts[zone_name]["warned"] += 1
        if was_alarmed:
            self.zone_counts[zone_name]["alarmed"] += 1

    def predict(self, zone_name: str) -> float:
        """
        Return P(alarm | warned) for a zone using Beta-Binomial posterior.

        posterior mean = (alarmed + alpha) / (warned + alpha + beta)
        """
        if zone_name not in self.zone_counts:
            # No data — return prior mean
            return self.alpha / (self.alpha + self.beta)

        counts = self.zone_counts[zone_name]
        return (counts["alarmed"] + self.alpha) / (
            counts["warned"] + self.alpha + self.beta
        )

    def predict_all(self, zone_names: list[str]) -> dict[str, float]:
        """Return prior probabilities for all zones."""
        return {zone: self.predict(zone) for zone in zone_names}

    def to_dict(self) -> dict:
        return {
            "zone_counts": self.zone_counts,
            "alpha": self.alpha,
            "beta": self.beta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BayesianPrior":
        bp = cls()
        bp.zone_counts = d.get("zone_counts", {})
        bp.alpha = d.get("alpha", 1.0)
        bp.beta = d.get("beta", 2.0)
        return bp


class AlarmPredictor:
    """
    Combined Bayesian + XGBoost alarm prediction model.

    Predicts P(Tel Aviv alarm) given warning event features.
    """

    def __init__(self):
        self.bayesian_prior = BayesianPrior()
        self.classifier = None  # XGBoost or logistic regression
        self.is_trained = False
        self.training_stats = {}

    def train(self, X: pd.DataFrame, y: pd.DataFrame, events_df: pd.DataFrame):
        """
        Train the model on historical data.

        Args:
            X: Feature matrix (from features.build_feature_matrix)
            y: Target matrix with 'any_tlv_alarmed' column
            events_df: Original events data (for Bayesian prior)
        """
        logger.info(f"Training on {len(X)} events...")

        # ── Stage 1: Train Bayesian prior ──
        self._train_bayesian_prior(events_df)

        # ── Stage 2: Train XGBoost classifier ──
        target = y["any_tlv_alarmed"].values

        # Check class balance
        n_positive = target.sum()
        n_negative = len(target) - n_positive
        logger.info(f"Class balance: {n_positive} positive, {n_negative} negative")

        if n_positive < 3 or n_negative < 3:
            logger.warning(
                f"Insufficient class balance (pos={n_positive}, neg={n_negative}). "
                "Using Bayesian prior only."
            )
            self.is_trained = True
            self.training_stats = {
                "n_samples": len(X),
                "n_positive": int(n_positive),
                "n_negative": int(n_negative),
                "model_type": "bayesian_only",
            }
            return

        # Scale weight to handle imbalance
        scale_pos_weight = n_negative / max(n_positive, 1)

        if HAS_XGBOOST:
            base_clf = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.info("XGBoost not available, using sklearn GradientBoosting")
            base_clf = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
            )

        # Time-series cross-validation for evaluation
        n_splits = min(3, len(X) // 10)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = []

            for train_idx, test_idx in tscv.split(X):
                X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                y_tr, y_te = target[train_idx], target[test_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue

                base_clf.fit(X_tr, y_tr)
                y_pred = base_clf.predict_proba(X_te)[:, 1]

                try:
                    auc = roc_auc_score(y_te, y_pred)
                    brier = brier_score_loss(y_te, y_pred)
                    cv_scores.append({"auc": auc, "brier": brier})
                except ValueError:
                    pass

            if cv_scores:
                avg_auc = np.mean([s["auc"] for s in cv_scores])
                avg_brier = np.mean([s["brier"] for s in cv_scores])
                logger.info(
                    f"CV scores — AUC: {avg_auc:.3f}, Brier: {avg_brier:.3f}"
                )
            else:
                avg_auc = 0.0
                avg_brier = 1.0
        else:
            avg_auc = 0.0
            avg_brier = 1.0

        # Train final model on all data
        base_clf.fit(X.values, target)

        # Calibrate probabilities
        try:
            self.classifier = CalibratedClassifierCV(
                base_clf, method="sigmoid", cv=min(3, n_splits) if n_splits >= 2 else 2
            )
            self.classifier.fit(X.values, target)
        except Exception as e:
            logger.warning(f"Calibration failed ({e}), using uncalibrated model")
            self.classifier = base_clf

        self.is_trained = True
        self.training_stats = {
            "n_samples": len(X),
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
            "cv_auc": float(avg_auc),
            "cv_brier": float(avg_brier),
            "model_type": "xgboost" if HAS_XGBOOST else "gradient_boosting",
        }

        logger.info(f"Training complete: {self.training_stats}")

    def _train_bayesian_prior(self, events_df: pd.DataFrame):
        """Train Bayesian prior from historical events."""
        from src.tel_aviv_zones import TEL_AVIV_ZONE_NAMES

        for _, event in events_df.iterrows():
            if not event.get("involves_gush_dan", False):
                continue

            alarmed_zones = set(event.get("tel_aviv_zones_alarmed", []))

            for zone in TEL_AVIV_ZONE_NAMES:
                self.bayesian_prior.update(zone, zone in alarmed_zones)

        logger.info("Bayesian prior trained:")
        for zone in TEL_AVIV_ZONE_NAMES:
            prob = self.bayesian_prior.predict(zone)
            logger.info(f"  {zone}: {prob:.1%}")

    def predict(self, features: dict) -> dict[str, float]:
        """
        Predict alarm probabilities for all Tel Aviv zones.

        Args:
            features: dict of feature_name → value (from features module)

        Returns:
            dict of zone_name → probability (0.0 to 1.0)
        """
        from src.tel_aviv_zones import TEL_AVIV_ZONES

        # Get Bayesian priors
        priors = self.bayesian_prior.predict_all(
            [z.name_he for z in TEL_AVIV_ZONES]
        )

        if self.classifier is None:
            # No ML model — return Bayesian priors only
            return priors

        # Get ML prediction (P(any TLV alarmed))
        feature_vec = np.array(
            [features.get(f, 0) for f in FEATURE_NAMES]
        ).reshape(1, -1)

        try:
            ml_prob = self.classifier.predict_proba(feature_vec)[0, 1]
        except Exception as e:
            logger.warning(f"ML prediction failed ({e}), using prior only")
            return priors

        # Combine: scale each zone's prior by the ML prediction
        # This adjusts the zone-level priors up/down based on
        # the event-level ML assessment
        combined = {}
        for zone_name, prior_prob in priors.items():
            # Weighted combination: 40% prior, 60% ML-adjusted
            # ML gives P(any TLV alarmed) — scale zone prior accordingly
            avg_prior = np.mean(list(priors.values()))
            if avg_prior > 0:
                zone_scale = prior_prob / avg_prior
            else:
                zone_scale = 1.0

            # Combined probability
            combined_prob = 0.4 * prior_prob + 0.6 * ml_prob * zone_scale
            combined[zone_name] = float(np.clip(combined_prob, 0.0, 1.0))

        return combined

    def save(self, path: Optional[Path] = None):
        """Save model to disk."""
        if path is None:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            path = MODELS_DIR / "alarm_model.pkl"

        model_data = {
            "bayesian_prior": self.bayesian_prior.to_dict(),
            "classifier": self.classifier,
            "is_trained": self.is_trained,
            "training_stats": self.training_stats,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AlarmPredictor":
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / "alarm_model.pkl"

        if not path.exists():
            logger.warning(f"No model found at {path}, creating untrained model")
            return cls()

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        predictor = cls()
        predictor.bayesian_prior = BayesianPrior.from_dict(
            model_data["bayesian_prior"]
        )
        predictor.classifier = model_data.get("classifier")
        predictor.is_trained = model_data.get("is_trained", False)
        predictor.training_stats = model_data.get("training_stats", {})

        logger.info(
            f"Model loaded from {path} "
            f"(trained={predictor.is_trained}, stats={predictor.training_stats})"
        )
        return predictor
