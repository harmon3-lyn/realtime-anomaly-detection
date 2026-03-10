"""
Two-layer anomaly detection: Isolation Forest + seasonality-aware z-score.

Layer 1 — ML (Isolation Forest)
  model.predict() returns -1 for anomalies; parameter sets contamination rate.

Layer 2 — Statistical (z-score + velocity + approval-rate)
  Uses pre-computed seasonal baselines so the live window cannot bias its own mean.

Surface conditions:
  - Both layers agree -> always surfaced
  - Statistical-only -> only if |z| >= HIGH_THRESH or |vel_pct| >= VELOCITY_THRESH with volume >= MIN_VOLUME
  - ML-only -> always surfaced (contamination already controls the rate)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
import joblib
import pandas as pd

from shared import config

logger = logging.getLogger(__name__)

# In-process cache: {group_type: (artifact_dict, blob_last_modified_datetime)}
_MODEL_CACHE: dict[str, tuple[dict, datetime]] = {}


# Data classes

@dataclass
class AnomalyEvent:
    timestamp: datetime
    group_type: str          # "overall", "state", "product_type", "transaction_status", "new_customer"
    group_value: str         # e.g. "TX", "mortgage", "approved", "True", "all"
    anomaly_type: str        # "volume_spike", "volume_drop", "velocity_spike", "velocity_drop", "approval_rate_drop"
    severity: str            # "high", "medium"
    current_value: float
    expected_value: float    # seasonal_mean for this (hour_of_day, day_of_week) bucket
    z_score: float
    ml_score: float          # IF decision_function score (lower = more anomalous)
    detected_by: str         # "both", "statistical", "ml"


# Model loading

def load_artifact(group_type: str, tmp_dir: str = "/tmp") -> Optional[dict]:
    """
    Load the model artifact from /tmp cache or Blob Storage.

    Cache hit: Blob last-modified <= cached timestamp -> reuse in-process copy.
    Cache miss or stale: re-download from Blob, update cache.
    Fallback: if download fails but stale cache exists, return stale copy.
    """
    from shared import blob as blob_store

    blob_name = f"models/{group_type}_latest.pkl" #
    local_path = os.path.join(tmp_dir, f"{group_type}_latest.pkl") #
    blob_mtime = blob_store.get_blob_last_modified(blob_name)

    if group_type in _MODEL_CACHE:
        artifact, cached_mtime = _MODEL_CACHE[group_type]
        if blob_mtime is None or blob_mtime <= cached_mtime:
            return artifact  # Cache hit

    try:
        blob_store.download_model(blob_name, local_path)
        artifact = joblib.load(local_path)
        _MODEL_CACHE[group_type] = (artifact, blob_mtime or datetime.now(timezone.utc))
        logger.info("Model artifact loaded", extra={"group_type": group_type})

        return artifact
    
    except Exception as exc:
        logger.warning("Could not load model artifact", extra={"group_type": group_type, "error": str(exc)})

        if group_type in _MODEL_CACHE:
            logger.info("Using stale cache", extra={"group_type": group_type})
            return _MODEL_CACHE[group_type][0]
        
        return None



# Isolation Forest 

def score_slice(
    agg: pd.DataFrame,
    group_type: str,
    artifact: Optional[dict],
) -> pd.DataFrame:
    """
    Run the Isolation Forest on the scoring agg.

    If the model is unavailable or scoring fails, sets if_flag=1 (not anomalous)
    so the statistical layer can still fire independently.
    """
    if artifact is None:
        agg = agg.copy()
        agg["if_flag"] = 1
        agg["if_score"] = 0.0
        return agg

    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact.get("features", ["volume_z_score", "velocity_pct", "hour_of_day", "day_of_week"])
    avail = [f for f in features if f in agg.columns]

    if not avail:
        agg = agg.copy()
        agg["if_flag"] = 1
        agg["if_score"] = 0.0
        return agg

    try:
        X = scaler.transform(agg[avail].fillna(0).values)

        agg = agg.copy()
        agg["if_flag"] = model.predict(X)
        agg["if_score"] = model.decision_function(X)

    except Exception as exc:
        logger.warning("IF scoring failed; falling back to statistical-only", extra={"group_type": group_type, "error": str(exc)})
        agg = agg.copy()
        agg["if_flag"] = 1
        agg["if_score"] = 0.0

    return agg



# Additional metrics for classification

def _classify_type(row: pd.Series, zscore_thresh: float, vel_thresh: float) -> str:
    vel = float(row.get("velocity_pct") or 0)
    ar_z = float(row.get("approval_rate_z_score") or 0)
    z = float(row.get("volume_z_score") or 0)

    if abs(vel) > vel_thresh:
        return "velocity_spike" if vel > 0 else "velocity_drop"
    
    if ar_z < -zscore_thresh:
        return "approval_rate_drop"
    
    return "volume_spike" if z > 0 else "volume_drop"




# Main function 

def detect_anomalies(
    agg: pd.DataFrame,
    group_type: str,
    group_value_col: Optional[str],
) -> list[AnomalyEvent]:
    """
    Score all hourly rows and return a list of AnomalyEvent objects.

    Parameters:
    agg: Scored hourly aggregation (output of score_slice).
    group_type: Slice name.
    group_value_col: Column holding the group dimension value, or None for "overall".
    """

    if agg.empty:
        return []

    zscore_thresh = config.get_zscore_threshold()
    high_thresh = config.get_high_threshold()
    vel_thresh = config.get_velocity_threshold()
    min_vol = config.get_min_volume()


    events: list[AnomalyEvent] = []

    for _, row in agg.iterrows():
        z = float(row.get("volume_z_score") or 0)
        ar_z = float(row.get("approval_rate_z_score") or 0)
        vel_pct = float(row.get("velocity_pct") or 0)
        vol = float(row.get("volume") or 0)

        stat_flag = (
            (abs(z) > zscore_thresh)
            or (ar_z < -zscore_thresh)
            or (abs(vel_pct) >= vel_thresh and vol >= min_vol)
        )
        ml_flag = row.get("if_flag", 1) == -1

        if not (stat_flag or ml_flag):
            continue

        detected_by = (
            "both"        if (stat_flag and ml_flag) else
            "statistical" if stat_flag else
            "ml"
        )

        # Single-layer statistical: require a strong signal to reduce noise
        if detected_by == "statistical":
            strong_z = abs(z) >= high_thresh
            strong_ar = ar_z <= -high_thresh
            strong_vel = abs(vel_pct) >= vel_thresh and vol >= min_vol

            if not (strong_z or strong_ar or strong_vel):
                continue

        # Skip low vol noise
        if detected_by == "statistical" and vol < min_vol:
            continue

        group_value = str(row[group_value_col]) if group_value_col else "all"


        events.append(AnomalyEvent(
            timestamp=row["hour_bucket"],
            group_type=group_type,
            group_value=group_value,
            anomaly_type=_classify_type(row, zscore_thresh, vel_thresh),
            severity="high" if abs(z) >= 4.0 else "medium",
            current_value=round(vol, 1),
            expected_value=round(float(row.get("volume_seasonal_mean") or 0), 1),
            z_score=round(z, 2),
            ml_score=round(float(row.get("if_score") or 0), 4),
            detected_by=detected_by,
        ))

    return events
