"""
Retrain pipeline - trains one IsolationForest per dimension slice, evaluates
on a chronological holdout, and uploads versioned artifacts to Azure Blob Storage.

Structure:
  {
    "model":              IsolationForest,
    "scaler":             StandardScaler,
    "features":           list[str],
    "group_type":         str,
    "group_cols":         list[str] | None,
    "baselines": {
      "volume": {
        "primary":  DataFrame - (group_cols, hour_of_day, day_of_week) -> mean/std,
        "fallback": DataFrame - (hour_of_day, day_of_week) -> mean/std (cross-group),
      },
      "approval_rate": { "primary": ..., "fallback": ... }, (for applicable slices)
    },
    "trained_at":         ISO-8601 UTC string,
    "n_train_rows":       int,
    "holdout_metrics":    {"n_holdout": int, "flag_rate": float, "contamination_target": float},
  }
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from shared import blob as blob_store, config, logging_utils
from shared.features import (
    APPROVAL_IF_FEATURE,
    APPROVAL_RATE_SLICES,
    BASE_IF_FEATURES,
    SLICES,
    extract_baseline_stats,
)
from retrain_pipeline.features import build_training_features, fetch_training_data

logger = logging.getLogger(__name__)


# Helpers

def _get_features(group_type: str, agg: pd.DataFrame) -> list[str]:
    """Return the IF feature list for this slice (adds approval_rate_z_score if available)."""
    features = BASE_IF_FEATURES.copy()
    if group_type in APPROVAL_RATE_SLICES and APPROVAL_IF_FEATURE in agg.columns:
        features.append(APPROVAL_IF_FEATURE)

    return features


# Training

def train_one_slice(
    df: pd.DataFrame,
    group_type: str,
    group_cols: list[str] | None,
    contamination: float,
    holdout_frac: float = 0.10,
    seed: int = 42,
) -> dict:
    """
    Train and evaluate a single slice model.

    Parameters:
    df: Raw transaction DataFrame (90-day training window).
    group_type: Slice name.
    group_cols: Group-by column(s) for this slice.
    contamination: IsolationForest contamination parameter.
    holdout_frac: Fraction of rows held out for evaluation.
    seed: Random seed.

    Returns:
    Artifact dict ready for joblib serialization.
    """
    agg = build_training_features(df, group_type, group_cols)
    if len(agg) < 10:
        raise ValueError(
            f"Insufficient training rows for '{group_type}': got {len(agg)}, need ≥ 10."
        )

    # Chronological holdout
    cutoff = int(len(agg) * (1 - holdout_frac))
    train_agg = agg.iloc[:cutoff]
    holdout_agg = agg.iloc[cutoff:]

    features = _get_features(group_type, agg)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_agg[features].fillna(0).values)

    model = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=seed,
    )

    model.fit(X_train)

    # Holdout evaluation
    X_holdout = scaler.transform(holdout_agg[features].fillna(0).values)
    holdout_flags = model.predict(X_holdout)
    holdout_flag_rate = float((holdout_flags == -1).mean())

    # Extract baselines from the full training agg (more data = more stable stats)
    vol_primary, vol_fallback = extract_baseline_stats(agg, "volume", group_cols)
    baselines: dict[str, dict] = { "volume": {"primary": vol_primary, "fallback": vol_fallback} }

    if group_type in APPROVAL_RATE_SLICES:
        ar_primary, ar_fallback = extract_baseline_stats(agg, "approval_rate", group_cols)
        baselines["approval_rate"] = {"primary": ar_primary, "fallback": ar_fallback}


    return {
        "model": model,
        "scaler": scaler,
        "features": features,
        "group_type": group_type,
        "group_cols": group_cols,
        "baselines": baselines,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_rows": len(agg),
        "holdout_metrics": {
            "n_holdout": len(holdout_agg),
            "flag_rate": round(holdout_flag_rate, 4),
            "contamination_target": contamination,
        }
    }


# Entry Point

def run_retrain() -> None:
    """Entry point called by the retrain_pipeline Azure Function (timer: Sunday 2 AM)."""
    run_ctx = logging_utils.RunContext()
    log = run_ctx.bind(logger, function="retrain_pipeline")
    log.info("Retrain pipeline started")

    contamination = config.get_contamination()
    lookback_days = config.get_lookback_days()

    try:
        df = fetch_training_data(lookback_days)
    except Exception as exc:
        log.error( "Training DB unreachable - aborting retrain", extra={"error": str(exc)} )
        raise

    if df.empty:
        log.warning("No training data returned; skipping retrain")
        return

    tmp_dir = "/tmp/models"
    os.makedirs(tmp_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for group_type, cfg in SLICES.items():
        group_cols = cfg["group_cols"]
        log.info("Training slice", extra={"group_type": group_type})

        try:
            artifact = train_one_slice(df, group_type, group_cols, contamination)
        except Exception as exc:
            log.error( "Training failed for slice", extra={"group_type": group_type, "error": str(exc)} )
            continue

        metrics = artifact["holdout_metrics"]
        log.info(
            "Slice trained",
            extra={
                "group_type": group_type,
                "n_train_rows": artifact["n_train_rows"],
                "holdout_flag_rate": metrics["flag_rate"],
                "contamination_target": metrics["contamination_target"],
            }
        )

        # Serialize -> upload latest + versioned archive
        local_path = os.path.join(tmp_dir, f"{group_type}_latest.pkl")
        joblib.dump(artifact, local_path)

        for blob_name in (
            f"models/{group_type}_latest.pkl",
            f"models/{group_type}_{timestamp}.pkl",
        ):
            try:
                blob_store.upload_model(local_path, blob_name)
                log.info("Artifact uploaded", extra={"blob_name": blob_name})

            except Exception as exc:
                log.error( "Blob upload failed", extra={"blob_name": blob_name, "error": str(exc)} )


    log.info("Retrain pipeline completed")
