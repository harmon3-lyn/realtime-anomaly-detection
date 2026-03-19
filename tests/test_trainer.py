"""
Unit tests for retrain pipeline.
"""
from __future__ import annotations

import pandas as pd
import pytest
from shared.features import APPROVAL_RATE_SLICES, extract_baseline_stats
from retrain_pipeline.features import build_training_features
from retrain_pipeline.trainer import train_one_slice, _get_features


# Feature engineering

def test_build_training_features_overall(sample_transactions):
    """build_training_features must return seasonal baseline columns for 'overall'."""
    agg = build_training_features(sample_transactions, "overall", None)

    assert "volume_z_score" in agg.columns
    assert "volume_seasonal_mean" in agg.columns
    assert "volume_seasonal_std" in agg.columns
    assert "velocity_pct" in agg.columns
    assert "hour_of_day" in agg.columns
    assert "day_of_week" in agg.columns
    assert not agg.empty


def test_build_training_features_state(sample_transactions):
    """State slice agg must have one row per (hour_bucket, state)."""
    agg = build_training_features(sample_transactions, "state", ["state"])

    assert "state" in agg.columns
    assert agg["state"].nunique() > 0


def test_build_training_features_approval_rate(sample_transactions):
    """Approval-rate slices must include approval_rate_z_score."""
    for group_type in APPROVAL_RATE_SLICES:
        cfg_cols = {
            "state": ["state"], 
            "product_type": ["product_type"],
            "overall": None, 
            "new_customer": ["is_new_customer"]
        }

        agg = build_training_features(sample_transactions, group_type, cfg_cols[group_type])
        assert "approval_rate_z_score" in agg.columns, (
            f"approval_rate_z_score missing for slice '{group_type}'"
        )


# Baseline extraction

def test_extract_baseline_stats_overall(sample_transactions):
    """Extracted primary baseline must contain seasonal mean/std indexed by (hour, dow)."""
    agg = build_training_features(sample_transactions, "overall", None)
    primary, fallback = extract_baseline_stats(agg, "volume", None)

    assert "volume_seasonal_mean" in primary.columns
    assert "volume_seasonal_std" in primary.columns
    assert "hour_of_day" in primary.columns
    assert "day_of_week" in primary.columns
    assert not primary.empty

    assert "volume_seasonal_mean" in fallback.columns
    assert not fallback.empty


def test_extract_baseline_stats_state(sample_transactions):
    """State-slice primary baseline must be keyed on (state, hour_of_day, day_of_week)."""
    agg = build_training_features(sample_transactions, "state", ["state"])
    primary, fallback = extract_baseline_stats(agg, "volume", ["state"])

    assert "state" in primary.columns
    assert "hour_of_day" in primary.columns


# Model training

def test_train_one_slice_overall(sample_transactions):
    """train_one_slice for 'overall' must return a valid artifact dict."""
    artifact = train_one_slice(sample_transactions, "overall", None, contamination=0.025)

    assert "model" in artifact
    assert "scaler" in artifact
    assert "features" in artifact
    assert "baselines" in artifact
    assert "volume" in artifact["baselines"]
    assert "primary" in artifact["baselines"]["volume"]
    assert "fallback" in artifact["baselines"]["volume"]
    assert artifact["group_type"] == "overall"
    assert artifact["n_train_rows"] > 0
    assert 0.0 <= artifact["holdout_metrics"]["flag_rate"] <= 1.0


def test_train_one_slice_state(sample_transactions):
    """State-slice artifact must include approval_rate baseline for 'state' (in APPROVAL_RATE_SLICES)."""
    artifact = train_one_slice(sample_transactions, "state", ["state"], contamination=0.025)

    assert artifact["group_type"] == "state"
    assert artifact["group_cols"] == ["state"]
    assert "approval_rate" in artifact["baselines"]


def test_train_one_slice_new_customer(sample_transactions):
    """new_customer slice must include approval_rate baseline (now in APPROVAL_RATE_SLICES)."""
    artifact = train_one_slice(sample_transactions, "new_customer", ["is_new_customer"], contamination=0.025)
    assert "approval_rate" in artifact["baselines"]


def test_get_features_includes_approval_rate_for_applicable_slices(sample_transactions):
    """_get_features must append approval_rate_z_score for all APPROVAL_RATE_SLICES."""
    for group_type in APPROVAL_RATE_SLICES:
        cfg_cols = {
            "state": ["state"], 
            "product_type": ["product_type"],
            "overall": None, 
            "new_customer": ["is_new_customer"]
        }

        agg = build_training_features(sample_transactions, group_type, cfg_cols[group_type])
        features = _get_features(group_type, agg)
        assert "approval_rate_z_score" in features, (
            f"approval_rate_z_score missing from features for '{group_type}'"
        )


def test_train_one_slice_insufficient_data_raises():
    """train_one_slice with < 10 rows must raise ValueError."""
    tiny = pd.DataFrame({
        "date": pd.to_datetime(["2024-10-01 09:00:00", "2024-10-01 10:00:00"]),
        "transaction_id": ["T1", "T2"],
        "customer_id": ["C1", "C2"],
        "product_type": ["mortgage", "auto_loan"],
        "state": ["CA", "TX"],
        "is_new_customer": [True, False],
        "transaction_status": ["approved", "denied"],
    })
    with pytest.raises(ValueError, match="Insufficient"):
        train_one_slice(tiny, "overall", None, contamination=0.025)


def test_artifact_is_serialisable(sample_transactions, tmp_path):
    """The artifact dict must round-trip through joblib without error."""
    import joblib

    artifact = train_one_slice(sample_transactions, "overall", None, contamination=0.025)
    path = tmp_path / "overall_latest.pkl"
    joblib.dump(artifact, path)
    loaded = joblib.load(path)

    assert loaded["group_type"] == "overall"
    assert "model" in loaded
