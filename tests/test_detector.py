"""
Unit tests for anomaly detector.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from shared.features import (
    BASE_IF_FEATURES,
    aggregate_hourly,
    compute_seasonal_baseline,
    compute_velocity,
)
from anomaly_monitor.detector import AnomalyEvent, detect_anomalies, score_slice


# Helpers

def _build_scored_agg(
    df_train: pd.DataFrame,
    df_score: pd.DataFrame,
    group_cols: list[str] | None = None,
    contamination: float = 0.05,
) -> pd.DataFrame:
    
    """Build + score features without touching any external services."""
    train_agg = aggregate_hourly(df_train, group_cols)
    score_agg = aggregate_hourly(df_score, group_cols)
    train_agg = compute_velocity(train_agg, group_cols)
    score_agg = compute_velocity(score_agg, group_cols)
    train_agg = compute_seasonal_baseline(train_agg, "volume", group_cols)
    score_agg = compute_seasonal_baseline(score_agg, "volume", group_cols, source_agg=train_agg)

    features = BASE_IF_FEATURES
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_agg[features].fillna(0).values)

    model = IsolationForest(contamination=contamination, n_estimators=50, random_state=0)
    model.fit(X_train)

    X_score = scaler.transform(score_agg[features].fillna(0).values)

    score_agg = score_agg.copy()
    score_agg["if_flag"] = model.predict(X_score)
    score_agg["if_score"] = model.decision_function(X_score)

    return score_agg



# Tests

def test_anomaly_event_structure(split_dfs):
    """All returned AnomalyEvent objects must have valid field values."""
    df_train, df_score = split_dfs
    scored = _build_scored_agg(df_train, df_score)
    events = detect_anomalies(scored, "overall", None)

    for e in events:
        assert isinstance(e, AnomalyEvent)
        assert e.group_type == "overall"
        assert e.group_value == "all"
        assert e.anomaly_type in {"volume_spike", "volume_drop", "velocity_spike", "velocity_drop", "approval_rate_drop"}
        assert e.severity in {"high", "medium"}
        assert e.detected_by in {"both", "statistical", "ml"}

        assert isinstance(e.z_score, float)
        assert isinstance(e.ml_score, float)
        assert isinstance(e.current_value, float)
        assert isinstance(e.expected_value, float)


def test_empty_agg_returns_no_events(split_dfs):
    """An empty aggregation must produce an empty event list without crashing."""
    _, df_score = split_dfs
    empty_agg = aggregate_hourly(df_score.iloc[:0])
    events = detect_anomalies(empty_agg, "overall", None)
    assert events == []


def test_score_slice_without_model(split_dfs):
    """score_slice with artifact=None must default if_flag=1 (no anomaly) for all rows."""
    _, df_score = split_dfs
    agg = aggregate_hourly(df_score)
    agg = compute_velocity(agg)
    agg = compute_seasonal_baseline(agg, "volume")

    scored = score_slice(agg, "overall", artifact=None)
    assert "if_flag" in scored.columns
    assert "if_score" in scored.columns
    assert (scored["if_flag"] == 1).all()


def test_group_value_populated_for_state_slice(split_dfs):
    """Detected events in the state slice must have a valid state code as group_value."""
    df_train, df_score = split_dfs
    scored = _build_scored_agg(df_train, df_score, group_cols=["state"])
    events = detect_anomalies(scored, "state", "state")

    valid_states = {"CA", "TX", "FL", "NY", "WA"}
    for e in events:
        assert e.group_value in valid_states, f"Unexpected group_value: {e.group_value!r}"


def test_severity_high_above_z4(split_dfs):
    """Events with |z_score| >= 4.0 must be classified as 'high' severity."""
    df_train, df_score = split_dfs
    scored = _build_scored_agg(df_train, df_score)
    events = detect_anomalies(scored, "overall", None)

    for e in events:
        if abs(e.z_score) >= 4.0:
            assert e.severity == "high", (
                f"Expected 'high' for z={e.z_score:.2f} but got {e.severity!r}"
            )
        else:
            assert e.severity == "medium"


def test_injected_spike_detected(sample_transactions):
    """
    An injected volume spike in the scoring window must be detected by both layers.

    Mirrors the A1 anomaly from the evaluation notebook: injects 25 extra rows at a specific hour.
    """
    from datetime import timedelta
    import numpy as np

    rng = np.random.default_rng(1)
    df = sample_transactions.copy()
    train_cutoff = pd.Timestamp("2024-10-17")

    # Inject spike in the scoring window: hour that has only 0-1 rows baseline
    spike_hour = pd.Timestamp("2024-10-20 14:00:00")
    spike_rows = pd.DataFrame({
        "date": [spike_hour + timedelta(minutes=int(rng.integers(0, 60))) for _ in range(25)],
        "transaction_id": [f"SPK{i:03d}" for i in range(25)],
        "customer_id": [f"C{rng.integers(1, 50):04d}" for _ in range(25)],
        "product_type": ["auto_loan"] * 25,
        "state": ["TX"] * 25,
        "is_new_customer": [False] * 25,
        "transaction_status": ["approved"] * 25,
    })

    df_all = pd.concat([df, spike_rows], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])

    df_train = df_all[df_all["date"] < train_cutoff]
    df_score = df_all[df_all["date"] >= train_cutoff]

    scored = _build_scored_agg(df_train, df_score, group_cols=["state"])
    events = detect_anomalies(scored, "state", "state")

    spike_events = [
        e for e in events
        if e.group_value == "TX" and pd.Timestamp(e.timestamp) == spike_hour
    ]
    assert spike_events, "Injected volume spike in TX was not detected"
    assert spike_events[0].z_score > 3.0


def test_aggregate_hourly_empty_input():
    """aggregate_hourly on an empty DataFrame must return a valid empty DataFrame."""
    empty = pd.DataFrame(columns=["date", "transaction_id", "transaction_status", "state", "product_type", "is_new_customer", "customer_id"])
    result = aggregate_hourly(empty)

    assert result.empty
    assert "hour_bucket" in result.columns


def test_seasonal_baseline_cross_group_fallback(sample_transactions):
    """
    Fallback must fill NaN z-scores for scoring-window (group, hour, day_of_week) 
    combos absent from the training window.
    """
    from shared.features import compute_seasonal_baseline, aggregate_hourly, compute_velocity

    df_train = sample_transactions[sample_transactions["date"] < pd.Timestamp("2024-10-15")]
    df_score = sample_transactions[sample_transactions["date"] >= pd.Timestamp("2024-10-15")]

    train_agg = aggregate_hourly(df_train, ["state"])
    score_agg = aggregate_hourly(df_score, ["state"])
    train_agg = compute_velocity(train_agg, ["state"])
    score_agg = compute_velocity(score_agg, ["state"])
    train_agg = compute_seasonal_baseline(train_agg, "volume", ["state"])
    score_agg = compute_seasonal_baseline(score_agg, "volume", ["state"], source_agg=train_agg)

    # No NaN z-scores in the scoring window after fallback fills them
    assert score_agg["volume_z_score"].notna().all(), (
        "Found NaN z-scores after cross-group fallback application"
    )
