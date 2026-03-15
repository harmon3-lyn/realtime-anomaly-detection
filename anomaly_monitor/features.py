"""
Feature engineering for the anomaly monitor (live DB -> scoring window).

The monitor fetches a longer context window (default 24h) for visualization
and uses pre-computed seasonal baselines from the model artifact, so the
2-hour detection window cannot bias its own z-scores.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
import pandas as pd
from sqlalchemy import text
from shared import config, db
from shared.features import (
    APPROVAL_RATE_SLICES,
    aggregate_hourly,
    apply_precomputed_baselines,
    compute_seasonal_baseline,
    compute_velocity,
)

logger = logging.getLogger(__name__)


# SQL Query
_MONITOR_QUERY = text("""
SELECT
    transaction_id,
    customer_id,
    product_type,
    state,
    CAST(is_new_customer AS BIT) AS is_new_customer,
    transaction_status,
    transaction_date AS date
FROM transactions
WHERE transaction_date >= :since
  AND transaction_date <  :until
""")


def fetch_monitor_data(window_hours: int | None = None) -> pd.DataFrame:
    """Pull the last 'window_hours' of data from live DB."""
    window_hours = window_hours or config.get_monitor_window_hours()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    since = now - timedelta(hours=window_hours)

    with db.get_connection("live") as conn:
        df = pd.read_sql(_MONITOR_QUERY, conn, params={"since": since, "until": now})

    df["date"] = pd.to_datetime(df["date"])
    logger.info("Fetched monitor data", extra={"rows": len(df), "window_hours": window_hours, "since": str(since)})
    
    return df


def build_scoring_features(
    df: pd.DataFrame,
    group_type: str,
    group_cols: list[str] | None,
    primary_baseline_volume: pd.DataFrame | None,
    fallback_baseline_volume: pd.DataFrame | None = None,
    primary_baseline_approval: pd.DataFrame | None = None,
    fallback_baseline_approval: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build hourly feature aggregates for one slice using stored baselines.

    When pre-computed baselines are available, they're applied via 'apply_precomputed_baselines' 
    so live window can't contaminate seasonal means. If no baselines are provided, falls back to 
    self-referential baselines to prevent crash.
    """
    agg = aggregate_hourly(df, group_cols)
    agg = compute_velocity(agg, group_cols)

    if primary_baseline_volume is not None and not primary_baseline_volume.empty:
        agg = apply_precomputed_baselines( agg, "volume", group_cols, primary_baseline_volume, fallback_baseline_volume)
    else:
        # Fallback:
        agg = compute_seasonal_baseline(agg, "volume", group_cols)

    if (
        group_type in APPROVAL_RATE_SLICES
        and primary_baseline_approval is not None
        and not primary_baseline_approval.empty
    ):
        agg = apply_precomputed_baselines( agg, "approval_rate", group_cols, primary_baseline_approval, fallback_baseline_approval)

    agg["group_type"] = group_type

    return agg
