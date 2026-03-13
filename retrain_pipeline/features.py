"""
Feature engineering for retrain pipeline.

Queries non-live (historical) database and builds hourly aggregates with
self-referential seasonal baselines.
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
    compute_seasonal_baseline,
    compute_velocity,
)

logger = logging.getLogger(__name__)


# SQL Query
_TRAINING_QUERY = text("""
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
ORDER BY transaction_date
""")


def fetch_training_data(lookback_days: int | None = None) -> pd.DataFrame:
    """Pull the last 'lookback_days' of data from the non-live training DB."""
    lookback_days = lookback_days or config.get_lookback_days()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    since = now - timedelta(days=lookback_days)

    with db.get_connection("training") as conn:
        df = pd.read_sql(_TRAINING_QUERY, conn, params={"since": since, "until": now})

    df["date"] = pd.to_datetime(df["date"])
    logger.info( "Fetched training data", extra={"rows": len(df), "lookback_days": lookback_days, "since": str(since)} )

    return df


def build_training_features(
    df: pd.DataFrame,
    group_type: str,
    group_cols: list[str] | None,
) -> pd.DataFrame:
    """
    Build hourly feature aggregates for one slice using self-referential baselines.

    The returned DataFrame contains the seasonal mean/std columns which are
    subsequently extracted and stored in the model artifact for use during live scoring.
    """
    agg = aggregate_hourly(df, group_cols)
    agg = compute_velocity(agg, group_cols)

    # Self-referential baseline: training data is clean so this is safe.
    agg = compute_seasonal_baseline(agg, "volume", group_cols)

    if group_type in APPROVAL_RATE_SLICES:
        agg = compute_seasonal_baseline(agg, "approval_rate", group_cols)

    agg["group_type"] = group_type

    return agg
