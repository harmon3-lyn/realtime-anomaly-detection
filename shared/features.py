"""
Core feature engineering functions shared between anomaly_monitor and retrain_pipeline.

Note: z-scores are grouped by (hour_of_day, day_of_week) over the
clean training window, vs a naive 24-hour rolling mean, avoiding false positives 
during every low-volume off-peak period.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# Constants

BASE_IF_FEATURES = ["volume_z_score", "velocity_pct", "hour_of_day", "day_of_week"]
APPROVAL_IF_FEATURE = "approval_rate_z_score"

# Slice configuration: name -> group columns
SLICES: dict[str, dict] = {
    "overall":            {"group_cols": None},
    "state":              {"group_cols": ["state"]},
    "product_type":       {"group_cols": ["product_type"]},
    "transaction_status": {"group_cols": ["transaction_status"]},
    "new_customer":       {"group_cols": ["is_new_customer"]},
}

# Slices with approval_rate signal
APPROVAL_RATE_SLICES: frozenset[str] = frozenset({"state", "product_type", "overall", "new_customer"})


# Aggregation

def aggregate_hourly(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate raw transaction rows to hourly counts per group slice.

    Returns a DataFrame with columns:
    - hour_bucket, [group_cols], volume, n_approved, approval_rate, hour_of_day, day_of_week
    """
    if df.empty:
        cols = ( ["hour_bucket"] + (group_cols or []) + ["volume", "n_approved", "approval_rate", "hour_of_day", "day_of_week"] )
        return pd.DataFrame(columns=cols)

    d = df.copy()
    d["hour_bucket"] = d["date"].dt.floor("h")
    by = ["hour_bucket"] + (group_cols or [])

    agg = (d.groupby(by).agg( volume=("transaction_id", "count"), n_approved=("transaction_status", lambda x: (x == "approved").sum())).reset_index())
    
    agg["approval_rate"] = agg["n_approved"] / agg["volume"]
    agg["hour_of_day"] = agg["hour_bucket"].dt.hour
    agg["day_of_week"] = agg["hour_bucket"].dt.dayofweek  # 0 = Monday

    return agg


# Velocity

def compute_velocity(
    agg: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add hour-over-hour volume delta (velocity) and relative rate (velocity_pct).
    velocity_pct capped to avoid unbounded breaks.
    """
    if agg.empty:
        agg = agg.copy()
        agg["velocity"] = pd.Series(dtype=float)
        agg["velocity_pct"] = pd.Series(dtype=float)
        return agg

    agg = agg.sort_values(["hour_bucket"] + (group_cols or [])).copy()

    if group_cols:
        agg["velocity"] = agg.groupby(group_cols)["volume"].diff()
        prior = agg.groupby(group_cols)["volume"].shift(1)
        agg["velocity_pct"] = agg["velocity"] / prior

    else:
        agg["velocity"] = agg["volume"].diff()
        agg["velocity_pct"] = agg["velocity"] / agg["volume"].shift(1)

    agg["velocity_pct"] = agg["velocity_pct"].fillna(0).clip(-5, 5) # 0 for first observation (null)

    return agg


# Seasonality

def compute_seasonal_baseline(
    agg: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None = None,
    min_samples: int = 4,
    source_agg: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute seasonal mean/std grouped by (group_cols, hour_of_day, day_of_week)
    and join them onto 'agg'.

    Parameters:
    - source_agg
        If provided, stats are derived from this DataFrame (the clean training
        window) and joined onto 'agg' (the scoring window).  This prevents an
        anomaly from biasing its own baseline - mirroring the non-live DB
        (train) vs live DB (score) split in production.
        If None, the baseline is self-referential (used for training-window
        features where training data is known to be clean).

    - Cross-group fallback
        For scoring-window buckets with no matching (group x hour x dow) bucket
        in the training window, the stats fall back to the cross-group
        (hour_of_day, day_of_week) mean from source_agg.
    """
    if agg.empty:
        agg = agg.copy()
        for suffix in ("seasonal_mean", "seasonal_std", "z_score"):
            agg[f"{value_col}_{suffix}"] = pd.Series(dtype=float)
        return agg

    source = source_agg if source_agg is not None else agg
    bucket_cols = (group_cols or []) + ["hour_of_day", "day_of_week"]

    # Primary: per-(group_value x hour_of_day x day_of_week) stats
    stats = (
        source.groupby(bucket_cols)[value_col]
        .agg(s_mean="mean", s_std="std", n="count")
        .reset_index()
        .rename(columns={
            "s_mean": f"{value_col}_seasonal_mean",
            "s_std":  f"{value_col}_seasonal_std",
            "n":      f"{value_col}_n_samples",
        })
    )
    stats[f"{value_col}_seasonal_std"] = (
        stats[f"{value_col}_seasonal_std"].fillna(1.0).clip(lower=0.5)
    )


    # Fallback: cross-group (hour_of_day x day_of_week) stats
    fallback = (
        source.groupby(["hour_of_day", "day_of_week"])[value_col]
        .agg(fb_mean="mean", fb_std="std")
        .reset_index()
    )
    fallback["fb_std"] = fallback["fb_std"].fillna(1.0).clip(lower=0.5)

    agg = agg.merge(stats, on=bucket_cols, how="left")
    agg = agg.merge(fallback, on=["hour_of_day", "day_of_week"], how="left")

    mask = agg[f"{value_col}_seasonal_mean"].isna()
    agg.loc[mask, f"{value_col}_seasonal_mean"] = agg.loc[mask, "fb_mean"]
    agg.loc[mask, f"{value_col}_seasonal_std"]  = agg.loc[mask, "fb_std"]
    agg.drop(columns=["fb_mean", "fb_std"], inplace=True)


    # Global fallback: for (hour, day_of_week) combos absent from entire training window, use the overall source mean/std so z-scores are never null.
    mask2 = agg[f"{value_col}_seasonal_mean"].isna()

    if mask2.any():
        global_mean = float(source[value_col].mean())
        global_std = max(float(source[value_col].std() or 1.0), 0.5)
        agg.loc[mask2, f"{value_col}_seasonal_mean"] = global_mean
        agg.loc[mask2, f"{value_col}_seasonal_std"]  = global_std

    agg[f"{value_col}_z_score"] = ( (agg[value_col] - agg[f"{value_col}_seasonal_mean"]) / agg[f"{value_col}_seasonal_std"] )
    
    return agg



# Pre-computed baseline application (monitor path)

def apply_precomputed_baselines(
    agg: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None,
    primary: pd.DataFrame,
    fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge pre-computed seasonal stats (from model artifact) onto a scoring-window agg and compute z-scores.

    Production equivalent of passing 'source_agg' to
    'compute_seasonal_baseline': stats come from the stored training baseline,
    not re-computed from the (sparse) live window.

    Parameters:
    - primary
        DataFrame with (group_cols, hour_of_day, day_of_week) as keys and
        '{value_col}_seasonal_mean' / '{value_col}_seasonal_std' as values.
    - fallback
        Cross-group (hour_of_day, day_of_week) baseline for unseen group values.
        If None, unseen buckets will have NaN z-scores.
    """
    if agg.empty:
        agg = agg.copy()
        for suffix in ("seasonal_mean", "seasonal_std", "z_score"):
            agg[f"{value_col}_{suffix}"] = pd.Series(dtype=float)
        return agg

    mean_col = f"{value_col}_seasonal_mean"
    std_col = f"{value_col}_seasonal_std"
    bucket_cols = (group_cols or []) + ["hour_of_day", "day_of_week"]

    merge_cols = [c for c in bucket_cols if c in primary.columns]
    agg = agg.merge(
        primary[[*merge_cols, mean_col, std_col]],
        on=merge_cols,
        how="left",
    )

    if fallback is not None:
        fb_mean = f"_fb_{value_col}_mean"
        fb_std = f"_fb_{value_col}_std"
        fallback_renamed = fallback.rename(columns={mean_col: fb_mean, std_col: fb_std})

        agg = agg.merge(
            fallback_renamed[["hour_of_day", "day_of_week", fb_mean, fb_std]],
            on=["hour_of_day", "day_of_week"],
            how="left",
        )

        mask = agg[mean_col].isna()
        agg.loc[mask, mean_col] = agg.loc[mask, fb_mean]
        agg.loc[mask, std_col] = agg.loc[mask, fb_std]
        agg.drop(columns=[fb_mean, fb_std], inplace=True)


    agg[std_col] = agg[std_col].fillna(1.0).clip(lower=0.5) # Ensure never 0
    agg[f"{value_col}_z_score"] = ((agg[value_col] - agg[mean_col]) / agg[std_col])

    return agg



# Baseline extraction (trainer path)

def extract_baseline_stats(
    agg: pd.DataFrame,
    value_col: str,
    group_cols: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract seasonal stats from a training agg.

    Returns:
    - primary: (group_cols, hour_of_day, day_of_week) -> (seasonal_mean, seasonal_std)
    - fallback: (hour_of_day, day_of_week) -> (seasonal_mean, seasonal_std) - cross-group
    """
    mean_col = f"{value_col}_seasonal_mean"
    std_col = f"{value_col}_seasonal_std"
    bucket_cols = (group_cols or []) + ["hour_of_day", "day_of_week"]

    primary = (agg[bucket_cols + [mean_col, std_col]].drop_duplicates(subset=bucket_cols).reset_index(drop=True))

    # Cross-group fallback: re-aggregate from raw values
    fallback = (
        agg.groupby(["hour_of_day", "day_of_week"])[value_col]
        .agg(s_mean="mean", s_std="std")
        .reset_index()
        .rename(columns={"s_mean": mean_col, "s_std": std_col})
    )
    fallback[std_col] = fallback[std_col].fillna(1.0).clip(lower=0.5)

    return primary, fallback
