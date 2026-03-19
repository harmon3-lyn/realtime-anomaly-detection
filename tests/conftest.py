"""
Shared pytest fixtures for unit tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """200 synthetic transactions spanning ~3 weeks."""
    rng = np.random.default_rng(0)
    n = 200
    start = datetime(2024, 10, 1)
    dates = [start + timedelta(hours=int(rng.integers(0, 504))) for _ in range(n)]

    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "transaction_id": [f"T{i:05d}" for i in range(n)],
        "customer_id": [f"C{rng.integers(1, 50):04d}" for _ in range(n)],
        "product_type": rng.choice(["mortgage", "auto_loan", "personal_loan", "credit_card"], n).tolist(),
        "state": rng.choice(["CA", "TX", "FL", "NY", "WA"], n).tolist(),
        "is_new_customer": (rng.random(n) < 0.2).tolist(),
        "transaction_status": rng.choice(["approved", "denied"], n, p=[0.8, 0.2]).tolist(),
    })


@pytest.fixture
def split_dfs(sample_transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sample_transactions into chronological train (~70%) and score (~30%)."""
    df = sample_transactions.sort_values("date").reset_index(drop=True)
    
    cutoff_idx = int(len(df) * 0.7)
    cutoff = df["date"].iloc[cutoff_idx]

    return df[df["date"] < cutoff].copy(), df[df["date"] >= cutoff].copy()
