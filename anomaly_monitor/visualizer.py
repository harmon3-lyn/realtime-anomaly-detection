"""
Visualization: generates a compact PNG per batch of anomalies.

Returns PNG bytes so no disk I/O required. Chart uploaded to Blob 
storage by the notifier and linked from the Teams Adaptive Card.
"""
from __future__ import annotations

import io
import logging
from datetime import timedelta
import matplotlib
matplotlib.use("Agg") 
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from shared.features import SLICES
from anomaly_monitor.detector import AnomalyEvent

logger = logging.getLogger(__name__)


plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _get_group_col(group_type: str) -> str | None:
    cols = SLICES.get(group_type, {}).get("group_cols")
    return cols[0] if cols else None


def generate_anomaly_chart(
    events: list[AnomalyEvent],
    slice_aggs: dict[str, pd.DataFrame],
    max_panels: int = 6,
) -> bytes:
    """
    Generate a multi-panel PNG for the top anomalous groups.

    Parameters:
    events: AnomalyEvent list from detect_anomalies.
    slice_aggs: Hourly aggregated DataFrames keyed by group_type.
    max_panels: Max number of subplots.

    Returns:
    PNG bytes, or empty bytes if no events.
    """
    if not events:
        return b""

    # Rank unique (group_type, group_value) panels by max |z_score|
    seen: set[tuple[str, str]] = set()
    panels: list[tuple[str, str]] = []

    for e in sorted(events, key=lambda e: abs(e.z_score), reverse=True):
        key = (e.group_type, e.group_value)
        if key not in seen:
            seen.add(key)
            panels.append(key)

        if len(panels) >= max_panels:
            break

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), squeeze=False)

    for i, (gtype, gval) in enumerate(panels):
        ax = axes[i][0]
        agg = slice_aggs.get(gtype, pd.DataFrame())

        if agg.empty:
            ax.set_title(f"{gtype} = {gval}  (no data)")
            continue

        group_col = _get_group_col(gtype)
        grp = (agg[agg[group_col].astype(str) == gval].copy() if group_col else agg.copy())
        grp = grp.sort_values("hour_bucket")

        # 7-day context window centred on earliest anomaly in this panel
        panel_events = [e for e in events if e.group_type == gtype and e.group_value == gval]

        if panel_events:
            anchor = min(e.timestamp for e in panel_events)
            w_start = pd.Timestamp(anchor) - timedelta(days=3)
            w_end = pd.Timestamp(anchor) + timedelta(days=4)
            grp = grp[(grp["hour_bucket"] >= w_start) & (grp["hour_bucket"] <= w_end)]

        if grp.empty:
            ax.set_title(f"{gtype} = {gval}  (no data in context window)")
            continue

        # Volume time-series
        ax.plot(grp["hour_bucket"], grp["volume"], color="steelblue", linewidth=1.3, label="Volume", zorder=2)

        # Seasonal mean ± 2 st dev band
        if "volume_seasonal_mean" in grp.columns:
            m = grp["volume_seasonal_mean"]
            s = grp.get("volume_seasonal_std", pd.Series([0] * len(grp), index=grp.index))

            ax.plot(
                grp["hour_bucket"], m,
                color="steelblue", linewidth=0.9, linestyle="--",
                alpha=0.55, label="Seasonal mean",
            )
            ax.fill_between(
                grp["hour_bucket"],
                (m - 2 * s).clip(lower=0), m + 2 * s,
                alpha=0.12, color="steelblue", label="±2σ expected",
            )

        # Anomaly markers
        annotated: set[tuple] = set()
        for e in panel_events:
            pt = grp[grp["hour_bucket"] == pd.Timestamp(e.timestamp)]
            if pt.empty:
                continue

            x_val = pt["hour_bucket"].values[0]
            y_val = pt["volume"].values[0]

            ax.scatter(x_val, y_val, color="crimson", s=100, zorder=5)
            key = (e.anomaly_type, round(e.z_score, 0))

            if key not in annotated:
                ax.annotate(
                    f"{e.anomaly_type}\nz={e.z_score:+.1f}  [{e.detected_by}]",
                    xy=(x_val, y_val),
                    xytext=(14, 8), textcoords="offset points",
                    fontsize=7.5, color="crimson",
                    arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
                )
                annotated.add(key)

        ax.set_title(f"{gtype} = {gval}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Transactions / hr", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:00"))
        ax.tick_params(axis="x", labelsize=7)
        if i == 0:
            ax.legend(fontsize=7.5, loc="upper left")

    axes[-1][0].set_xlabel("Hour bucket (UTC)", fontsize=9)
    plt.suptitle("Anomaly Detection - Hourly volume with seasonality baseline", fontsize=11, fontweight="bold", y=1.005)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
