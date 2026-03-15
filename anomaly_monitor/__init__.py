"""
anomaly_monitor - Azure Function entry point (hourly timer trigger).

Flow
----
1. Fetch 24 h of live transactions (wider window for visualization context).
2. For each of the 5 dimension slices:
   a. Load the slice model artifact from Blob Storage (cached in /tmp).
   b. Build hourly feature aggregates using stored seasonal baselines.
   c. Score the last 2 h with the Isolation Forest.
   d. Run the two-layer detector to produce AnomalyEvent objects.
3. Generate a PNG chart (up to 6 panels, ranked by |z_score|).
4. Upload the chart to Blob Storage; post a Teams Adaptive Card.

Error strategy
--------------
- Live DB unreachable  -> log error, return (do not crash the function host)
- No model for a slice -> log warning, skip that slice
- Notification failure -> log error (anomaly record is not suppressed)
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

from shared import config, logging_utils
from shared.features import SLICES
from anomaly_monitor.features import build_scoring_features, fetch_monitor_data
from anomaly_monitor.detector import AnomalyEvent, detect_anomalies, load_artifact, score_slice
from anomaly_monitor.visualizer import generate_anomaly_chart
from anomaly_monitor.notifier import send_notification

logger = logging.getLogger(__name__)

__all__ = ["run_monitor"]


def run_monitor() -> None:
    """Entry point called by the anomaly_monitor Azure Function."""
    run_ctx = logging_utils.RunContext()
    log = run_ctx.bind(logger, function="anomaly_monitor")
    log.info("Anomaly monitor started")

    t0 = time.monotonic()
    window_hours = config.get_monitor_window_hours()
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    detect_since = now - timedelta(hours=2)  # 2-hour detection window

    # ── 1. Fetch live data ────────────────────────────────────────────────────
    try:
        df = fetch_monitor_data(window_hours=window_hours)
    except Exception as exc:
        log.error("Live DB unreachable - skipping run", extra={"error": str(exc)})
        return

    if df.empty:
        log.info("No transactions in monitor window; skipping detection")
        return

    # ── 2. Per-slice detection ────────────────────────────────────────────────
    all_events: list[AnomalyEvent] = []
    slice_aggs: dict[str, object] = {}  # full 24-h aggs for the visualizer

    for group_type, cfg in SLICES.items():
        group_cols = cfg["group_cols"]
        group_value_col = group_cols[0] if group_cols else None

        # Load model artifact (Blob -> /tmp cache)
        artifact = load_artifact(group_type)
        if artifact is None:
            log.warning("No model artifact; skipping slice", extra={"group_type": group_type})
            continue

        baselines = artifact.get("baselines", {})
        vol_b = baselines.get("volume", {})
        ar_b = baselines.get("approval_rate", {})

        # Build full 24-h agg (for visualization) using stored baselines
        try:
            full_agg = build_scoring_features(
                df, group_type, group_cols,
                primary_baseline_volume=vol_b.get("primary"),
                fallback_baseline_volume=vol_b.get("fallback"),
                primary_baseline_approval=ar_b.get("primary"),
                fallback_baseline_approval=ar_b.get("fallback"),
            )
        except Exception as exc:
            log.error(
                "Feature engineering failed",
                extra={"group_type": group_type, "error": str(exc)},
            )
            continue

        slice_aggs[group_type] = full_agg

        # Narrow to the 2-hour detection window before scoring
        detect_agg = full_agg[full_agg["hour_bucket"] >= detect_since].copy()

        # Score with Isolation Forest
        detect_agg = score_slice(detect_agg, group_type, artifact)

        # Two-layer detection
        events = detect_anomalies(detect_agg, group_type, group_value_col)
        all_events.extend(events)

        log.info(
            "Slice scored",
            extra={
                "group_type": group_type,
                "n_rows_full": len(full_agg),
                "n_rows_detect": len(detect_agg),
                "n_events": len(events),
            },
        )

    elapsed_ms = round((time.monotonic() - t0) * 1000)
    log.info(
        "Detection complete",
        extra={"total_events": len(all_events), "duration_ms": elapsed_ms},
    )

    if not all_events:
        return

    # ── 3. Visualization ──────────────────────────────────────────────────────
    chart_bytes: bytes | None = None
    try:
        chart_bytes = generate_anomaly_chart(all_events, slice_aggs)
    except Exception as exc:
        log.error("Chart generation failed", extra={"error": str(exc)})

    # ── 4. Notify ─────────────────────────────────────────────────────────────
    try:
        send_notification(all_events, chart_bytes)
    except Exception as exc:
        log.error("Notification failed", extra={"error": str(exc)})
