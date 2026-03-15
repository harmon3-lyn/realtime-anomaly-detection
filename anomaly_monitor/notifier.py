"""
Notifier: posts Teams Adaptive Card with inline chart image.

Delivery: PNG is uploaded to 'anomaly-charts' Blob container
with a 1 hr SAS URL, then referenced in the Adaptive Card Image element.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional
import requests

from shared import blob as blob_store, config
from anomaly_monitor.detector import AnomalyEvent

logger = logging.getLogger(__name__)


def _build_card(events: list[AnomalyEvent], chart_url: Optional[str]) -> dict:
    """Construct Teams Adaptive Card payload."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = len(events)
    high_count = sum(1 for e in events if e.severity == "high")

    # Group events by group_type
    by_type: dict[str, list[AnomalyEvent]] = {}
    for e in events:
        by_type.setdefault(e.group_type, []).append(e)

    body: list[dict] = [
        {
            "type": "TextBlock",
            "text": f"\u26a0 {n} anomal{'y' if n == 1 else 'ies'} detected \u2014 {now_str}",
            "weight": "bolder",
            "size": "medium",
            "color": "attention" if high_count else "warning",
        }
    ]

    for group_type, grp_events in sorted(by_type.items()):
        body.append({
            "type": "TextBlock",
            "text": f"**{group_type.replace('_', ' ').title()}**",
            "weight": "bolder",
            "spacing": "medium",
        })
        facts = [
            {
                "title": f"{e.group_value} - {e.anomaly_type}",
                "value": (
                    f"severity={e.severity}, z={e.z_score:+.2f}, "
                    f"current={e.current_value}, expected={e.expected_value}, "
                    f"detected_by={e.detected_by}"
                ),
            }
            for e in sorted(grp_events, key=lambda e: abs(e.z_score), reverse=True)
        ]
        body.append({"type": "FactSet", "facts": facts})

    if chart_url:
        body.append({
            "type": "Image",
            "url": chart_url,
            "altText": "Anomaly detection chart",
            "size": "stretch",
        })

    return {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": body,
                },
            }
        ],
    }



def send_notification(
    events: list[AnomalyEvent],
    chart_bytes: Optional[bytes] = None,
) -> None:
    """Post Teams Adaptive Card; upload chart to Blob if provided."""
    webhook_url = config.get_teams_webhook_url()
    chart_url: Optional[str] = None

    if chart_bytes:
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            blob_name = f"charts/anomaly_{ts}.png"
            chart_url = blob_store.upload_chart(chart_bytes, blob_name)
            logger.info("Chart uploaded", extra={"blob_name": blob_name})

        except Exception as exc:
            logger.warning("Chart upload failed; sending card without image", extra={"error": str(exc)})

    payload = _build_card(events, chart_url)
    resp = requests.post(
        webhook_url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=15,
    )

    resp.raise_for_status()
    logger.info( "Notification sent", extra={"http_status": resp.status_code, "n_events": len(events)} )
