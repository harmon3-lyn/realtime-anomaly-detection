"""
Azure Functions v2 entry point.

Registers both timer-triggered functions:
    - anomaly_monitor  - hourly (start of every hour)
    - retrain_pipeline - weekly (Sundays 2:00 UTC)
"""
import logging

import azure.functions as func

from anomaly_monitor import run_monitor
from retrain_pipeline import run_retrain

logger = logging.getLogger(__name__)
app = func.FunctionApp()


@app.timer_trigger(
    schedule="0 0 * * * *",   # Every hour at :00:00
    arg_name="timer",
    run_on_startup=False,
)
def anomaly_monitor(timer: func.TimerRequest) -> None:
    """Hourly anomaly detection across all 5 dimension slices."""
    if timer.past_due:
        logger.warning("anomaly_monitor timer is past due")
    run_monitor()


@app.timer_trigger(
    schedule="0 0 2 * * 0",   # Sunday 02:00 UTC
    arg_name="timer",
    run_on_startup=False,
)
def retrain_pipeline(timer: func.TimerRequest) -> None:
    """Weekly model retrain using 90 days of non-live DB history."""
    if timer.past_due:
        logger.warning("retrain_pipeline timer is past due")
    run_retrain()
