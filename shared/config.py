"""
Central accessor for all environment variables.
Creates ValueError on startup if required variable is missing.
"""
import os


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise ValueError(f"Required environment variable '{key}' is not set.")
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default)



# Databases

def get_live_conn_str() -> str:
    return _require("DB_LIVE_CONN_STR")

def get_training_conn_str() -> str:
    return _require("DB_TRAINING_CONN_STR")


# Azure Blob Storage

def get_storage_conn_str() -> str:
    return _optional("AZURE_STORAGE_CONNECTION_STRING")

def get_storage_account_url() -> str:
    return _optional("AZURE_STORAGE_ACCOUNT_URL")

def get_model_container() -> str:
    return _optional("MODEL_CONTAINER_NAME", "anomaly-models")

def get_chart_container() -> str:
    return _optional("CHART_CONTAINER_NAME", "anomaly-charts")


# Detection Thresholds

def get_zscore_threshold() -> float:
    return float(_optional("ZSCORE_THRESH", "3.0"))

def get_high_threshold() -> float:
    return float(_optional("HIGH_THRESH", "4.5"))

def get_velocity_threshold() -> float:
    return float(_optional("VELOCITY_THRESH", "2.0"))

def get_ml_score_threshold() -> float:
    return float(_optional("ML_SCORE_THRESH", "-0.15"))

def get_min_volume() -> int:
    return int(_optional("MIN_VOLUME", "5"))


# Parameters for Retrain

def get_contamination() -> float:
    return float(_optional("RETRAIN_CONTAMINATION", "0.025"))

def get_lookback_days() -> int:
    return int(_optional("RETRAIN_LOOKBACK_DAYS", "90"))

def get_monitor_window_hours() -> int:
    return int(_optional("MONITOR_WINDOW_HOURS", "24"))


# Notification Webhook

def get_teams_webhook_url() -> str:
    return _require("TEAMS_WEBHOOK_URL")
