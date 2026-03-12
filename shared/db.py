"""
Database connection; both live (scoring) and training (non-live) databases use SQL 
Server via the mssql+pyodbc dialect with ODBC Driver 18.

Connection strings are read from environment variables:
  DB_LIVE_CONN_STR - live transaction database (queried hourly by monitor)
  DB_TRAINING_CONN_STR - historical database (queried weekly by retrain pipeline)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Literal

from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from shared import config

logger = logging.getLogger(__name__)

_engines: dict[str, Engine] = {}


def get_engine(target: Literal["live", "training"]) -> Engine: 
    """Return cached SQLAlchemy engine for requested database."""
    if target not in _engines:
        conn_str = (
            config.get_live_conn_str() if target == "live" # prod
            else config.get_training_conn_str()
        )
        _engines[target] = create_engine(
            conn_str,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=5,
            pool_recycle=1800,
        )
        logger.info("Created DB engine", extra={"target": target})
    return _engines[target]


@contextmanager
def get_connection(target: Literal["live", "training"]) -> Generator[Connection, None, None]:
    """Context manager yielding single DB connection; close on exit."""
    engine = get_engine(target)

    with engine.connect() as conn:
        yield conn
