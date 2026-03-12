"""
Structured logger wrapper.

All log records emitted through a bound logger now include 'run_id'
plus any fields passed at bind time. These appear as custom dimensions in
Application Insights when the Functions runtime is configured with the
APPLICATIONINSIGHTS_CONNECTION_STRING setting.
"""
import logging
import uuid


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class RunContext:
    """Unique identifier for single function invocation."""

    def __init__(self) -> None:
        self.run_id = str(uuid.uuid4())[:8]

    def bind(self, logger: logging.Logger, **extra) -> "BoundLogger":
        return BoundLogger(logger, {"run_id": self.run_id, **extra})


class BoundLogger(logging.LoggerAdapter):
    """LoggerAdapter that merges per-invocation fields into every record."""

    def process(self, msg: str, kwargs: dict) -> tuple:
        merged = {**self.extra, **kwargs.pop("extra", {})}
        kwargs["extra"] = merged
        return msg, kwargs
