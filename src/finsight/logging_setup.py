"""
Structured logging factory — #8.

All modules import get_logger() from here instead of calling print().
Outputs JSON lines to stderr by default so Streamlit's stdout stays clean.

Log level is controlled by the FINSIGHT_LOG_LEVEL env var (default INFO).
Set to DEBUG for verbose per-call tracing, WARNING to quiet everything down.

Example structured log line:
  {"ts": "2026-08-17T14:05:03Z", "level": "INFO", "name": "finsight.config",
   "msg": "task='routing' -> model='llama-3.1-8b-instant'"}
"""
import json
import logging
import os
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Emits one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        payload = {
            "ts": ts,
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured with JSON output.

    Call once per module at module level:
        logger = get_logger(__name__)
    """
    logger = logging.getLogger(name)

    # Only add a handler if none exists yet (prevents duplicate lines when
    # the same module is imported multiple times in a session).
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)

    level_name = os.getenv("FINSIGHT_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    # Prevent log records from bubbling up to the root logger (which might
    # have its own handler that would print duplicates).
    logger.propagate = False

    return logger
