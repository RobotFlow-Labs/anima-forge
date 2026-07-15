"""CLI logging configuration.

Provides rotating file logging with predictable defaults:
- maxBytes=1,000,000 (1 MB)
- backupCount=2
"""

from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

CLI_LOG_MAX_BYTES = 1_000_000
CLI_LOG_BACKUP_COUNT = 2


class _JsonLogFormatter(logging.Formatter):
    """Emit one JSON object per line for automation consumers."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _resolve_log_path(path: str) -> Path:
    resolved = Path(os.path.expanduser(path)).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _clear_cli_handlers(logger: logging.Logger) -> None:
    """Remove handlers previously installed by FORGE CLI logging."""
    for handler in list(logger.handlers):
        if getattr(handler, "_forge_cli_managed", False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def setup_cli_logging(
    log_file: str | None,
    log_level: str = "INFO",
    log_to_stdout: bool = False,
    log_json: bool = False,
) -> None:
    """Configure the shared FORGE CLI logger.

    Logging defaults to no-op if already configured in this process. If `log_file`
    is provided, use a rotating file handler capped at 1MB with 2 backup files.
    """
    logger = logging.getLogger("forge")
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    _clear_cli_handlers(logger)

    formatter: logging.Formatter
    if log_json:
        formatter = _JsonLogFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    configured = False

    if log_file:
        try:
            path = _resolve_log_path(log_file)
        except OSError:
            fallback_handler = logging.StreamHandler()
            setattr(fallback_handler, "_forge_cli_managed", True)
            fallback_handler.setFormatter(formatter)
            logger.addHandler(fallback_handler)
            logger.warning("Could not initialize log file path '%s'; logging to stdout fallback only.", log_file)
            setattr(logger, "_forge_cli_configured", True)
            setattr(
                logger,
                "_forge_cli_rotation",
                {
                    "max_bytes": CLI_LOG_MAX_BYTES,
                    "backup_count": CLI_LOG_BACKUP_COUNT,
                },
            )
            return
        else:
            file_handler = RotatingFileHandler(
                path,
                maxBytes=CLI_LOG_MAX_BYTES,
                backupCount=CLI_LOG_BACKUP_COUNT,
                encoding="utf-8",
            )
            setattr(file_handler, "_forge_cli_managed", True)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            configured = True

    if log_to_stdout:
        stdout_handler = logging.StreamHandler()
        setattr(stdout_handler, "_forge_cli_managed", True)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        configured = True

    if not configured:
        null_handler = logging.NullHandler()
        setattr(null_handler, "_forge_cli_managed", True)
        logger.addHandler(null_handler)

    logger.info(
        "CLI logging configured: file=%s json=%s stdout=%s max_bytes=%s backups=%s",
        log_file,
        log_json,
        log_to_stdout,
        CLI_LOG_MAX_BYTES,
        CLI_LOG_BACKUP_COUNT,
    )
    setattr(logger, "_forge_cli_configured", True)
    setattr(
        logger,
        "_forge_cli_rotation",
        {
            "max_bytes": CLI_LOG_MAX_BYTES,
            "backup_count": CLI_LOG_BACKUP_COUNT,
        },
    )
