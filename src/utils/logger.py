"""Structured logging for ReconStruct pipeline."""

import logging
import sys
from pathlib import Path

try:
    from config.settings import LOG_LEVEL, LOG_FILE
except ImportError:
    LOG_LEVEL = "INFO"
    LOG_FILE = Path("logs/reconstruct.log")


def get_logger(name: str = "reconstruct") -> logging.Logger:
    """Return configured logger with file and console handlers."""
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    # File
    if LOG_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log
