"""Logging defaults for training and evaluation CLIs."""

from __future__ import annotations

import logging
import warnings

DEFAULT_LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s: %(message)s"
LIGER_TOKEN_ACCURACY_WARNING = (
    r"liger-kernel did not return token_accuracy when requested\..*"
)
NOISY_DEPENDENCY_LOGGERS = (
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
)


def configure_cli_logging(level: int = logging.INFO) -> None:
    """Keep package logs visible while suppressing noisy dependency INFO logs."""
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
    for logger_name in NOISY_DEPENDENCY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=LIGER_TOKEN_ACCURACY_WARNING,
        category=UserWarning,
    )
