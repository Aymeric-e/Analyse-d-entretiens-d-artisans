"""
Logging configuration for BERT Artisan NLP project.

Provides structured logging across all modules with multiple handlers
for console output, detailed file logging, and error tracking.

This module centralizes all logging configuration to ensure consistent
logging format and levels across the project.

Example:
    >>> from src.logger_config import setup_logger
    >>> logger = setup_logger(__name__, level="INFO")
    >>> logger.info("Application started successfully")
    >>> logger.debug("Detailed diagnostic information")
    >>> logger.warning("Potential issue detected")
    >>> logger.error("Serious error occurred", exc_info=True)
"""

import logging
import logging.handlers
from pathlib import Path

# Create logs directory
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Set up logger with file and console handlers.

    Initializes a logger with multiple handlers for different output
    destinations and logging levels. Includes both console and file
    output with automatic file rotation to prevent disk space issues.

    Args:
        name: Logger name (typically __name__ of the module)
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logging.Logger instance ready for use

    Raises:
        ValueError: If level is not a valid logging level

    Example:
        >>> logger = setup_logger(__name__, level="DEBUG")
        >>> logger.info("Model training started")
        >>> logger.debug("Learning rate: 2e-5")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_format)

    # File handler (DEBUG and above) - detailed logs
    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "bert_artisan.log", maxBytes=10_000_000, backupCount=5  # 10 MB
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)

    # Error file handler (ERROR and above) - critical issues
    error_handler = logging.handlers.RotatingFileHandler(LOGS_DIR / "errors.log", maxBytes=5_000_000, backupCount=3)  # 5 MB
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)

    return logger
