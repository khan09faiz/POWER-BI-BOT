"""
Simple logging utilities for the pipeline
Clean, minimal logging without over-engineering
"""
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "ongc_pipeline") -> logging.Logger:
    """Setup simple logger with file and console output"""

    # Create logs directory
    Path("src/logs").mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    log_file = f"src/logs/preprocessing.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_processing_start(logger, message: str):
    """Log processing start with banner"""
    logger.info("=" * 60)
    logger.info(f"  {message}")
    logger.info("=" * 60)


def log_step_complete(logger, step_name: str, details: str = ""):
    """Log step completion"""
    if details:
        logger.info(f"✅ {step_name} - {details}")
    else:
        logger.info(f"✅ {step_name}")


def log_error(logger, step_name: str, error: str):
    """Log error with step context"""
    logger.error(f"❌ {step_name} failed: {error}")


def format_number(num: int) -> str:
    """Format large numbers with commas"""
    return f"{num:,}"


# Global logger instance
logger = setup_logger()
