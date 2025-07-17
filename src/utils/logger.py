"""
Logging utilities for the POWER-BI-BOT project
"""

import logging
import sys
from pathlib import Path
from config.settings import LOG_LEVEL, LOG_FORMAT

def setup_logger(name: str, log_file: str = None, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with both console and file handlers
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """
    Log comprehensive information about a DataFrame
    
    Args:
        logger: Logger instance
        df: pandas DataFrame
        name: Name to identify the DataFrame in logs
    """
    logger.info(f"{name} Info:")
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"  Columns: {df.columns.tolist()}")
    logger.info(f"  Data types: {df.dtypes.value_counts().to_dict()}")
    logger.info(f"  Missing values: {df.isnull().sum().sum()}")
    logger.info(f"  Duplicate rows: {df.duplicated().sum()}")

def log_processing_step(logger: logging.Logger, step_name: str, start_shape=None, end_shape=None):
    """
    Log processing step information
    
    Args:
        logger: Logger instance
        step_name: Name of the processing step
        start_shape: Shape before processing
        end_shape: Shape after processing
    """
    logger.info(f"Processing Step: {step_name}")
    if start_shape and end_shape:
        logger.info(f"  Shape change: {start_shape} -> {end_shape}")
        rows_change = end_shape[0] - start_shape[0]
        cols_change = end_shape[1] - start_shape[1]
        logger.info(f"  Rows: {rows_change:+d}, Columns: {cols_change:+d}")
