"""
Logging utilities for semantic data completion pipeline
Provides structured logging with GPU monitoring capabilities
"""
import logging
import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps


class GPUMemoryMonitor:
    """Monitor GPU memory usage during operations"""

    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get current GPU memory usage information"""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "free_gb": (torch.cuda.get_device_properties(0).total_memory -
                       torch.cuda.memory_allocated()) / 1e9,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }

    @staticmethod
    def log_memory_usage(logger: logging.Logger, operation: str):
        """Log current GPU memory usage for an operation"""
        if torch.cuda.is_available():
            memory_info = GPUMemoryMonitor.get_gpu_memory_info()
            logger.info(f"{operation} - GPU Memory: "
                       f"Allocated: {memory_info['allocated_gb']:.2f}GB, "
                       f"Free: {memory_info['free_gb']:.2f}GB")


def setup_logger(
    name: str = "semantic_pipeline",
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup structured logger with file and console output"""

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """Decorator to log function execution time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Get logger
            log = logger or logging.getLogger("semantic_pipeline")

            # Log GPU memory before operation
            if torch.cuda.is_available():
                GPUMemoryMonitor.log_memory_usage(log, f"Before {func.__name__}")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                log.info(f"{func.__name__} completed in {execution_time:.2f} seconds")

                # Log GPU memory after operation
                if torch.cuda.is_available():
                    GPUMemoryMonitor.log_memory_usage(log, f"After {func.__name__}")

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                log.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {e}")
                raise

        return wrapper
    return decorator


def log_pipeline_start(logger: logging.Logger, config_info: Dict[str, Any]):
    """Log pipeline startup information"""
    logger.info("=" * 60)
    logger.info("SEMANTIC DATA COMPLETION PIPELINE STARTED")
    logger.info("=" * 60)

    # Log configuration
    for key, value in config_info.items():
        logger.info(f"{key}: {value}")

    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        GPUMemoryMonitor.log_memory_usage(logger, "Pipeline Start")
    else:
        logger.info("Running on CPU")

    logger.info("=" * 60)


def log_pipeline_complete(logger: logging.Logger, results: Dict[str, Any]):
    """Log pipeline completion information"""
    logger.info("=" * 60)
    logger.info("SEMANTIC DATA COMPLETION PIPELINE COMPLETED")
    logger.info("=" * 60)

    for key, value in results.items():
        logger.info(f"{key}: {value}")

    if torch.cuda.is_available():
        GPUMemoryMonitor.log_memory_usage(logger, "Pipeline Complete")

    logger.info("=" * 60)


# Create default logger
default_logger = setup_logger(
    log_file=f"src/logs/semantic_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
