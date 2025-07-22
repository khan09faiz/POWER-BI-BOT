"""
Logging utilities for ONGC Equipment Data Processing Pipeline
"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from functools import wraps
import psutil

class PerformanceLogger:
    """Logger for performance metrics and monitoring"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def record_metric(self, name: str, value: Any, unit: str = None):
        """Record a performance metric"""
        self.metrics[name] = {
            'value': value,
            'unit': unit,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return self.metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        current_time = time.time()
        current_memory = self._get_memory_usage()
        
        # Calculate additional system metrics
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')
        except:
            cpu_percent = 0
            memory_info = None
            disk_usage = None
        
        summary = {
            'total_execution_time': current_time - self.start_time,
            'memory_delta': current_memory - self.start_memory,
            'current_memory_mb': current_memory,
            'metrics_count': len(self.metrics),
            'recorded_metrics': self.metrics,
            'system_metrics': {
                'cpu_percent': cpu_percent,
                'memory_available_gb': memory_info.available / (1024**3) if memory_info else 0,
                'memory_used_percent': memory_info.percent if memory_info else 0,
                'disk_free_gb': disk_usage.free / (1024**3) if disk_usage else 0
            }
        }
        
        # Calculate performance insights
        if self.metrics:
            execution_times = [m['value'] for m in self.metrics.values() 
                             if m.get('unit') == 'seconds' and 'execution_time' in str(m)]
            if execution_times:
                summary['performance_insights'] = {
                    'avg_execution_time': sum(execution_times) / len(execution_times),
                    'max_execution_time': max(execution_times),
                    'min_execution_time': min(execution_times),
                    'total_processing_time': sum(execution_times)
                }
        
        return summary

# Global performance logger
_perf_logger = PerformanceLogger()

def get_performance_logger() -> PerformanceLogger:
    """Get global performance logger"""
    return _perf_logger

def setup_logging(log_level: str = "INFO", logs_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging system"""
    
    # Create logs directory
    Path(logs_dir).mkdir(exist_ok=True)
    
    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main log file
    main_log = Path(logs_dir) / f"pipeline_{timestamp}.log"
    error_log = Path(logs_dir) / f"errors_{timestamp}.log"
    performance_log = Path(logs_dir) / f"performance_{timestamp}.log"
    
    # Create main logger
    logger = logging.getLogger('ongc_pipeline')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Detailed file handler
    file_handler = logging.FileHandler(main_log)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(error_log)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    
    # Performance file handler
    perf_handler = logging.FileHandler(performance_log)
    perf_formatter = logging.Formatter('%(asctime)s - PERFORMANCE - %(message)s')
    perf_handler.setFormatter(perf_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    
    # Create performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.addHandler(perf_handler)
    perf_logger.propagate = False
    
    return logger

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get logger
        logger = logging.getLogger('ongc_pipeline')
        perf_logger = get_performance_logger()
        
        # Record start metrics
        start_time = time.time()
        start_memory = perf_logger._get_memory_usage()
        
        logger.info(f"⏱️ Starting {func.__name__}")
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Record end metrics
            end_time = time.time()
            end_memory = perf_logger._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            # Log performance
            logger.info(
                f"✅ Completed {func.__name__}: "
                f"Time={execution_time:.2f}s, "
                f"Memory Δ={memory_delta:.2f}MB"
            )
            
            # Record metrics
            perf_logger.record_metric(f"{func.__name__}_execution_time", execution_time, "seconds")
            perf_logger.record_metric(f"{func.__name__}_memory_delta", memory_delta, "MB")
            
            return result
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.error(
                f"❌ Failed {func.__name__}: "
                f"Time={execution_time:.2f}s, "
                f"Error={str(e)}"
            )
            
            # Record failure metrics
            perf_logger.record_metric(f"{func.__name__}_failed", True)
            perf_logger.record_metric(f"{func.__name__}_failure_time", execution_time, "seconds")
            
            raise
    
    return wrapper

def log_system_info(logger: logging.Logger):
    """Log system information"""
    try:
        # System info
        import platform
        logger.info(f"🖥️ System: {platform.system()} {platform.release()}")
        logger.info(f"🐍 Python: {platform.python_version()}")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"💾 Total Memory: {memory.total / (1024**3):.1f} GB")
        logger.info(f"💾 Available Memory: {memory.available / (1024**3):.1f} GB")
        
        # CPU info
        logger.info(f"⚡ CPU Cores: {psutil.cpu_count()}")
        
    except Exception as e:
        logger.warning(f"Could not log system info: {e}")

def create_logger(name: str = None) -> logging.Logger:
    """Create a logger instance"""
    if name is None:
        name = 'ongc_pipeline'
    
    return logging.getLogger(name)