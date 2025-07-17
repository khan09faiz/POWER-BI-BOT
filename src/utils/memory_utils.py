"""
Memory optimization utilities for handling large datasets
Enhanced with Principal Data Scientist performance optimization strategies
"""

import pandas as pd
import numpy as np
import psutil
import time
import gc
from typing import Dict, Any, Union, List
from functools import wraps
from config.settings import OPTIMIZE_DTYPES, LOW_MEMORY

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

class PerformanceOptimizer:
    """
    Enterprise-grade performance optimization strategies integrated with existing codebase
    - Memory-efficient data processing
    - Vectorized operations over loops
    - Optimal file format selection
    - Real-time monitoring
    """
    
    def __init__(self):
        self.performance_metrics = {}
        self.optimization_history = []
    
    def monitor_performance(self, operation_name: str = None):
        """
        Performance monitoring decorator/context manager
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                metrics = {
                    'function': operation_name or func.__name__,
                    'execution_time': end_time - start_time,
                    'memory_delta': end_memory - start_memory,
                    'peak_memory': end_memory
                }
                
                self.performance_metrics[operation_name or func.__name__] = metrics
                print(f"⚡ {operation_name or func.__name__}: {metrics['execution_time']:.2f}s, {metrics['memory_delta']:.2f}MB delta")
                
                return result
            return wrapper
        return decorator
    
    def vectorized_operation(self, df: pd.DataFrame, operation: str, columns: List[str] = None) -> pd.DataFrame:
        """
        Apply vectorized operations instead of loops for better performance
        """
        start_time = time.time()
        
        if operation == 'standardize' and columns:
            for col in columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        elif operation == 'normalize' and columns:
            for col in columns:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
        
        execution_time = time.time() - start_time
        print(f"🚀 Vectorized {operation}: {execution_time:.2f}s")
        
        return df
    
    def optimal_file_format_selection(self, df: pd.DataFrame, base_path: str) -> str:
        """
        Select optimal file format based on data characteristics
        """
        row_count = len(df)
        col_count = len(df.columns)
        
        # For large datasets, prefer Parquet
        if PYARROW_AVAILABLE and (row_count > 100000 or col_count > 50):
            file_path = f"{base_path}.parquet"
            df.to_parquet(file_path, compression='snappy', index=False)
            print(f"📁 Saved as Parquet (optimized for large data): {file_path}")
            return file_path
        
        # For smaller datasets, CSV is fine
        else:
            file_path = f"{base_path}.csv"
            df.to_csv(file_path, index=False)
            print(f"📁 Saved as CSV: {file_path}")
            return file_path

# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()

def get_memory_usage() -> Dict[str, Any]:
    """
    Enhanced memory usage reporting with process-level details
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    return {
        'process_memory_mb': memory_info.rss / 1024 / 1024,
        'process_memory_percent': process.memory_percent(),
        'system_memory_percent': system_memory.percent,
        'available_memory_mb': system_memory.available / 1024 / 1024,
        'total_memory_mb': system_memory.total / 1024 / 1024
    }

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame data types to reduce memory usage
    ENHANCED: Better handling of mixed data types and errors with performance monitoring
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized data types
    """
    if not OPTIMIZE_DTYPES:
        return df
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    optimized_df = df.copy()
    
    # Integer optimization with enhanced range checking
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        try:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if pd.isna(col_min) or pd.isna(col_max):
                continue
                
            if col_min >= -128 and col_max <= 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
        except (ValueError, OverflowError, TypeError):
            continue
    
    # Float optimization with precision checking
    float_columns = df.select_dtypes(include=['float64']).columns
    for col in float_columns:
        try:
            if df[col].isnull().all():
                continue
                
            original_values = df[col].dropna()
            if len(original_values) > 0:
                abs_max = np.abs(original_values).max()
                if abs_max < 3.4e38:  # float32 range
                    float32_values = original_values.astype('float32')
                    
                    # Check for precision loss
                    non_zero_mask = original_values != 0
                    if non_zero_mask.any():
                        relative_error = np.abs(
                            (original_values[non_zero_mask] - float32_values[non_zero_mask]) / 
                            original_values[non_zero_mask]
                        ).max()
                        
                        if relative_error < 1e-6:  # Acceptable precision loss
                            optimized_df[col] = optimized_df[col].astype('float32')
                    else:
                        optimized_df[col] = optimized_df[col].astype('float32')
        except (ValueError, TypeError):
            continue
    
    # Enhanced categorical optimization
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        try:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            
            # Convert to category if less than 50% unique values and reasonable cardinality
            if unique_count / total_count < 0.5 and unique_count > 1 and unique_count < 1000:
                optimized_df[col] = optimized_df[col].astype('category')
        except (ValueError, TypeError):
            continue
    
    new_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
    reduction = (1 - new_memory/original_memory) * 100 if original_memory > 0 else 0
    
    # Record performance metrics
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    performance_optimizer.performance_metrics['optimize_dtypes'] = {
        'function': 'optimize_dtypes',
        'execution_time': end_time - start_time,
        'memory_delta': end_memory - start_memory,
        'peak_memory': end_memory,
        'data_memory_reduction': reduction
    }
    
    print(f"🔧 Memory optimization: {original_memory:.2f} MB -> {new_memory:.2f} MB ({reduction:.1f}% reduction)")
    
    return optimized_df

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive memory reduction for DataFrames
    ENHANCED: Integrated with performance monitoring
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    start_time = time.time()
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            # Convert to category if beneficial
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    execution_time = time.time() - start_time
    
    # Record performance metrics
    performance_optimizer.performance_metrics['reduce_memory_usage'] = {
        'function': 'reduce_memory_usage',
        'execution_time': execution_time,
        'memory_reduction_mb': start_mem - end_mem,
        'memory_reduction_percent': 100 * (start_mem - end_mem) / start_mem if start_mem > 0 else 0
    }
    
    print(f'🗂️ Memory usage after optimization: {end_mem:.2f} MB')
    print(f'📉 Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

@performance_optimizer.monitor_performance("chunk_processing")
def chunk_processor(file_path: str, chunk_size: int = 10000, **kwargs):
    """
    Process large files in chunks to manage memory
    ENHANCED: Integrated with performance monitoring
    
    Args:
        file_path: Path to the file
        chunk_size: Size of each chunk
        **kwargs: Additional arguments for pandas read functions
        
    Yields:
        DataFrame chunks
    """
    if file_path.endswith('.csv'):
        reader = pd.read_csv(file_path, chunksize=chunk_size, **kwargs)
        for chunk in reader:
            yield chunk
    
    elif file_path.endswith('.parquet') and PYARROW_AVAILABLE:
        # Efficient Parquet reading
        table = pq.read_table(file_path)
        df = table.to_pandas()
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i+chunk_size]
    
    elif file_path.endswith(('.xlsx', '.xls')):
        # For Excel files, read entire file first (pandas limitation)
        df = pd.read_excel(file_path, **kwargs)
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i+chunk_size]
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def memory_cleanup():
    """
    Force garbage collection to free memory
    ENHANCED: More aggressive cleanup
    """
    gc.collect()
    # Run multiple times for better cleanup
    for _ in range(3):
        gc.collect()
    
    print("🧹 Memory cleanup completed")

def optimal_dtype_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligent dtype inference for better memory efficiency
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            if not numeric_series.isna().all():
                df[col] = numeric_series
            
            # Try to convert to datetime
            try:
                datetime_series = pd.to_datetime(df[col], errors='coerce')
                if not datetime_series.isna().all():
                    df[col] = datetime_series
            except:
                pass
    
    return df

def batch_processing_strategy(data_list: List[pd.DataFrame], batch_size: int = 5) -> pd.DataFrame:
    """
    Process multiple DataFrames in batches to manage memory
    """
    results = []
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        
        # Process batch
        batch_result = pd.concat(batch, ignore_index=True)
        batch_result = optimize_dtypes(batch_result)
        
        results.append(batch_result)
        
        # Clean up memory after each batch
        memory_cleanup()
    
    final_result = pd.concat(results, ignore_index=True)
    print(f"📊 Processed {len(data_list)} DataFrames in {len(results)} batches")
    
    return final_result

def get_performance_report() -> Dict[str, Any]:
    """
    Generate comprehensive performance report
    """
    return {
        'performance_metrics': performance_optimizer.performance_metrics,
        'current_memory': get_memory_usage(),
        'total_operations': len(performance_optimizer.performance_metrics)
    }
