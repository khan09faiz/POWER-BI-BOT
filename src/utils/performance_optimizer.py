"""
Principal Data Scientist approach to performance optimization
Focus: Scalable data processing for production environments
"""

import pandas as pd
import numpy as np
import psutil
import time
import gc
from functools import wraps
from typing import Union, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

class PerformanceOptimizer:
    """
    Enterprise-grade performance optimization strategies
    - Memory-efficient data processing
    - Vectorized operations over loops
    - Optimal file format selection
    - Real-time monitoring
    """
    
    def __init__(self):
        self.performance_metrics = {}
        self.optimization_history = []
        
    def monitor_performance(self, func):
        """
        DO: Monitor memory and time for every operation
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Pre-execution metrics
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Post-execution metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Store metrics
            metrics = {
                'function': func.__name__,
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'peak_memory': end_memory
            }
            
            self.performance_metrics[func.__name__] = metrics
            
            print(f"⚡ {func.__name__}:")
            print(f"   ⏱️  Time: {metrics['execution_time']:.2f}s")
            print(f"   🧠 Memory: {metrics['memory_delta']:.2f}MB delta, {metrics['peak_memory']:.2f}MB peak")
            
            return result
        
        return wrapper
    
    def optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DO: Aggressive but safe data type optimization
        """
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        print("🔧 OPTIMIZING DATA TYPES FOR MEMORY EFFICIENCY")
        print("=" * 50)
        
        optimized_df = df.copy()
        memory_before = df.memory_usage(deep=True).sum() / 1024**2
        
        # Integer optimization
        int_columns = df.select_dtypes(include=['int64']).columns
        for col in int_columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            if col_min >= -128 and col_max <= 127:
                optimized_df[col] = optimized_df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                optimized_df[col] = optimized_df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                optimized_df[col] = optimized_df[col].astype('int32')
        
        # Float optimization
        float_columns = df.select_dtypes(include=['float64']).columns
        for col in float_columns:
            if df[col].isnull().all():
                continue
                
            # Check if float32 precision is sufficient
            original_values = df[col].dropna()
            if len(original_values) > 0:
                # Check range for float32 compatibility
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
        
        # Categorical optimization for object columns
        object_columns = df.select_dtypes(include=['object']).columns
        for col in object_columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            
            # Convert to category if less than 50% unique values
            if unique_count / total_count < 0.5 and unique_count > 1:
                optimized_df[col] = optimized_df[col].astype('category')
        
        memory_after = optimized_df.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = (memory_before - memory_after) / memory_before * 100
        
        print(f"📊 Memory Optimization Results:")
        print(f"   Before: {memory_before:.2f} MB")
        print(f"   After: {memory_after:.2f} MB")
        print(f"   Reduction: {memory_reduction:.1f}%")
        
        # Record performance metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        self.performance_metrics['optimize_data_types'] = {
            'function': 'optimize_data_types',
            'execution_time': end_time - start_time,
            'memory_delta': end_memory - start_memory,
            'peak_memory': end_memory
        }
        
        return optimized_df
    
    def vectorized_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DO: Use vectorized operations instead of loops
        """
        print("⚡ VECTORIZED FEATURE ENGINEERING")
        print("=" * 50)
        
        engineered_df = df.copy()
        
        # ✅ VECTORIZED: Equipment age calculation
        if 'created_on' in df.columns:
            current_date = pd.Timestamp.now()
            # Convert to datetime if not already
            created_on = pd.to_datetime(df['created_on'], errors='coerce')
            engineered_df['equipment_age_days'] = (current_date - created_on).dt.days
            
            # Vectorized age categories
            engineered_df['age_category'] = pd.cut(
                engineered_df['equipment_age_days'],
                bins=[0, 365, 1825, 3650, float('inf')],
                labels=['New', 'Young', 'Mature', 'Old']
            )
        
        # ✅ VECTORIZED: Value-based classifications
        if 'acquisition_value' in df.columns:
            # Handle missing values first
            values_filled = df['acquisition_value'].fillna(df['acquisition_value'].median())
            
            # Vectorized quantile-based categorization
            try:
                value_quantiles = values_filled.quantile([0.25, 0.5, 0.75])
                
                engineered_df['value_tier'] = pd.cut(
                    values_filled,
                    bins=[0, value_quantiles[0.25], value_quantiles[0.5], 
                          value_quantiles[0.75], float('inf')],
                    labels=['Budget', 'Standard', 'Premium', 'Critical']
                )
            except Exception:
                # Fallback if quantile calculation fails
                engineered_df['value_tier'] = 'Standard'
            
            # Vectorized value per weight ratio
            if 'weight' in df.columns:
                weight_filled = df['weight'].fillna(1)  # Avoid division by zero
                engineered_df['value_density'] = np.where(
                    weight_filled > 0,
                    values_filled / weight_filled,
                    0
                )
        
        # ✅ VECTORIZED: Missing data indicators
        missing_columns = ['manufacturer_of_asset', 'model_number', 'inventory_number']
        available_columns = [col for col in missing_columns if col in df.columns]
        
        if available_columns:
            # Vectorized missing data count
            engineered_df['missing_critical_info'] = (
                df[available_columns].isnull().sum(axis=1)
            )
            
            # Vectorized completeness score
            engineered_df['data_completeness'] = (
                1 - (engineered_df['missing_critical_info'] / len(available_columns))
            ) * 100
        
        # ✅ VECTORIZED: Manufacturer grouping
        if 'manufacturer_of_asset' in df.columns:
            # Vectorized manufacturer frequency encoding
            manufacturer_counts = df['manufacturer_of_asset'].value_counts()
            
            engineered_df['manufacturer_frequency'] = df['manufacturer_of_asset'].map(
                manufacturer_counts
            ).fillna(0)
            
            # Vectorized manufacturer tier based on frequency
            engineered_df['manufacturer_tier'] = pd.cut(
                engineered_df['manufacturer_frequency'],
                bins=[0, 5, 20, 100, float('inf')],
                labels=['Rare', 'Uncommon', 'Common', 'Dominant']
            )
        
        # ✅ VECTORIZED: Equipment type analysis
        if 'objecttype' in df.columns:
            # One-hot encode equipment types (vectorized)
            equipment_types = pd.get_dummies(df['objecttype'], prefix='equipment_type')
            engineered_df = pd.concat([engineered_df, equipment_types], axis=1)
        
        # ✅ VECTORIZED: Date-based features
        date_columns = df.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if col != 'created_on':  # Already processed
                col_name = col.replace('_', '').replace(' ', '')
                engineered_df[f'{col_name}_year'] = pd.to_datetime(df[col]).dt.year
                engineered_df[f'{col_name}_month'] = pd.to_datetime(df[col]).dt.month
                engineered_df[f'{col_name}_quarter'] = pd.to_datetime(df[col]).dt.quarter
        
        new_features = len(engineered_df.columns) - len(df.columns)
        print(f"✅ Created {new_features} new features using vectorized operations")
        
        return engineered_df
    
    def efficient_file_operations(self, df: pd.DataFrame, file_path: str, 
                                 operation: str = 'save') -> Union[pd.DataFrame, None]:
        """
        DO: Use optimal file formats for different use cases
        """
        print(f"💾 EFFICIENT FILE OPERATIONS - {operation.upper()}")
        print("=" * 50)
        
        if operation == 'save':
            # Save in multiple formats for different use cases
            base_path = file_path.rsplit('.', 1)[0]
            
            # 1. Parquet for analytics (best compression and speed)
            if PYARROW_AVAILABLE:
                parquet_path = f"{base_path}.parquet"
                start_time = time.time()
                try:
                    df.to_parquet(
                        parquet_path,
                        engine='pyarrow',
                        compression='snappy',
                        index=False
                    )
                    parquet_time = time.time() - start_time
                    parquet_size = self.get_file_size(parquet_path)
                    print(f"✅ Parquet saved: {parquet_time:.2f}s, {parquet_size:.2f}MB")
                except Exception as e:
                    print(f"⚠️ Parquet save failed: {str(e)}")
                    parquet_time = 0
                    parquet_size = 0
            else:
                print("⚠️ PyArrow not available, skipping Parquet format")
                parquet_time = 0
                parquet_size = 0
            
            # 2. CSV for Power BI (chunked for large files)
            csv_path = f"{base_path}.csv"
            start_time = time.time()
            
            try:
                if len(df) > 50000:  # Chunk large files
                    chunk_size = 50000
                    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
                    
                    csv_size = 0
                    for i, chunk in enumerate(chunks):
                        chunk_path = f"{base_path}_chunk_{i+1}.csv"
                        chunk.to_csv(chunk_path, index=False)
                        csv_size += self.get_file_size(chunk_path)
                    
                    csv_time = time.time() - start_time
                    print(f"✅ CSV chunks saved: {len(chunks)} files, {csv_time:.2f}s, {csv_size:.2f}MB total")
                else:
                    df.to_csv(csv_path, index=False)
                    csv_time = time.time() - start_time
                    csv_size = self.get_file_size(csv_path)
                    print(f"✅ CSV saved: {csv_time:.2f}s, {csv_size:.2f}MB")
            except Exception as e:
                print(f"⚠️ CSV save failed: {str(e)}")
                csv_time = 0
                csv_size = 0
            
            # 3. JSON for AI agent (optimized structure)
            json_path = f"{base_path}_ai_ready.json"
            start_time = time.time()
            
            try:
                # Create AI-optimized JSON structure
                ai_ready_data = {
                    'metadata': {
                        'total_records': len(df),
                        'columns': list(df.columns),
                        'data_types': df.dtypes.astype(str).to_dict(),
                        'created_at': pd.Timestamp.now().isoformat()
                    },
                    'summary_statistics': df.describe().to_dict(),
                    'sample_data': df.head(100).to_dict('records')  # Sample for AI context
                }
                
                import json
                with open(json_path, 'w') as f:
                    json.dump(ai_ready_data, f, indent=2, default=str)
                
                json_time = time.time() - start_time
                json_size = self.get_file_size(json_path)
                print(f"✅ JSON (AI) saved: {json_time:.2f}s, {json_size:.2f}MB")
            except Exception as e:
                print(f"⚠️ JSON save failed: {str(e)}")
                json_time = 0
                json_size = 0
            
            # Performance summary
            print(f"\n📊 File Format Performance Summary:")
            if parquet_size > 0:
                print(f"   Parquet: {parquet_time:.2f}s, {parquet_size:.2f}MB")
            if csv_size > 0:
                print(f"   CSV: {csv_time:.2f}s, {csv_size:.2f}MB")
            if json_size > 0:
                print(f"   JSON (AI): {json_time:.2f}s, {json_size:.2f}MB")
            
            return None
            
        elif operation == 'load':
            file_ext = file_path.split('.')[-1].lower()
            
            try:
                if file_ext == 'parquet' and PYARROW_AVAILABLE:
                    return pd.read_parquet(file_path, engine='pyarrow')
                elif file_ext == 'csv':
                    # Optimized CSV reading
                    return pd.read_csv(
                        file_path,
                        low_memory=False
                    )
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
            except Exception as e:
                print(f"⚠️ File load failed: {str(e)}")
                return None
    
    def memory_efficient_processing(self, df: pd.DataFrame, 
                                   chunk_size: int = 10000) -> pd.DataFrame:
        """
        DO: Process large datasets in memory-efficient chunks
        """
        print(f"🧠 MEMORY-EFFICIENT CHUNK PROCESSING")
        print("=" * 50)
        
        if len(df) <= chunk_size:
            print(f"Dataset size ({len(df)}) is within chunk limit. Processing normally.")
            return self.vectorized_feature_engineering(df)
        
        print(f"Processing {len(df)} records in chunks of {chunk_size}")
        
        processed_chunks = []
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(df), chunk_size):
            chunk_num = (i // chunk_size) + 1
            print(f"Processing chunk {chunk_num}/{total_chunks}")
            
            chunk = df.iloc[i:i+chunk_size].copy()
            
            # Process chunk
            processed_chunk = self.vectorized_feature_engineering(chunk)
            processed_chunks.append(processed_chunk)
            
            # Force garbage collection after each chunk
            del chunk
            gc.collect()
        
        # Combine all processed chunks
        print("Combining processed chunks...")
        result = pd.concat(processed_chunks, ignore_index=True)
        
        # Final garbage collection
        del processed_chunks
        gc.collect()
        
        print(f"✅ Successfully processed all {total_chunks} chunks")
        return result
    
    def get_file_size(self, file_path: str) -> float:
        """Helper method to get file size in MB"""
        try:
            import os
            return os.path.getsize(file_path) / 1024 / 1024
        except:
            return 0.0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        DO: Generate comprehensive performance report
        """
        print("📊 PERFORMANCE OPTIMIZATION REPORT")
        print("=" * 50)
        
        if not self.performance_metrics:
            print("No performance metrics available. Run optimization methods first.")
            return {}
        
        total_time = sum(metric['execution_time'] for metric in self.performance_metrics.values())
        total_memory_delta = sum(metric['memory_delta'] for metric in self.performance_metrics.values())
        
        report = {
            'summary': {
                'total_execution_time': total_time,
                'total_memory_delta': total_memory_delta,
                'number_of_operations': len(self.performance_metrics)
            },
            'detailed_metrics': self.performance_metrics,
            'recommendations': self.generate_optimization_recommendations()
        }
        
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Total Memory Delta: {total_memory_delta:.2f}MB")
        print(f"Operations Monitored: {len(self.performance_metrics)}")
        
        return report
    
    def generate_optimization_recommendations(self) -> List[str]:
        """
        DO: Provide actionable optimization recommendations
        """
        recommendations = []
        
        # Analyze performance metrics
        if self.performance_metrics:
            slowest_operation = max(
                self.performance_metrics.items(),
                key=lambda x: x[1]['execution_time'],
                default=(None, {'execution_time': 0})
            )
            
            highest_memory = max(
                self.performance_metrics.items(),
                key=lambda x: x[1]['memory_delta'],
                default=(None, {'memory_delta': 0})
            )
            
            if slowest_operation[1]['execution_time'] > 10:
                recommendations.append(
                    f"Consider optimizing {slowest_operation[0]} - took {slowest_operation[1]['execution_time']:.2f}s"
                )
            
            if highest_memory[1]['memory_delta'] > 500:  # > 500MB
                recommendations.append(
                    f"High memory usage in {highest_memory[0]} - {highest_memory[1]['memory_delta']:.2f}MB delta"
                )
        
        # General recommendations
        recommendations.extend([
            "Use Parquet format for analytical workloads",
            "Implement chunked processing for datasets > 100K records",
            "Monitor memory usage in production environments",
            "Use categorical data types for low-cardinality string columns",
            "Implement data validation checkpoints for quality assurance"
        ])
        
        return recommendations
    
    def benchmark_operations(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Benchmark different operation types for performance comparison
        """
        print("🏁 PERFORMANCE BENCHMARKING")
        print("=" * 50)
        
        benchmarks = {}
        
        # Test vectorized vs loop operations (small sample)
        test_df = df.head(1000).copy()
        
        # Vectorized operation benchmark
        start_time = time.time()
        if 'acquisition_value' in test_df.columns:
            test_df['value_log'] = np.log1p(test_df['acquisition_value'].fillna(0))
        else:
            test_df['dummy_calc'] = test_df.index * 2
        vectorized_time = time.time() - start_time
        benchmarks['vectorized_operation'] = vectorized_time
        
        # Memory usage benchmark
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        optimized_df = self.optimize_data_types(test_df)
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        benchmarks['memory_optimization'] = end_memory - start_memory
        
        # File I/O benchmark
        if len(df) > 0:
            start_time = time.time()
            self.efficient_file_operations(
                test_df, 
                'temp_benchmark_file', 
                operation='save'
            )
            file_io_time = time.time() - start_time
            benchmarks['file_io_operation'] = file_io_time
        
        print(f"Benchmarking Results:")
        for operation, timing in benchmarks.items():
            print(f"  {operation}: {timing:.4f}s")
        
        return benchmarks
    
    def create_optimization_summary(self, original_df: pd.DataFrame, 
                                   optimized_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create comprehensive optimization summary
        """
        original_memory = original_df.memory_usage(deep=True).sum() / 1024**2
        optimized_memory = optimized_df.memory_usage(deep=True).sum() / 1024**2
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        original_features = len(original_df.columns)
        optimized_features = len(optimized_df.columns)
        feature_change = optimized_features - original_features
        
        summary = {
            'memory_optimization': {
                'original_memory_mb': original_memory,
                'optimized_memory_mb': optimized_memory,
                'memory_reduction_percentage': memory_reduction
            },
            'feature_engineering': {
                'original_features': original_features,
                'optimized_features': optimized_features,
                'new_features_added': max(0, feature_change)
            },
            'performance_metrics': self.performance_metrics,
            'recommendations': self.generate_optimization_recommendations()
        }
        
        return summary
