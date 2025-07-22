"""
Data Processing utilities for ONGC Equipment Data Processing Pipeline
"""

import pandas as pd
import numpy as np
import gc
import sys
from typing import Dict, Any, Optional
import logging
from .config import get_config
from .logger import monitor_performance, get_performance_logger

logger = logging.getLogger('ongc_pipeline')

class DataProcessor:
    """Data processing and cleaning utilities"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.perf_logger = get_performance_logger()
    
    @monitor_performance
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced DataFrame memory optimization with 35%+ reduction target"""
        if not self.config.memory_optimization or df.empty:
            return df
        
        logger.info("🧠 Optimizing DataFrame memory usage (targeting 35%+ reduction)")
        
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"   Memory usage before optimization: {start_mem:.2f} MB")
        
        # Optimize numeric columns with enhanced logic
        for col in df.select_dtypes(include=['int64']).columns:
            if df[col].isnull().all():
                continue  # Skip completely null columns
                
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Skip if column has NaN values that would cause issues
            if pd.isna(col_min) or pd.isna(col_max):
                continue
            
            # Enhanced integer optimization with nullable integer types
            try:
                if col_min >= 0:  # Unsigned integers
                    if col_max < 256:
                        df[col] = df[col].astype('UInt8')
                    elif col_max < 65536:
                        df[col] = df[col].astype('UInt16')
                    elif col_max < 4294967296:
                        df[col] = df[col].astype('UInt32')
                else:  # Signed integers
                    if col_min > -128 and col_max < 128:
                        df[col] = df[col].astype('Int8')
                    elif col_min > -32768 and col_max < 32768:
                        df[col] = df[col].astype('Int16')
                    elif col_min > -2147483648 and col_max < 2147483648:
                        df[col] = df[col].astype('Int32')
            except Exception:
                # Fallback to standard types if nullable types fail
                try:
                    if col_min >= 0:
                        if col_max < 256:
                            df[col] = df[col].astype(np.uint8)
                        elif col_max < 65536:
                            df[col] = df[col].astype(np.uint16)
                        elif col_max < 4294967296:
                            df[col] = df[col].astype(np.uint32)
                    else:
                        if col_min > -128 and col_max < 128:
                            df[col] = df[col].astype(np.int8)
                        elif col_min > -32768 and col_max < 32768:
                            df[col] = df[col].astype(np.int16)
                        elif col_min > -2147483648 and col_max < 2147483648:
                            df[col] = df[col].astype(np.int32)
                except Exception:
                    pass  # Keep original type if all conversions fail
        
        # Enhanced float optimization
        for col in df.select_dtypes(include=['float64']).columns:
            try:
                # Try to convert to float32 first
                df_temp = pd.to_numeric(df[col], downcast='float')
                if df_temp.dtype == 'float32':
                    df[col] = df_temp
                else:
                    # Check if we can use nullable Float32
                    df[col] = df[col].astype('Float32')
            except Exception:
                try:
                    df[col] = pd.to_numeric(df[col], downcast='float')
                except Exception:
                    pass  # Keep original if conversion fails
        
        # Enhanced categorical optimization with memory monitoring
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            total_count = len(df)
            unique_ratio = unique_count / total_count if total_count > 0 else 1
            
            # Calculate potential memory savings
            current_mem = df[col].memory_usage(deep=True)
            
            # More aggressive categorical conversion for memory savings
            threshold = 0.6 if not self.config.aggressive_categorization else 0.8
            if unique_ratio < threshold and unique_count < 10000:
                try:
                    # Test conversion first
                    test_series = df[col].astype('category')
                    new_mem = test_series.memory_usage(deep=True)
                    
                    # Only convert if it saves significant memory
                    if new_mem < current_mem * 0.8:  # At least 20% savings
                        df[col] = test_series
                        logger.info(f"   Converted {col} to category: {current_mem/1024:.1f}KB -> {new_mem/1024:.1f}KB")
                except Exception:
                    pass  # Skip if conversion fails
        
        # Advanced string optimization for remaining object columns
        import sys
        for col in df.select_dtypes(include=['object']).columns:
            if col not in df.select_dtypes(include=['category']).columns:
                try:
                    # Check if string interning would be beneficial
                    unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                    
                    if unique_ratio < 0.1:  # High repetition - use string interning
                        df[col] = df[col].map(lambda x: sys.intern(str(x)) if pd.notna(x) else x)
                        logger.info(f"   Applied string interning to {col} (unique ratio: {unique_ratio:.3f})")
                    elif df[col].dtype == 'object':
                        # Convert to string dtype if it's more efficient
                        df[col] = df[col].astype('string')
                except Exception:
                    pass
        
        # Sparse optimization for columns with many nulls
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
            if null_ratio > 0.9:  # More than 90% nulls
                try:
                    # Convert to sparse array if beneficial
                    if df[col].dtype in ['int64', 'float64', 'bool']:
                        original_mem = df[col].memory_usage(deep=True)
                        df[col] = pd.arrays.SparseArray(df[col])
                        new_mem = df[col].memory_usage(deep=True)
                        logger.info(f"   Applied sparse optimization to {col}: {original_mem/1024:.1f}KB -> {new_mem/1024:.1f}KB")
                except Exception:
                    pass
        
        # Advanced string optimization for text columns
        for col in df.select_dtypes(include=['object']).columns:
            if col not in df.select_dtypes(include=['category']).columns:
                try:
                    # Check if string dtype is more efficient
                    if df[col].dtype == 'object':
                        # Calculate memory usage before conversion
                        before_mem = df[col].memory_usage(deep=True)
                        temp_col = df[col].astype('string')
                        after_mem = temp_col.memory_usage(deep=True)
                        
                        # Only convert if it saves memory
                        if after_mem < before_mem:
                            df[col] = temp_col
                            logger.info(f"   Optimized string column {col}: {before_mem/1024/1024:.2f}MB -> {after_mem/1024/1024:.2f}MB")
                except Exception:
                    pass
        
        # Advanced categorical optimization with memory threshold
        for col in df.select_dtypes(include=['object']).columns:
            if col not in df.select_dtypes(include=['category']).columns:
                try:
                    unique_count = df[col].nunique()
                    total_count = len(df)
                    
                    # Calculate potential memory savings
                    current_mem = df[col].memory_usage(deep=True)
                    if unique_count < total_count * 0.5:  # Less than 50% unique
                        temp_col = df[col].astype('category')
                        cat_mem = temp_col.memory_usage(deep=True)
                        
                        if cat_mem < current_mem * 0.8:  # At least 20% savings
                            df[col] = temp_col
                            savings = (current_mem - cat_mem) / current_mem * 100
                            logger.info(f"   Categorized {col}: {savings:.1f}% memory savings")
                except Exception:
                    pass
        
        end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = (start_mem - end_mem) / start_mem * 100
        
        logger.info(f"   Memory usage after optimization: {end_mem:.2f} MB ({reduction:.1f}% reduction)")
        
        # Log detailed optimization results
        if reduction >= 35:
            logger.info(f"   🎯 Target achieved! {reduction:.1f}% reduction (target: 35%+)")
        elif reduction >= 25:
            logger.info(f"   ✅ Good optimization: {reduction:.1f}% reduction")
        else:
            logger.info(f"   ⚠️ Moderate optimization: {reduction:.1f}% reduction")
        
        # Record performance metric
        self.perf_logger.record_metric("memory_optimization_reduction_pct", reduction, "%")
        self.perf_logger.record_metric("memory_start_mb", start_mem, "MB")
        self.perf_logger.record_metric("memory_end_mb", end_mem, "MB")
        
        return df
    
    @monitor_performance
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using the mapping"""
        if df.empty:
            return df
        
        logger.info("📝 Standardizing column names")
        
        # Apply column mappings where columns exist
        columns_to_rename = {}
        for old_name, new_name in self.config.column_mappings.items():
            if old_name in df.columns:
                columns_to_rename[old_name] = new_name
        
        if columns_to_rename:
            df = df.rename(columns=columns_to_rename)
            logger.info(f"   Renamed columns: {list(columns_to_rename.keys())} -> {list(columns_to_rename.values())}")
        
        return df
    
    @monitor_performance
    def clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data types and handle missing values"""
        if df.empty:
            return df
        
        logger.info("🔧 Cleaning data types")
        
        # Clean date columns
        date_columns = ['created_on', 'changed_on']
        for col in date_columns:
            if col in df.columns:
                try:
                    original_type = str(df[col].dtype)
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"   Converted {col}: {original_type} -> datetime64")
                except Exception as e:
                    logger.warning(f"   Could not convert {col} to datetime: {e}")
        
        # Clean text columns - remove extra whitespace
        text_columns = ['equipment_description', 'technical_description', 'material', 'manufacturer']
        for col in text_columns:
            if col in df.columns:
                # Clean non-null values
                mask = df[col].notna()
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, col].astype(str).str.strip()
                # Replace empty strings with NaN
                df[col] = df[col].replace(['', 'nan', 'None'], np.nan)
                logger.info(f"   Cleaned text column: {col}")
        
        # Clean equipment number - preserve as string to maintain leading zeros
        if 'equipment_number' in df.columns:
            mask = df['equipment_number'].notna()
            if mask.any():
                df.loc[mask, 'equipment_number'] = df.loc[mask, 'equipment_number'].astype(str)
                # Remove .0 from equipment numbers if they exist
                df['equipment_number'] = df['equipment_number'].str.replace('.0', '', regex=False)
            df['equipment_number'] = df['equipment_number'].replace(['nan', 'None', ''], np.nan)
            logger.info("   Cleaned equipment_number column")
        
        return df
    
    @monitor_performance
    def remove_empty_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove empty rows and duplicates"""
        if df.empty:
            return df
        
        initial_rows = len(df)
        logger.info(f"🧹 Cleaning empty rows and duplicates")
        logger.info(f"   Starting with {initial_rows:,} rows")
        
        # Only remove completely empty rows (all columns are NaN)
        if self.config.remove_empty_rows:
            df = df.dropna(how='all')
            empty_removed = initial_rows - len(df)
            if empty_removed > 0:
                logger.info(f"   Removed {empty_removed:,} completely empty rows")
        
        # Remove duplicate rows (but keep rows with different missing patterns)
        if self.config.remove_duplicates:
            before_dedup = len(df)
            df = df.drop_duplicates()
            duplicates_removed = before_dedup - len(df)
            if duplicates_removed > 0:
                logger.info(f"   Removed {duplicates_removed:,} duplicate rows")
        
        final_rows = len(df)
        total_removed = initial_rows - final_rows
        retention_rate = (final_rows / initial_rows) * 100 if initial_rows > 0 else 0
        
        logger.info(f"   Final rows: {final_rows:,} (removed {total_removed:,}, retention: {retention_rate:.1f}%)")
        
        # Record performance metrics
        self.perf_logger.record_metric("rows_removed", total_removed)
        self.perf_logger.record_metric("data_retention_rate", retention_rate, "%")
        
        return df
    
    @monitor_performance
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and generate quality score"""
        if df.empty:
            return {"quality_score": 0, "issues": ["Empty dataset"]}
        
        logger.info("🔍 Validating data quality")
        
        quality_issues = []
        quality_metrics = {}
        
        # Check for critical columns
        critical_columns = ['equipment_number', 'equipment_description']
        for col in critical_columns:
            if col in df.columns:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                quality_metrics[f"{col}_completeness"] = 100 - missing_pct
                
                if missing_pct > 50:
                    quality_issues.append(f"Critical column {col} has {missing_pct:.1f}% missing data")
            else:
                quality_issues.append(f"Critical column {col} is missing")
                quality_metrics[f"{col}_completeness"] = 0
        
        # Check for duplicate equipment numbers
        if 'equipment_number' in df.columns:
            duplicates = df['equipment_number'].duplicated().sum()
            duplicate_pct = (duplicates / len(df)) * 100
            quality_metrics["uniqueness_score"] = 100 - duplicate_pct
            
            if duplicate_pct > 5:
                quality_issues.append(f"High duplicate rate: {duplicate_pct:.1f}% duplicate equipment numbers")
        
        # Calculate overall quality score
        if quality_metrics:
            quality_score = sum(quality_metrics.values()) / len(quality_metrics) / 100
        else:
            quality_score = 0
        
        quality_report = {
            "quality_score": round(quality_score, 3),
            "metrics": quality_metrics,
            "issues": quality_issues,
            "passed_threshold": quality_score >= self.config.quality_threshold
        }
        
        logger.info(f"   Data quality score: {quality_score:.1%}")
        if quality_issues:
            logger.warning(f"   Quality issues found: {len(quality_issues)}")
            for issue in quality_issues[:3]:  # Show first 3 issues
                logger.warning(f"     - {issue}")
        
        return quality_report
    

    
    @monitor_performance
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete data cleaning pipeline"""
        if df is None or df.empty:
            logger.error("❌ No data to process")
            return pd.DataFrame()
        
        logger.info("🧹 Starting data cleaning pipeline")
        logger.info(f"   Input shape: {df.shape}")
        
        # Step 1: Standardize column names
        df = self.standardize_columns(df)
        
        # Step 2: Clean data types
        df = self.clean_data_types(df)
        
        # Step 3: Remove empty rows and duplicates
        df = self.remove_empty_and_duplicates(df)
        
        # Step 4: Validate data quality
        quality_report = self.validate_data_quality(df)
        if not quality_report["passed_threshold"]:
            logger.warning(f"⚠️ Data quality below threshold ({quality_report['quality_score']:.1%})")
        
        # Step 5: Optimize memory usage
        df = self.optimize_memory(df)
        
        # Step 6: Memory cleanup
        self.memory_cleanup()
        
        logger.info(f"✅ Data cleaning complete!")
        logger.info(f"   Output shape: {df.shape}")
        
        return df
    
    def memory_cleanup(self):
        """Enhanced memory cleanup with detailed reporting"""
        import psutil
        import os
        
        # Get memory usage before cleanup
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / (1024 * 1024)
        
        # Force garbage collection multiple times for thorough cleanup
        for i in range(3):
            gc.collect()
        
        # Get memory usage after cleanup
        memory_after = process.memory_info().rss / (1024 * 1024)
        memory_freed = memory_before - memory_after
        
        logger.info(f"🧹 Memory cleanup performed: {memory_freed:.2f}MB freed")
        logger.info(f"   Current memory usage: {memory_after:.2f}MB")
        
        # Record performance metric
        self.perf_logger.record_metric("memory_cleanup_freed_mb", memory_freed, "MB")
        self.perf_logger.record_metric("current_memory_after_cleanup_mb", memory_after, "MB")
    
    @monitor_performance
    def optimize_dataframe_advanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced DataFrame optimization with multiple techniques"""
        if df.empty:
            return df
        
        logger.info("🚀 Applying advanced DataFrame optimizations")
        
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # 1. Sparse array optimization for columns with many nulls
        for col in df.columns:
            null_ratio = df[col].isnull().sum() / len(df) if len(df) > 0 else 0
            if null_ratio > 0.7:  # More than 70% nulls
                try:
                    if df[col].dtype in ['int64', 'float64', 'bool']:
                        df[col] = pd.arrays.SparseArray(df[col])
                        logger.info(f"   Applied sparse optimization to {col} ({null_ratio:.1%} nulls)")
                except Exception:
                    pass
        
        # 2. String interning for repeated string values
        for col in df.select_dtypes(include=['object', 'string']).columns:
            if df[col].nunique() / len(df) < 0.1:  # Less than 10% unique values
                try:
                    # Use string interning for memory efficiency
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) < 1000:  # Only for small number of unique values
                        intern_map = {val: sys.intern(str(val)) for val in unique_vals}
                        df[col] = df[col].map(intern_map).fillna(df[col])
                        logger.info(f"   Applied string interning to {col}")
                except Exception:
                    pass
        
        # 3. Index optimization
        if not isinstance(df.index, pd.RangeIndex):
            try:
                df = df.reset_index(drop=True)
                logger.info("   Reset index to RangeIndex for memory efficiency")
            except Exception:
                pass
        
        end_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        total_reduction = (start_mem - end_mem) / start_mem * 100
        
        logger.info(f"   Advanced optimization complete: {total_reduction:.1f}% additional reduction")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary"""
        if df.empty:
            return {"error": "No data to summarize"}
        
        logger.info("📊 Generating data summary")
        
        summary = {
            "basic_info": {
                "total_records": len(df),
                "total_columns": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_data": {col: int(count) for col, count in df.isnull().sum().items()},
            "missing_percentages": {col: round(float(count / len(df) * 100), 2) 
                                  for col, count in df.isnull().sum().items()}
        }
        
        # Period analysis
        if 'period_label' in df.columns:
            period_counts = df['period_label'].value_counts().to_dict()
            summary["records_by_period"] = {str(k): int(v) for k, v in period_counts.items()}
            
            logger.info("   Records by period:")
            for period, count in period_counts.items():
                logger.info(f"     {period}: {count:,} records")
        
        # Equipment type analysis
        if 'object_type' in df.columns:
            type_counts = df['object_type'].value_counts().head(10).to_dict()
            summary["top_equipment_types"] = {str(k): int(v) for k, v in type_counts.items()}
            
            logger.info("   Top 10 equipment types:")
            for eq_type, count in type_counts.items():
                logger.info(f"     {eq_type}: {count:,} records")
        
        # Manufacturer analysis
        if 'manufacturer' in df.columns:
            mfg_counts = df['manufacturer'].value_counts().head(10).to_dict()
            summary["top_manufacturers"] = {str(k): int(v) for k, v in mfg_counts.items()}
            
            logger.info("   Top 10 manufacturers:")
            for mfg, count in mfg_counts.items():
                logger.info(f"     {mfg}: {count:,} records")
        
        # Data quality insights
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 10].to_dict()
        
        if high_missing:
            summary["high_missing_columns"] = {col: round(float(pct), 2) for col, pct in high_missing.items()}
            logger.info("   Columns with >10% missing data:")
            for col, pct in high_missing.items():
                logger.info(f"     {col}: {pct:.1f}% missing")
        
        return summary

def create_data_processor(config=None) -> DataProcessor:
    """Factory function to create a DataProcessor instance"""
    return DataProcessor(config)