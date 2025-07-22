"""
Enhanced Data Preprocessing Module for ONGC Equipment Data Processing Pipeline
Senior Programmer Implementation with Industry Best Practices

This module handles:
1. Loading and concatenating multiple Excel files
2. Column filtering (keeping only specified columns)
3. Data cleaning without imputation
4. Standardized column naming
5. Quality assurance and validation
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
from .config import get_config
from .logger import monitor_performance

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger('ongc_pipeline')

@dataclass
class PreprocessingStats:
    """Statistics from the preprocessing pipeline"""
    total_files_processed: int = 0
    successful_files: int = 0
    failed_files: List[str] = None
    original_total_rows: int = 0
    final_total_rows: int = 0
    columns_found: List[str] = None
    columns_missing: List[str] = None
    duplicates_removed: int = 0
    empty_rows_removed: int = 0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    processing_time_seconds: float = 0.0
    
    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []
        if self.columns_found is None:
            self.columns_found = []
        if self.columns_missing is None:
            self.columns_missing = []

class ONGCDataPreprocessor:
    """
    Advanced data preprocessor for ONGC equipment data
    Implements industry best practices for data pipeline processing
    """
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.stats = PreprocessingStats()
        
        # Validate configuration
        self._validate_config()
        
        logger.info("🔧 ONGC Data Preprocessor initialized")
        logger.info(f"   Target columns: {len(self.config.required_columns)}")
        logger.info(f"   Source directory: {self.config.raw_data_dir}")
    
    def _validate_config(self):
        """Validate configuration parameters"""
        if not self.config.required_columns:
            raise ValueError("No required columns specified in configuration")
        
        raw_dir = Path(self.config.raw_data_dir)
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")
    
    @monitor_performance
    def preprocess_data(self) -> Tuple[pd.DataFrame, PreprocessingStats]:
        """
        Main preprocessing method that orchestrates the entire pipeline
        
        Returns:
            Tuple[pd.DataFrame, PreprocessingStats]: Processed dataframe and statistics
        """
        import time
        start_time = time.time()
        
        logger.info("🚀 Starting ONGC Equipment Data Preprocessing Pipeline")
        logger.info("=" * 70)
        
        try:
            # Step 1: Discover and validate files
            files = self._discover_files()
            self.stats.total_files_processed = len(files)
            
            # Step 2: Load and concatenate all files
            combined_df = self._load_and_concatenate_files(files)
            
            if combined_df is None or combined_df.empty:
                raise ValueError("No data loaded from files")
            
            # Step 3: Filter columns to keep only required ones
            filtered_df = self._filter_required_columns(combined_df)
            
            # Step 4: Clean data (remove duplicates, empty rows)
            cleaned_df = self._clean_data(filtered_df)
            
            # Step 5: Standardize column names
            standardized_df = self._standardize_column_names(cleaned_df)
            
            # Step 6: Add metadata and validate
            final_df = self._add_metadata_and_validate(standardized_df)
            
            # Step 7: Final statistics and optimization
            final_df = self._optimize_final_dataframe(final_df)
            
            # Update processing statistics
            self.stats.processing_time_seconds = time.time() - start_time
            self.stats.final_total_rows = len(final_df)
            
            self._log_final_statistics()
            
            logger.info("✅ Preprocessing pipeline completed successfully!")
            logger.info("=" * 70)
            
            return final_df, self.stats
            
        except Exception as e:
            logger.error(f"❌ Preprocessing pipeline failed: {str(e)}")
            logger.error(f"   Time elapsed: {time.time() - start_time:.2f}s")
            raise
    
    def _discover_files(self) -> List[str]:
        """Discover and validate data files"""
        logger.info("📁 Step 1: Discovering data files")
        
        raw_dir = Path(self.config.raw_data_dir)
        discovered_files = []
        
        # First, try to find exact matches
        for expected_file in self.config.expected_files:
            file_path = raw_dir / expected_file
            if file_path.exists():
                discovered_files.append(expected_file)
                logger.info(f"   ✅ Found: {expected_file}")
            else:
                logger.warning(f"   ⚠️ Missing: {expected_file}")
        
        # If no exact matches and auto-discovery is enabled
        if not discovered_files and self.config.auto_discover_files:
            logger.info("   🔍 Auto-discovering Excel files...")
            excel_files = list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
            
            for file_path in excel_files:
                discovered_files.append(file_path.name)
                logger.info(f"   📊 Auto-discovered: {file_path.name}")
        
        if not discovered_files:
            raise FileNotFoundError("No data files found to process")
        
        logger.info(f"   📈 Total files to process: {len(discovered_files)}")
        return discovered_files
    
    def _load_and_concatenate_files(self, files: List[str]) -> pd.DataFrame:
        """Load all files and concatenate them"""
        logger.info("📊 Step 2: Loading and concatenating files")
        
        dataframes = []
        
        for file_name in files:
            try:
                df = self._load_single_file(file_name)
                if df is not None and not df.empty:
                    # No metadata columns added - keep only original data
                    dataframes.append(df)
                    self.stats.successful_files += 1
                    self.stats.original_total_rows += len(df)
                    
                    logger.info(f"   ✅ Loaded {file_name}: {len(df):,} rows")
                else:
                    self.stats.failed_files.append(file_name)
                    logger.warning(f"   ⚠️ Failed to load or empty: {file_name}")
                    
            except Exception as e:
                self.stats.failed_files.append(file_name)
                logger.error(f"   ❌ Error loading {file_name}: {str(e)}")
        
        if not dataframes:
            raise ValueError("No dataframes loaded successfully")
        
        # Concatenate all dataframes
        logger.info("   🔗 Concatenating all dataframes...")
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        
        # Track memory usage
        self.stats.memory_before_mb = combined_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        logger.info(f"   📊 Combined dataset shape: {combined_df.shape}")
        logger.info(f"   💾 Memory usage: {self.stats.memory_before_mb:.2f} MB")
        
        return combined_df
    
    def _load_single_file(self, file_name: str) -> Optional[pd.DataFrame]:
        """Load a single Excel or CSV file with robust error handling"""
        file_path = Path(self.config.raw_data_dir) / file_name
        
        try:
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                # Load Excel file with optimized settings
                df = pd.read_excel(
                    file_path,
                    dtype=str,  # Load all as strings initially to prevent data loss
                    na_values=['', 'N/A', 'n/a', 'NA', 'null', 'NULL', 'None']
                )
            elif file_path.suffix.lower() == '.csv':
                # Try different encodings for CSV
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            dtype=str,
                            na_values=['', 'N/A', 'n/a', 'NA', 'null', 'NULL', 'None']
                        )
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    logger.error(f"   ❌ Could not decode CSV file: {file_name}")
                    return None
            else:
                logger.error(f"   ❌ Unsupported file format: {file_name}")
                return None
            
            # Basic validation
            if df.empty:
                logger.warning(f"   ⚠️ Empty file: {file_name}")
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"   ❌ Error loading {file_name}: {str(e)}")
            return None
    
    def _extract_period_from_filename(self, file_name: str) -> str:
        """Extract period information from filename"""
        # Remove common prefixes and extensions
        period = file_name.replace('EQUI_', '').replace('.xlsx', '').replace('.xls', '').replace('.csv', '')
        return period.strip()
    
    def _filter_required_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to keep only required columns"""
        logger.info("🔍 Step 3: Filtering required columns")
        
        # Check which required columns are actually available
        available_columns = []
        missing_columns = []
        
        for col in self.config.required_columns:
            if col in df.columns:
                available_columns.append(col)
            else:
                missing_columns.append(col)
        
        # Log column availability
        self.stats.columns_found = available_columns.copy()
        self.stats.columns_missing = missing_columns.copy()
        
        if missing_columns:
            logger.warning(f"   ⚠️ Missing columns: {missing_columns}")
        
        if not available_columns:
            raise ValueError("None of the required columns found in the data")
        
        logger.info(f"   ✅ Found columns: {available_columns}")
        
        # Keep only the required columns (no metadata columns)
        filtered_df = df[available_columns].copy()
        
        logger.info(f"   📊 Filtered shape: {filtered_df.shape}")
        logger.info(f"   🗂️ Columns kept: {len(available_columns)} (only required columns)")
        
        return filtered_df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing duplicates and empty rows"""
        logger.info("🧹 Step 4: Cleaning data")
        
        original_rows = len(df)
        
        # Remove completely empty rows (all values are NaN)
        if self.config.remove_empty_rows:
            # Consider a row empty if all required columns are NaN
            required_cols_in_df = [col for col in self.config.required_columns if col in df.columns]
            
            if required_cols_in_df:
                before_empty = len(df)
                df = df.dropna(how='all', subset=required_cols_in_df)
                empty_removed = before_empty - len(df)
                self.stats.empty_rows_removed = empty_removed
                
                if empty_removed > 0:
                    logger.info(f"   🗑️ Removed {empty_removed:,} completely empty rows")
        
        # Remove exact duplicates
        if self.config.remove_duplicates:
            before_dup = len(df)
            df = df.drop_duplicates()
            duplicates_removed = before_dup - len(df)
            self.stats.duplicates_removed = duplicates_removed
            
            if duplicates_removed > 0:
                logger.info(f"   🗑️ Removed {duplicates_removed:,} duplicate rows")
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        rows_cleaned = original_rows - len(df)
        logger.info(f"   📊 Cleaning summary: {original_rows:,} → {len(df):,} rows ({rows_cleaned:,} removed)")
        
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using configured mappings"""
        logger.info("📝 Step 5: Standardizing column names")
        
        # Apply column mappings
        columns_renamed = 0
        for old_name, new_name in self.config.column_mappings.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                columns_renamed += 1
                logger.info(f"   📝 Renamed: '{old_name}' → '{new_name}'")
        
        logger.info(f"   ✅ Standardized {columns_renamed} column names")
        
        return df
    
    def _add_metadata_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate final dataset without adding metadata columns"""
        logger.info("🔍 Step 6: Data validation (no metadata added)")
        
        # Validate data types and ranges without adding extra columns
        self._validate_data_quality(df)
        
        logger.info(f"   📊 Final dataset shape: {df.shape}")
        logger.info(f"   ✅ Validation completed - only original columns retained")
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame):
        """Validate data quality without modifying the data"""
        logger.info("   🔍 Validating data quality...")
        
        # Check for completely empty critical columns
        critical_columns = ['equipment_number', 'Equipment']  # Both possible names
        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_ratio = null_count / len(df) if len(df) > 0 else 0
                
                if null_ratio > 0.9:
                    logger.warning(f"   ⚠️ Column '{col}' is >90% missing ({null_ratio:.1%})")
                elif null_ratio > 0.5:
                    logger.warning(f"   ⚠️ Column '{col}' is >50% missing ({null_ratio:.1%})")
        
        # Log basic statistics about the data
        logger.info(f"   📊 Final dataset contains {len(df):,} records")
        logger.info(f"   �️ Final dataset contains {len(df.columns)} columns")
    
    def _optimize_final_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final optimizations to the dataframe"""
        logger.info("⚡ Step 7: Final optimization")
        
        if not self.config.memory_optimization:
            return df
        
        # Optimize data types while preserving data integrity
        original_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Convert date columns
        date_columns = ['created_on', 'changed_on']
        for col in date_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.info(f"   📅 Converted {col} to datetime")
                except Exception:
                    logger.warning(f"   ⚠️ Could not convert {col} to datetime")
        
        # Optimize categorical columns
        categorical_candidates = ['object_type', 'manufacturer', 'material']
        for col in categorical_candidates:
            if col in df.columns and df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 1
                if unique_ratio < 0.5:  # Less than 50% unique values
                    df[col] = df[col].astype('category')
                    logger.info(f"   🏷️ Optimized {col} as category")
        
        # Final memory check
        optimized_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
        memory_reduction = ((original_memory - optimized_memory) / original_memory) * 100 if original_memory > 0 else 0
        
        self.stats.memory_after_mb = optimized_memory
        
        logger.info(f"   💾 Memory optimization: {original_memory:.2f}MB → {optimized_memory:.2f}MB ({memory_reduction:.1f}% reduction)")
        
        return df
    
    def _log_final_statistics(self):
        """Log comprehensive processing statistics"""
        logger.info("📈 Processing Statistics Summary")
        logger.info("-" * 50)
        logger.info(f"Files processed: {self.stats.successful_files}/{self.stats.total_files_processed}")
        
        if self.stats.failed_files:
            logger.warning(f"Failed files: {', '.join(self.stats.failed_files)}")
        
        logger.info(f"Original rows: {self.stats.original_total_rows:,}")
        logger.info(f"Final rows: {self.stats.final_total_rows:,}")
        logger.info(f"Duplicates removed: {self.stats.duplicates_removed:,}")
        logger.info(f"Empty rows removed: {self.stats.empty_rows_removed:,}")
        
        if self.stats.columns_missing:
            logger.warning(f"Missing columns: {', '.join(self.stats.columns_missing)}")
        
        logger.info(f"Memory usage: {self.stats.memory_before_mb:.2f}MB → {self.stats.memory_after_mb:.2f}MB")
        logger.info(f"Processing time: {self.stats.processing_time_seconds:.2f}s")
    
    def get_statistics(self) -> PreprocessingStats:
        """Get preprocessing statistics"""
        return self.stats

def create_preprocessor(config=None) -> ONGCDataPreprocessor:
    """Factory function to create a preprocessor instance"""
    return ONGCDataPreprocessor(config)
