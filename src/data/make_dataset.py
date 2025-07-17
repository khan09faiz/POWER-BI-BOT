import pandas as pd
import glob
import logging
import re
from functools import reduce
from pathlib import Path
import numpy as np
from typing import List, Dict, Any, Optional

# Import our custom utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import RAW_FILES, CHUNK_SIZE, REFERENCE_DATE
from config.column_mappings import COLUMN_MAPPINGS, DATE_COLUMNS
from src.utils.logger import setup_logger, log_dataframe_info, log_processing_step
from src.utils.memory_utils import optimize_dtypes, chunk_processor, memory_cleanup
from src.utils.helpers import standardize_column_names, extract_time_period, validate_data_integrity

# Setup logger
logger = setup_logger(__name__, "logs/data_processing.log")

def load_and_concatenate_data(raw_data_path: str) -> pd.DataFrame:
    """
    IMPROVED: Load all Excel files and CONCATENATE them (time-series approach)
    This is the correct approach for time-series equipment data across multiple periods.
    """
    logger.info(f"Starting time-series data loading from {raw_data_path}")
    
    # Use predefined file list for better control
    raw_path = Path(raw_data_path)
    all_files = [raw_path / filename for filename in RAW_FILES if (raw_path / filename).exists()]
    
    if not all_files:
        raise FileNotFoundError(f"No data files found in {raw_data_path}")
    
    logger.info(f"Found {len(all_files)} files to process: {[f.name for f in all_files]}")

    df_list = []
    total_rows = 0
    
    for file_path in all_files:
        try:
            logger.info(f"Processing file: {file_path.name}")
            
            # Extract time period information from filename
            period_info = extract_time_period(file_path.name)
            
            # Load the file
            if file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                except UnicodeDecodeError:
                    logger.warning(f"UTF-8 failed for {file_path}, trying latin1.")
                    df = pd.read_csv(file_path, low_memory=False, encoding='latin1')
            else:
                df = pd.read_excel(file_path)
            
            initial_shape = df.shape
            logger.info(f"Loaded {file_path.name}: {initial_shape}")
            
            # Standardize column names immediately
            df = standardize_column_names(df)
            
            # Add metadata columns for time-series analysis
            if period_info:
                df['data_source_file'] = file_path.name
                df['period_start'] = period_info.get('start_date')
                df['period_end'] = period_info.get('end_date')
                df['period_label'] = period_info.get('period_label', 'unknown')
            else:
                df['data_source_file'] = file_path.name
                df['period_label'] = 'unknown'
            
            # Memory optimization
            df = optimize_dtypes(df)
            
            df_list.append(df)
            total_rows += len(df)
            logger.info(f"Successfully processed {file_path.name}: {df.shape}")
            
        except Exception as e:
            logger.error(f"Could not load {file_path}: {e}")
            continue

    if not df_list:
        raise ValueError("No dataframes were successfully loaded.")
    
    # CONCATENATE (not merge) - this preserves all records across time periods
    logger.info("Concatenating all dataframes vertically (time-series approach)")
    concatenated_df = pd.concat(df_list, ignore_index=True, sort=False)
    
    logger.info(f"Successfully concatenated all files. Final shape: {concatenated_df.shape}")
    logger.info(f"Total equipment records across all periods: {total_rows}")
    
    # Log summary statistics
    log_dataframe_info(logger, concatenated_df, "Concatenated Dataset")
    
    return concatenated_df


def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    ENHANCED: Comprehensive data cleaning and imputation with better handling
    """
    logger.info("Starting enhanced cleaning and imputation")
    initial_shape = df.shape
    
    # 1. Data validation and integrity check
    validation_results = validate_data_integrity(df, 'equipment')
    logger.info(f"Data validation results: {validation_results}")
    
    # 2. Handle duplicate equipment records intelligently
    if 'equipment' in df.columns:
        # For equipment data, we might have legitimate duplicates across time periods
        # Only remove exact duplicates (all columns identical)
        before_dedup = len(df)
        df = df.drop_duplicates()
        after_dedup = len(df)
        logger.info(f"Removed {before_dedup - after_dedup} exact duplicate rows")
    
    # 3. Smart column dropping based on missing values
    missing_percentage = df.isnull().sum() / len(df)
    cols_to_drop = missing_percentage[missing_percentage > 0.9].index  # Increased threshold
    if not cols_to_drop.empty:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped {len(cols_to_drop)} columns with >90% missing values: {list(cols_to_drop)}")
    
    # 4. Enhanced date parsing
    date_columns_to_parse = [col for col in df.columns if any(date_keyword in col.lower() 
                            for date_keyword in ['date', 'created', 'changed', 'warranty', 'start', 'end'])]
    
    for col in date_columns_to_parse:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                logger.info(f"Parsed date column: {col}")
            except:
                logger.warning(f"Could not parse date column: {col}")
    
    # 5. Intelligent imputation strategies
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # For numeric columns: use median for most, mean for costs/values
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            if any(keyword in col.lower() for keyword in ['value', 'cost', 'price', 'amount']):
                # Use forward fill for financial data, then median
                df[col] = df[col].fillna(method='ffill').fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns: use mode or 'Unknown'
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0 and df[col].value_counts().iloc[0] > len(df) * 0.1:  # Mode > 10% of data
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna('Unknown')
    
    # 6. Data type optimization
    df = optimize_dtypes(df)
    
    # 7. Memory cleanup
    memory_cleanup()
    
    final_shape = df.shape
    log_processing_step(logger, "Enhanced Cleaning and Imputation", initial_shape, final_shape)
    log_dataframe_info(logger, df, "Cleaned Dataset")
    
    return df