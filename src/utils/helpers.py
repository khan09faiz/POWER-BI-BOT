"""
General helper functions for the POWER-BI-BOT project
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case and remove special characters
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Standardize column names
    new_columns = []
    for col in df_copy.columns:
        # Convert to string and handle various cases
        col_str = str(col).strip()
        
        # Replace special characters and spaces with underscores
        col_clean = re.sub(r'[^0-9a-zA-Z]+', '_', col_str)
        
        # Convert to lowercase
        col_clean = col_clean.lower()
        
        # Remove leading/trailing underscores
        col_clean = col_clean.strip('_')
        
        # Handle empty column names
        if not col_clean:
            col_clean = f'unnamed_{len(new_columns)}'
        
        new_columns.append(col_clean)
    
    df_copy.columns = new_columns
    return df_copy

def detect_date_columns(df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
    """
    Automatically detect date columns in a DataFrame
    
    Args:
        df: Input DataFrame
        threshold: Minimum percentage of valid dates to consider a column as date
        
    Returns:
        List of column names that are likely date columns
    """
    date_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to parse as datetime and check success rate
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                valid_ratio = parsed.notna().sum() / len(df)
                
                if valid_ratio >= threshold:
                    date_columns.append(col)
            except:
                continue
    
    return date_columns

def extract_time_period(filename: str) -> Dict[str, Any]:
    """
    Extract time period information from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        Dictionary with period information
    """
    # Pattern to match date ranges in filename
    pattern = r'(\d{2})\.(\d{2})\.(\d{4})\s+TO\s+(\d{2})\.(\d{2})\.(\d{4})'
    match = re.search(pattern, filename)
    
    if match:
        start_day, start_month, start_year = match.groups()[:3]
        end_day, end_month, end_year = match.groups()[3:]
        
        start_date = pd.to_datetime(f"{start_year}-{start_month}-{start_day}")
        end_date = pd.to_datetime(f"{end_year}-{end_month}-{end_day}")
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'period_years': int(end_year) - int(start_year),
            'period_label': f"{start_year}-{end_year}"
        }
    
    return {}

def identify_categorical_columns(df: pd.DataFrame, 
                                max_unique_ratio: float = 0.05,
                                max_unique_count: int = 50) -> List[str]:
    """
    Identify columns that should be treated as categorical
    
    Args:
        df: Input DataFrame
        max_unique_ratio: Maximum ratio of unique values to total rows
        max_unique_count: Maximum number of unique values
        
    Returns:
        List of column names that should be categorical
    """
    categorical_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            unique_count = df[col].nunique()
            unique_ratio = unique_count / len(df)
            
            if (unique_ratio <= max_unique_ratio or unique_count <= max_unique_count):
                categorical_columns.append(col)
    
    return categorical_columns

def create_backup(df: pd.DataFrame, backup_path: str) -> None:
    """
    Create a backup of the DataFrame
    
    Args:
        df: DataFrame to backup
        backup_path: Path where backup should be saved
    """
    backup_path = Path(backup_path)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    
    if backup_path.suffix == '.parquet':
        df.to_parquet(backup_path, index=False)
    elif backup_path.suffix == '.csv':
        df.to_csv(backup_path, index=False)
    else:
        # Default to parquet for efficiency
        backup_path = backup_path.with_suffix('.parquet')
        df.to_parquet(backup_path, index=False)

def validate_data_integrity(df: pd.DataFrame, 
                           id_column: str = 'equipment_id') -> Dict[str, Any]:
    """
    Validate data integrity and return summary
    
    Args:
        df: DataFrame to validate
        id_column: Column to use as primary identifier
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'duplicate_rows': df.duplicated().sum(),
        'missing_values_total': df.isnull().sum().sum(),
        'missing_values_by_column': df.isnull().sum().to_dict(),
        'empty_columns': df.columns[df.isnull().all()].tolist(),
        'data_types': df.dtypes.to_dict(),
    }
    
    if id_column in df.columns:
        validation_results['duplicate_ids'] = df[id_column].duplicated().sum()
        validation_results['missing_ids'] = df[id_column].isnull().sum()
    
    return validation_results

def safe_convert_numeric(series: pd.Series) -> pd.Series:
    """
    Safely convert a series to numeric, handling errors gracefully
    
    Args:
        series: Pandas Series to convert
        
    Returns:
        Converted Series
    """
    try:
        return pd.to_numeric(series, errors='coerce')
    except:
        return series

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"
