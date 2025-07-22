"""
Memory optimization utilities
Simple, effective memory reduction techniques
"""
import pandas as pd
from typing import Dict


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage through dtype conversion
    """
    df = df.copy()

    # Optimize integer columns
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()

        if pd.isna(col_min) or pd.isna(col_max):
            continue

        if col_min >= 0:  # Unsigned integers
            if col_max < 255:
                df[col] = df[col].astype('uint8')
            elif col_max < 65535:
                df[col] = df[col].astype('uint16')
            elif col_max < 4294967295:
                df[col] = df[col].astype('uint32')
        else:  # Signed integers
            if col_min >= -128 and col_max <= 127:
                df[col] = df[col].astype('int8')
            elif col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype('int16')
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype('int32')

    # Optimize float columns
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # Convert categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')

    return df


def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get memory usage statistics for DataFrame
    """
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return {
        'total_memory_mb': round(memory_mb, 2),
        'memory_per_row_kb': round(memory_mb * 1024 / len(df), 3) if len(df) > 0 else 0,
        'rows': len(df),
        'columns': len(df.columns)
    }


def calculate_memory_reduction(before: Dict[str, float], after: Dict[str, float]) -> float:
    """
    Calculate percentage memory reduction
    """
    if before['total_memory_mb'] == 0:
        return 0

    reduction = (before['total_memory_mb'] - after['total_memory_mb']) / before['total_memory_mb']
    return round(reduction * 100, 1)
