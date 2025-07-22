"""
Data combination utilities
Simple, efficient dataset merging and concatenation
"""
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


def concatenate_dataframes(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate multiple DataFrames with consistent columns
    """
    if not dataframes:
        raise ValueError("No dataframes provided for concatenation")

    # Ensure all dataframes have the same columns
    all_columns = set()
    for df in dataframes:
        all_columns.update(df.columns)

    # Standardize columns across all dataframes
    standardized_dfs = []
    for df in dataframes:
        df_copy = df.copy()

        # Add missing columns with NaN values
        for col in all_columns:
            if col not in df_copy.columns:
                df_copy[col] = pd.NA

        # Reorder columns consistently
        df_copy = df_copy.reindex(columns=sorted(all_columns))
        standardized_dfs.append(df_copy)

    # Concatenate all dataframes
    combined_df = pd.concat(standardized_dfs, ignore_index=True, sort=False)

    return combined_df


def filter_required_columns(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame to keep only required columns
    """
    available_columns = [col for col in required_columns if col in df.columns]

    if not available_columns:
        raise ValueError("None of the required columns found in the data")

    return df[available_columns].copy()


def clean_dataframe(df: pd.DataFrame, remove_duplicates: bool = True,
                   remove_empty_rows: bool = True) -> pd.DataFrame:
    """
    Clean DataFrame by removing duplicates and empty rows
    """
    df_clean = df.copy()

    original_rows = len(df_clean)

    # Remove completely empty rows
    if remove_empty_rows:
        df_clean = df_clean.dropna(how='all')

    # Remove duplicate rows
    if remove_duplicates:
        df_clean = df_clean.drop_duplicates()

    # Reset index
    df_clean = df_clean.reset_index(drop=True)

    rows_removed = original_rows - len(df_clean)

    return df_clean


def standardize_column_names(df: pd.DataFrame, column_mappings: Dict[str, str]) -> pd.DataFrame:
    """
    Standardize column names using provided mappings
    """
    return df.rename(columns=column_mappings)


def get_combination_stats(original_dfs: List[pd.DataFrame], combined_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get statistics about the combination process
    """
    total_original_rows = sum(len(df) for df in original_dfs)

    return {
        'files_combined': len(original_dfs),
        'original_total_rows': total_original_rows,
        'final_rows': len(combined_df),
        'rows_removed': total_original_rows - len(combined_df),
        'final_columns': len(combined_df.columns)
    }
