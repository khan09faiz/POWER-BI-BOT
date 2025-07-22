"""
Data combination step
Combine multiple DataFrames into a single dataset
"""
import pandas as pd
from typing import List, Tuple, Dict, Any

from config import config, logger, log_step_complete, log_error, format_number
from utils import concatenate_dataframes, filter_required_columns, clean_dataframe, standardize_column_names, get_combination_stats


def combine_all_data(dataframes: List[pd.DataFrame], file_names: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Combine all loaded DataFrames into a single dataset
    """
    try:
        logger.info("Starting data combination step")

        if not dataframes:
            raise ValueError("No dataframes provided for combination")

        # Step 1: Concatenate all dataframes
        logger.info("Concatenating dataframes...")
        combined_df = concatenate_dataframes(dataframes)
        logger.info(f"Combined shape: {combined_df.shape}")

        # Step 2: Filter to required columns
        logger.info("Filtering to required columns...")
        filtered_df = filter_required_columns(combined_df, config.REQUIRED_COLUMNS)

        missing_columns = [col for col in config.REQUIRED_COLUMNS if col not in combined_df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")

        available_columns = list(filtered_df.columns)
        logger.info(f"Kept {len(available_columns)} columns: {available_columns}")

        # Step 3: Clean data
        logger.info("Cleaning data...")
        cleaned_df = clean_dataframe(
            filtered_df,
            remove_duplicates=config.REMOVE_DUPLICATES,
            remove_empty_rows=config.REMOVE_EMPTY_ROWS
        )

        # Step 4: Standardize column names
        logger.info("Standardizing column names...")
        final_df = standardize_column_names(cleaned_df, config.COLUMN_MAPPINGS)

        # Get combination statistics
        combination_stats = get_combination_stats(dataframes, final_df)
        combination_stats.update({
            'available_columns': available_columns,
            'missing_columns': missing_columns,
            'standardized_columns': list(final_df.columns)
        })

        log_step_complete(logger, "Data Combination",
                         f"{format_number(len(final_df))} rows, {len(final_df.columns)} columns")

        return final_df, combination_stats

    except Exception as e:
        log_error(logger, "Data Combination", str(e))
        raise


def validate_combined_data(df: pd.DataFrame) -> bool:
    """
    Validate the combined dataset
    """
    if df.empty:
        logger.error("Combined dataset is empty")
        return False

    if len(df.columns) == 0:
        logger.error("Combined dataset has no columns")
        return False

    # Check for basic data quality issues
    total_rows = len(df)

    # Check for completely null columns
    null_columns = []
    for col in df.columns:
        null_ratio = df[col].isna().sum() / total_rows
        if null_ratio == 1.0:
            null_columns.append(col)
        elif null_ratio > 0.9:
            logger.warning(f"Column '{col}' is {null_ratio:.1%} null")

    if null_columns:
        logger.warning(f"Completely null columns: {null_columns}")

    logger.info("Combined data validation passed")
    return True


def get_combination_report(stats: Dict[str, Any]) -> str:
    """
    Generate a formatted combination report
    """
    report = f"""
                Data Combination Report:
                -----------------------
                Files Combined: {stats['files_combined']}
                Original Total Rows: {format_number(stats['original_total_rows'])}
                Final Rows: {format_number(stats['final_rows'])}
                Rows Removed: {format_number(stats['rows_removed'])}
                Final Columns: {stats['final_columns']}
                Available Columns: {len(stats['available_columns'])}
                Missing Columns: {len(stats['missing_columns'])}
            """.strip()

    if stats['missing_columns']:
        report += f"\nMissing: {', '.join(stats['missing_columns'])}"

    return report
