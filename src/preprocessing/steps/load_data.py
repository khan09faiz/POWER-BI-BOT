"""
Load raw data step
Simple, reliable data loading from Excel files
"""
import pandas as pd
from pathlib import Path
from typing import List, Tuple

from config import config, logger, log_step_complete, log_error, format_number
from utils import load_excel_file, discover_excel_files


def load_all_data() -> Tuple[List[pd.DataFrame], List[str]]:
    """
    Load all Excel files from the raw data directory
    """
    try:
        logger.info("Starting data loading step")

        # Discover Excel files
        raw_dir = Path(config.RAW_DATA_DIR)
        excel_files = discover_excel_files(raw_dir)

        logger.info(f"Found {len(excel_files)} Excel files to process")

        # Load all files
        dataframes = []
        file_names = []

        for file_path in excel_files:
            try:
                df = load_excel_file(file_path)
                dataframes.append(df)
                file_names.append(file_path.name)

                logger.info(f"Loaded {file_path.name}: {format_number(len(df))} rows, {len(df.columns)} columns")

            except Exception as e:
                log_error(logger, f"Loading {file_path.name}", str(e))
                continue

        if not dataframes:
            raise ValueError("No files could be loaded successfully")

        total_rows = sum(len(df) for df in dataframes)
        log_step_complete(logger, "Data Loading",
                         f"{len(dataframes)} files, {format_number(total_rows)} total rows")

        return dataframes, file_names

    except Exception as e:
        log_error(logger, "Data Loading", str(e))
        raise


def validate_loaded_data(dataframes: List[pd.DataFrame], file_names: List[str]) -> bool:
    """
    Basic validation of loaded data
    """
    if not dataframes:
        logger.error("No dataframes to validate")
        return False

    if len(dataframes) != len(file_names):
        logger.error("Mismatch between dataframes and file names")
        return False

    for i, (df, file_name) in enumerate(zip(dataframes, file_names)):
        if df.empty:
            logger.warning(f"Empty dataframe loaded from {file_name}")

        if len(df.columns) == 0:
            logger.error(f"No columns in dataframe from {file_name}")
            return False

    logger.info("Data validation passed")
    return True


def get_loading_summary(dataframes: List[pd.DataFrame], file_names: List[str]) -> dict:
    """
    Get summary statistics for loaded data
    """
    return {
        'files_loaded': len(dataframes),
        'file_names': file_names,
        'total_rows': sum(len(df) for df in dataframes),
        'file_details': [
            {'name': name, 'rows': len(df), 'columns': len(df.columns)}
            for name, df in zip(file_names, dataframes)
        ]
    }
