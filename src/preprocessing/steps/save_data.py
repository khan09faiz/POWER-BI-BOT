"""
Save data step
Save processed data in multiple formats with reports
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from config import config, logger, log_step_complete, log_error
from utils import save_csv, save_parquet, save_json_report, create_output_paths


def save_processed_data(df: pd.DataFrame, processing_stats: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
    """
    Save processed data in multiple formats
    """
    try:
        logger.info("Starting data saving step")

        if df.empty:
            raise ValueError("Cannot save empty DataFrame")

        # Create output directories
        output_dir = Path(config.OUTPUT_DIR)
        reports_dir = Path(config.REPORTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        # Create output file paths
        output_paths = create_output_paths(output_dir, "ongc_equipment_data")
        report_path = reports_dir / f"processing_report_{_get_timestamp()}.json"

        # Save CSV
        if config.SAVE_CSV:
            csv_path = save_csv(df, output_paths['csv'])
            saved_files['csv'] = csv_path
            logger.info(f"Saved CSV: {Path(csv_path).name}")

        # Save Parquet
        if config.SAVE_PARQUET:
            try:
                parquet_path = save_parquet(df, output_paths['parquet'])
                saved_files['parquet'] = parquet_path
                logger.info(f"Saved Parquet: {Path(parquet_path).name}")
            except ImportError:
                logger.warning("Parquet saving skipped - pyarrow not available")

        # Create and save processing report
        report_data = _create_processing_report(df, processing_stats, saved_files)
        report_path_str = save_json_report(report_data, report_path)
        saved_files['report'] = report_path_str

        log_step_complete(logger, "Data Saving", f"{len(saved_files)} files saved")

        return saved_files, report_path_str

    except Exception as e:
        log_error(logger, "Data Saving", str(e))
        raise


def _create_processing_report(df: pd.DataFrame, processing_stats: Dict[str, Any],
                             saved_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Create comprehensive processing report
    """
    report = {
        'processing_info': {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '2.0.0-simplified',
            'total_records': len(df),
            'total_columns': len(df.columns)
        },
        'processing_statistics': processing_stats,
        'data_quality': _analyze_data_quality(df),
        'output_files': saved_files,
        'column_info': _get_column_info(df)
    }

    return report


def _analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze data quality of final dataset
    """
    total_rows = len(df)

    quality_metrics = {
        'total_rows': total_rows,
        'total_columns': len(df.columns),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        'completeness': {}
    }

    # Calculate completeness for each column
    for col in df.columns:
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        completeness = (non_null_count / total_rows * 100) if total_rows > 0 else 0

        quality_metrics['completeness'][col] = {
            'non_null_count': int(non_null_count),
            'null_count': int(null_count),
            'completeness_percent': round(completeness, 2)
        }

    return quality_metrics


def _get_column_info(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about each column
    """
    column_info = {}

    for col in df.columns:
        info = {
            'data_type': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),
            'non_null_count': int(df[col].notna().sum()),
            'null_percentage': round(df[col].isna().sum() / len(df) * 100, 2)
        }

        # Add sample values for categorical/object columns
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            sample_values = df[col].dropna().unique()[:5]  # First 5 unique values
            info['sample_values'] = [str(val) for val in sample_values]

        column_info[col] = info

    return column_info


def _get_timestamp() -> str:
    """Get timestamp for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def validate_saved_files(saved_files: Dict[str, str]) -> bool:
    """
    Validate that all saved files exist and are readable
    """
    for file_type, file_path in saved_files.items():
        path = Path(file_path)

        if not path.exists():
            logger.error(f"Saved file does not exist: {file_path}")
            return False

        if path.stat().st_size == 0:
            logger.error(f"Saved file is empty: {file_path}")
            return False

    logger.info("All saved files validated successfully")
    return True
