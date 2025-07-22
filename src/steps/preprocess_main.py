"""
Main preprocessing orchestrator
"""
import pandas as pd
from typing import Dict, Any

from config import config, logger, log_processing_start, log_step_complete, log_error, format_number
from .load_data import load_all_data, validate_loaded_data
from .combine_data import combine_all_data, validate_combined_data
from .reduce_memory import reduce_memory_usage
from .save_data import save_processed_data, validate_saved_files


def run_preprocessing_pipeline() -> Dict[str, Any]:
    """
    Run the complete preprocessing pipeline

    Returns:
        Dictionary with pipeline results and statistics
    """
    try:
        log_processing_start(logger, "ONGC Equipment Data Preprocessing Pipeline v2.0")

        # Validate setup
        if not config.validate_setup():
            raise ValueError("Setup validation failed - check raw data directory and files")

        config.create_directories()

        all_stats = {}

        # Step 1: Load all data
        logger.info("Step 1/4: Loading data files")
        dataframes, file_names = load_all_data()

        if not validate_loaded_data(dataframes, file_names):
            raise ValueError("Data validation failed after loading")

        all_stats['loading'] = {
            'files_loaded': len(dataframes),
            'total_original_rows': sum(len(df) for df in dataframes)
        }

        # Step 2: Combine all data
        logger.info("Step 2/4: Combining datasets")
        combined_df, combination_stats = combine_all_data(dataframes, file_names)

        if not validate_combined_data(combined_df):
            raise ValueError("Data validation failed after combination")

        all_stats['combination'] = combination_stats

        # Step 3: Optimize memory
        logger.info("Step 3/4: Optimizing memory usage")
        optimized_df, memory_stats = reduce_memory_usage(combined_df)
        all_stats['memory_optimization'] = memory_stats

        # Step 4: Save results
        logger.info("Step 4/4: Saving processed data")
        saved_files, report_path = save_processed_data(optimized_df, all_stats)

        if not validate_saved_files(saved_files):
            raise ValueError("File validation failed after saving")

        all_stats['saving'] = {
            'files_saved': len(saved_files),
            'saved_files': saved_files,
            'report_path': report_path
        }

        # Final results
        final_results = {
            'status': 'success',
            'message': 'Preprocessing completed successfully',
            'final_shape': optimized_df.shape,
            'final_records': len(optimized_df),
            'final_columns': len(optimized_df.columns),
            'statistics': all_stats
        }

        # Log final summary
        _log_final_summary(final_results)

        return final_results

    except Exception as e:
        error_message = f"Pipeline failed: {str(e)}"
        log_error(logger, "Preprocessing Pipeline", error_message)

        return {
            'status': 'failed',
            'message': error_message,
            'error': str(e)
        }


def _log_final_summary(results: Dict[str, Any]):
    """Log comprehensive final summary"""
    stats = results['statistics']

    logger.info("="*60)
    logger.info("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*60)

    # Loading summary
    loading_stats = stats.get('loading', {})
    logger.info(f"Files processed: {loading_stats.get('files_loaded', 0)}")
    logger.info(f"Original total rows: {format_number(loading_stats.get('total_original_rows', 0))}")

    # Combination summary
    combo_stats = stats.get('combination', {})
    logger.info(f"Final rows: {format_number(combo_stats.get('final_rows', 0))}")
    logger.info(f"Rows removed: {format_number(combo_stats.get('rows_removed', 0))}")
    logger.info(f"Final columns: {combo_stats.get('final_columns', 0)}")

    # Memory optimization summary
    memory_stats = stats.get('memory_optimization', {})
    if memory_stats:
        logger.info(f"Memory optimized: {memory_stats.get('memory_reduction_percent', 0):.1f}% reduction")
        logger.info(f"Memory saved: {memory_stats.get('memory_saved_mb', 0):.2f} MB")

    # Saving summary
    saving_stats = stats.get('saving', {})
    logger.info(f"Output files: {saving_stats.get('files_saved', 0)}")

    # Show saved files
    saved_files = saving_stats.get('saved_files', {})
    if saved_files:
        logger.info("Saved files:")
        for file_type, file_path in saved_files.items():
            from pathlib import Path
            logger.info(f"  {file_type.upper()}: {Path(file_path).name}")

    logger.info("="*60)


def get_pipeline_info() -> Dict[str, Any]:
    """Get information about the pipeline configuration"""
    return {
        'version': '2.0.0-simplified',
        'required_columns': config.REQUIRED_COLUMNS,
        'column_mappings': config.COLUMN_MAPPINGS,
        'processing_options': {
            'remove_duplicates': config.REMOVE_DUPLICATES,
            'remove_empty_rows': config.REMOVE_EMPTY_ROWS,
            'optimize_memory': config.OPTIMIZE_MEMORY,
            'save_csv': config.SAVE_CSV,
            'save_parquet': config.SAVE_PARQUET
        },
        'directories': {
            'raw_data': config.RAW_DATA_DIR,
            'output': config.OUTPUT_DIR,
            'reports': config.REPORTS_DIR,
            'logs': config.LOGS_DIR
        }
    }

