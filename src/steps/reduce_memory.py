"""
Memory reduction step
Apply memory optimizations to reduce DataFrame memory usage
"""
import pandas as pd
from typing import Tuple, Dict, Any

from config import logger, log_step_complete, log_error
from utils import optimize_dataframe_memory, get_memory_usage, calculate_memory_reduction


def reduce_memory_usage(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply memory optimization to DataFrame
    """
    try:
        logger.info("Starting memory optimization step")

        # Get memory usage before optimization
        memory_before = get_memory_usage(df)
        logger.info(f"Memory usage before: {memory_before['total_memory_mb']} MB")

        # Apply memory optimization
        df_optimized = optimize_dataframe_memory(df)

        # Get memory usage after optimization
        memory_after = get_memory_usage(df_optimized)
        logger.info(f"Memory usage after: {memory_after['total_memory_mb']} MB")

        # Calculate reduction
        reduction_percent = calculate_memory_reduction(memory_before, memory_after)

        # Apply additional optimizations
        df_optimized = _apply_additional_optimizations(df_optimized)

        # Final memory usage
        memory_final = get_memory_usage(df_optimized)
        final_reduction = calculate_memory_reduction(memory_before, memory_final)

        optimization_stats = {
            'memory_before_mb': memory_before['total_memory_mb'],
            'memory_after_mb': memory_final['total_memory_mb'],
            'memory_reduction_percent': final_reduction,
            'memory_saved_mb': memory_before['total_memory_mb'] - memory_final['total_memory_mb']
        }

        log_step_complete(logger, "Memory Optimization",
                         f"{final_reduction}% reduction ({optimization_stats['memory_saved_mb']:.2f} MB saved)")

        return df_optimized, optimization_stats

    except Exception as e:
        log_error(logger, "Memory Optimization", str(e))
        raise


def _apply_additional_optimizations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply additional DataFrame optimizations
    """
    df_opt = df.copy()

    # Convert date columns
    date_columns = ['created_on', 'changed_on']
    for col in date_columns:
        if col in df_opt.columns:
            try:
                df_opt[col] = pd.to_datetime(df_opt[col], errors='coerce')
                logger.info(f"Converted {col} to datetime")
            except Exception:
                logger.warning(f"Could not convert {col} to datetime")

    # Optimize string columns with low cardinality
    for col in df_opt.select_dtypes(include=['object']).columns:
        if col not in date_columns:  # Skip date columns
            unique_ratio = df_opt[col].nunique() / len(df_opt) if len(df_opt) > 0 else 1

            if unique_ratio < 0.3:  # Less than 30% unique values
                df_opt[col] = df_opt[col].astype('category')
                logger.info(f"Converted {col} to category (unique ratio: {unique_ratio:.2%})")

    return df_opt


def get_optimization_report(stats: Dict[str, Any]) -> str:
    """
    Generate a formatted optimization report
    """
    report = f"""
                Memory Optimization Report:
                --------------------------
                Before: {stats['memory_before_mb']:.2f} MB
                After:  {stats['memory_after_mb']:.2f} MB
                Saved:  {stats['memory_saved_mb']:.2f} MB
                Reduction: {stats['memory_reduction_percent']:.1f}%
            """.strip()

    return report
