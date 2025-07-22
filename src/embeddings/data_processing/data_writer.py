"""
Data writing utilities for semantic data completion pipeline
Handles output in various formats with similarity scores
"""
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from ..config import settings
from ..utils import log_execution_time, default_logger


class DataWriter:
    """Writes completed data with optional similarity scores"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger

    def _create_output_filename(self, original_file: str, suffix: str = "completed") -> str:
        """Create output filename based on original file"""
        path = Path(original_file)

        # Use configured output format
        output_ext = f".{settings.processing.output_format}"
        output_filename = f"{path.stem}_{suffix}{output_ext}"

        return str(path.parent / output_filename)

    def _backup_original(self, original_file: str):
        """Create backup of original file"""
        if not settings.processing.backup_original:
            return

        original_path = Path(original_file)
        backup_path = original_path.parent / f"{original_path.stem}_backup{original_path.suffix}"

        try:
            backup_path.write_bytes(original_path.read_bytes())
            self.logger.info(f"Created backup: {backup_path.name}")
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {str(e)}")

    def _prepare_output_data(self, df: pd.DataFrame, include_scores: bool = True) -> pd.DataFrame:
        """Prepare data for output with proper column ordering"""
        output_df = df.copy()

        # Define preferred column order
        base_columns = ['equipment_number', 'equipment_type', 'equipment_description']

        # Add similarity score columns if they exist and are requested
        score_columns = []
        if include_scores:
            for col in df.columns:
                if 'similarity' in col.lower() or 'score' in col.lower():
                    score_columns.append(col)

        # Add any other columns
        other_columns = [col for col in df.columns
                        if col not in base_columns + score_columns]

        # Reorder columns
        ordered_columns = []
        for col in base_columns:
            if col in df.columns:
                ordered_columns.append(col)

        ordered_columns.extend(other_columns)
        ordered_columns.extend(score_columns)

        return output_df[ordered_columns]

    @log_execution_time()
    def save_completed_data(
        self,
        df: pd.DataFrame,
        original_file: str,
        output_file: Optional[str] = None,
        include_similarity_scores: Optional[bool] = None
    ) -> str:
        """Save completed data to file"""

        # Use config default if not specified
        if include_similarity_scores is None:
            include_similarity_scores = settings.processing.include_similarity_scores

        # Create output filename if not provided
        if output_file is None:
            output_file = self._create_output_filename(original_file)

        # Backup original file
        self._backup_original(original_file)

        # Prepare output data
        output_df = self._prepare_output_data(df, include_similarity_scores)

        self.logger.info(f"Saving completed data to: {Path(output_file).name}")

        try:
            # Save based on file extension
            output_path = Path(output_file)
            output_extension = output_path.suffix.lower()

            if output_extension == '.csv':
                output_df.to_csv(output_file, index=False, encoding='utf-8')
            elif output_extension in ['.xlsx', '.xls']:
                output_df.to_excel(output_file, index=False)
            else:
                # Fallback to CSV
                csv_file = output_path.with_suffix('.csv')
                output_df.to_csv(csv_file, index=False, encoding='utf-8')
                output_file = str(csv_file)
                self.logger.warning(f"Unsupported output format, saved as CSV: {csv_file.name}")

            self.logger.info(f"Successfully saved {len(output_df)} records")
            return output_file

        except Exception as e:
            self.logger.error(f"Failed to save output file: {str(e)}")
            raise

    def create_completion_report(
        self,
        original_df: pd.DataFrame,
        completed_df: pd.DataFrame,
        output_file: str
    ) -> Dict[str, Any]:
        """Create completion report with statistics"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "output_file": output_file,
            "total_records": len(completed_df),
            "original_records": len(original_df)
        }

        # Calculate completion statistics
        for field in ['equipment_type', 'equipment_description']:
            if field in original_df.columns and field in completed_df.columns:
                original_missing = original_df[field].isna().sum()
                completed_missing = completed_df[field].isna().sum()
                filled_count = original_missing - completed_missing

                report[f"{field}_statistics"] = {
                    "originally_missing": int(original_missing),
                    "filled_by_pipeline": int(filled_count),
                    "still_missing": int(completed_missing),
                    "completion_rate": f"{(filled_count / original_missing * 100):.1f}%" if original_missing > 0 else "0%"
                }

        # Add similarity score statistics if available
        similarity_columns = [col for col in completed_df.columns
                             if 'similarity' in col.lower() or 'score' in col.lower()]

        if similarity_columns:
            for col in similarity_columns:
                scores = completed_df[col].dropna()
                if len(scores) > 0:
                    report[f"{col}_statistics"] = {
                        "mean_score": float(scores.mean()),
                        "median_score": float(scores.median()),
                        "min_score": float(scores.min()),
                        "max_score": float(scores.max()),
                        "scores_available": int(len(scores))
                    }

        return report

    @log_execution_time()
    def save_completion_report(
        self,
        report: Dict[str, Any],
        output_file: str
    ) -> str:
        """Save completion report as JSON"""
        report_file = str(Path(output_file).with_suffix('.json'))

        try:
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Completion report saved: {Path(report_file).name}")
            return report_file

        except Exception as e:
            self.logger.error(f"Failed to save completion report: {str(e)}")
            raise
