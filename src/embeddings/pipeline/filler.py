"""
Core pipeline for semantic data completion
Orchestrates the entire process of loading, embedding, searching, and filling missing fields
"""
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
import logging
from pathlib import Path

from config import settings
from utils import log_execution_time, log_pipeline_start, log_pipeline_complete, default_logger
from data_processing import DataLoader, DataWriter
from model import Qwen3Embedder
from search import FAISSSearchEngine


class SemanticDataFiller:
    """Main pipeline for semantic data completion"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger
        self.data_loader = DataLoader(logger=self.logger)
        self.data_writer = DataWriter(logger=self.logger)
        self.embedder = None
        self.search_engine = None

    def _identify_missing_fields(self, df: pd.DataFrame) -> Dict[str, int]:
        """Identify which fields are missing and how many records need completion"""
        missing_info = {}

        target_fields = ['equipment_type', 'equipment_description']

        for field in target_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                if missing_count > 0:
                    missing_info[field] = missing_count

        return missing_info

    def _prepare_embedding_texts(self, df: pd.DataFrame, embedder: Qwen3Embedder) -> List[str]:
        """Prepare texts for embedding from equipment data"""
        texts = []

        for _, row in df.iterrows():
            # Get available fields
            equipment_type = row.get('equipment_type')
            equipment_description = row.get('equipment_description')

            # Combine fields into text for embedding
            text = embedder.combine_equipment_fields(
                equipment_type=equipment_type,
                equipment_description=equipment_description
            )

            texts.append(text)

        return texts

    @log_execution_time()
    def _build_master_index(self, master_file: str) -> Tuple[pd.DataFrame, FAISSSearchEngine]:
        """Load master data and build search index"""
        # Load master data
        self.logger.info("Loading master data.")
        master_df = self.data_loader.load_master_data(master_file)

        # Initialize embedder
        self.logger.info("Initializing embedding model.")
        embedder = Qwen3Embedder(logger=self.logger)
        embedder.load_model()
        self.embedder = embedder

        # Prepare texts for embedding
        self.logger.info("Preparing master data for embedding.")
        master_texts = self._prepare_embedding_texts(master_df, embedder)

        # Generate embeddings
        self.logger.info("Generating embeddings for master data.")
        master_embeddings = embedder.encode_texts(master_texts)

        # Build search index
        self.logger.info("Building search index.")
        search_engine = FAISSSearchEngine(logger=self.logger)
        search_engine.build_index(master_embeddings, master_df)
        self.search_engine = search_engine

        return master_df, search_engine

    @log_execution_time()
    def _fill_missing_fields(
        self,
        target_df: pd.DataFrame,
        missing_fields: List[str]
    ) -> pd.DataFrame:
        """Fill missing fields using semantic search"""
        if not missing_fields:
            self.logger.info("No missing fields to fill")
            return target_df

        # Create a copy for modification
        filled_df = target_df.copy()

        # Find rows with missing data
        missing_mask = filled_df[missing_fields].isna().any(axis=1)
        missing_rows = filled_df[missing_mask]

        if len(missing_rows) == 0:
            self.logger.info("No rows with missing data found")
            return filled_df

        self.logger.info(f"Processing {len(missing_rows)} rows with missing data")

        # Prepare query texts from rows with missing data
        query_texts = self._prepare_embedding_texts(missing_rows, self.embedder)

        # Generate embeddings for queries
        self.logger.info("Generating embeddings for target data.")
        query_embeddings = self.embedder.encode_texts(query_texts)

        # Find matches
        self.logger.info("Finding semantic matches.")
        matches = self.search_engine.find_best_matches(query_embeddings, missing_fields)

        # Fill missing fields
        filled_count = 0
        for i, (orig_idx, match) in enumerate(zip(missing_rows.index, matches)):
            if match['found_matches']:
                filled_fields = match['filled_fields']

                for field in missing_fields:
                    if field in filled_fields and pd.isna(filled_df.at[orig_idx, field]):
                        filled_df.at[orig_idx, field] = filled_fields[field]

                        # Add similarity score if configured
                        if settings.processing.include_similarity_scores:
                            score_field = f'{field}_similarity_score'
                            if score_field in filled_fields:
                                filled_df.at[orig_idx, score_field] = filled_fields[score_field]

                        filled_count += 1

        self.logger.info(f"Filled {filled_count} missing field values")
        return filled_df

    @log_execution_time()
    def process_files(
        self,
        master_file: str,
        target_file: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process master and target files to fill missing data"""

        # Log pipeline start
        config_info = settings.get_device_info()
        log_pipeline_start(self.logger, config_info)

        try:
            # Step 1: Build master index
            master_df, search_engine = self._build_master_index(master_file)

            # Step 2: Load target data
            self.logger.info("Loading target data.")
            target_df = self.data_loader.load_target_data(target_file)

            # Step 3: Identify missing fields
            missing_info = self._identify_missing_fields(target_df)
            if not missing_info:
                self.logger.info("No missing fields found in target data")
                return {
                    'status': 'completed',
                    'message': 'No missing fields to fill',
                    'output_file': target_file,
                    'statistics': {'filled_fields': 0}
                }

            self.logger.info(f"Missing fields identified: {missing_info}")

            # Step 4: Fill missing fields
            missing_fields = list(missing_info.keys())
            filled_df = self._fill_missing_fields(target_df, missing_fields)

            # Step 5: Save results
            self.logger.info("Saving completed data.")
            output_path = self.data_writer.save_completed_data(
                filled_df,
                target_file,
                output_file
            )

            # Step 6: Create completion report
            completion_report = self.data_writer.create_completion_report(
                target_df, filled_df, output_path
            )

            # Save report
            report_path = self.data_writer.save_completion_report(completion_report, output_path)

            # Final results
            results = {
                'status': 'completed',
                'message': 'Data completion successful',
                'output_file': output_path,
                'report_file': report_path,
                'statistics': {
                    'master_records': len(master_df),
                    'target_records': len(target_df),
                    'completed_records': len(filled_df),
                    'missing_fields_identified': missing_info,
                    'index_stats': search_engine.get_index_stats()
                }
            }

            log_pipeline_complete(self.logger, results)
            return results

        except Exception as e:
            error_msg = f"Pipeline failed: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'failed',
                'message': error_msg,
                'error': str(e)
            }

        finally:
            # Cleanup resources
            self._cleanup()

    def _cleanup(self):
        """Clean up resources"""
        if self.embedder is not None:
            self.embedder._cleanup()
            self.embedder = None

        if self.search_engine is not None:
            self.search_engine.cleanup()
            self.search_engine = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self._cleanup()
