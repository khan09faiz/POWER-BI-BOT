"""
Data Loading utilities for ONGC Equipment Data Processing Pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import concurrent.futures
from .config import get_config
from .logger import monitor_performance

logger = logging.getLogger('ongc_pipeline')

class DataLoader:
    """Data loading and file discovery utilities"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    @monitor_performance
    def discover_files(self) -> List[str]:
        """Autonomously discover data files in the directory"""
        raw_data_dir = Path(self.config.raw_data_dir)
        found_files = []
        
        logger.info(f"🔍 Searching for data files in: {raw_data_dir}")
        
        # First try exact matches
        for file_name in self.config.expected_files:
            file_path = raw_data_dir / file_name
            if file_path.exists():
                found_files.append(file_name)
                logger.info(f"✅ Found expected file: {file_name}")
        
        # If no exact matches and auto-discovery is enabled
        if not found_files and self.config.auto_discover_files:
            logger.info("🔍 No exact matches found. Auto-discovering files...")
            
            # Search for Excel and CSV files
            patterns = ["*.xlsx", "*.xls", "*.csv"]
            for pattern in patterns:
                files = list(raw_data_dir.glob(pattern))
                for file_path in files:
                    if file_path.name not in found_files:
                        found_files.append(file_path.name)
                        logger.info(f"📁 Auto-discovered: {file_path.name}")
        
        if not found_files:
            logger.warning("⚠️ No data files found!")
        else:
            logger.info(f"📊 Total files found: {len(found_files)}")
        
        return found_files
    
    @monitor_performance
    def load_file(self, file_name: str) -> Optional[pd.DataFrame]:
        """Load a single data file with error handling"""
        file_path = Path(self.config.raw_data_dir) / file_name
        
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_name}")
            return None
        
        try:
            logger.info(f"📊 Loading file: {file_name}")
            
            # Determine file type and load accordingly
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == '.csv':
                # Try different encodings for CSV files
                encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        logger.info(f"   Successfully loaded with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if df is None:
                    logger.error(f"❌ Could not load CSV file with any encoding: {file_name}")
                    return None
            else:
                logger.error(f"❌ Unsupported file format: {file_name}")
                return None
            
            logger.info(f"   Original shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_name}: {str(e)}")
            return None
    
    @monitor_performance
    def filter_columns(self, df: pd.DataFrame, file_name: str) -> pd.DataFrame:
        """Filter DataFrame to keep only required columns"""
        if df is None or df.empty:
            return df
        
        # Check which required columns are available
        available_columns = []
        missing_columns = []
        
        for col in self.config.required_columns:
            if col in df.columns:
                available_columns.append(col)
            else:
                missing_columns.append(col)
        
        if missing_columns:
            logger.warning(f"   Missing columns in {file_name}: {missing_columns}")
        
        if not available_columns:
            logger.error(f"   No required columns found in {file_name}")
            return pd.DataFrame()
        
        # Keep only available required columns
        df_filtered = df[available_columns].copy()
        
        # Add metadata columns
        df_filtered['data_source_file'] = file_name
        df_filtered['period_label'] = self._extract_period_from_filename(file_name)
        
        logger.info(f"   Filtered shape: {df_filtered.shape}")
        logger.info(f"   Kept columns: {available_columns}")
        
        return df_filtered
    
    def _extract_period_from_filename(self, file_name: str) -> str:
        """Extract period information from filename"""
        # Remove common prefixes and suffixes
        period = file_name.replace('EQUI_', '').replace('.xlsx', '').replace('.xls', '').replace('.csv', '')
        return period
    
    @monitor_performance
    def load_and_concatenate(self) -> Optional[pd.DataFrame]:
        """Load all files and concatenate them with parallel processing support"""
        logger.info("🚀 Starting data loading and concatenation")
        
        # Discover files
        files = self.discover_files()
        
        if not files:
            logger.error("❌ No files to process")
            return None
        
        all_dataframes = []
        
        # Check if parallel processing is enabled
        if getattr(self.config, 'parallel_processing', False) and len(files) > 1:
            logger.info("⚡ Using parallel processing for file loading")
            all_dataframes = self._load_files_parallel(files)
        else:
            logger.info("📊 Using sequential processing for file loading")
            all_dataframes = self._load_files_sequential(files)
        
        if not all_dataframes:
            logger.error("❌ No data could be loaded from any file")
            return None
        
        # Concatenate all dataframes
        logger.info("🔗 Concatenating all dataframes...")
        
        # Ensure all dataframes have the same columns
        all_columns = set()
        for df in all_dataframes:
            all_columns.update(df.columns)
        
        # Standardize columns across all dataframes
        standardized_dfs = []
        for df in all_dataframes:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Reorder columns consistently
            df = df.reindex(columns=sorted(all_columns))
            standardized_dfs.append(df)
        
        # Concatenate with memory optimization
        try:
            master_df = pd.concat(standardized_dfs, ignore_index=True, copy=False)
        except Exception as e:
            logger.warning(f"⚠️ Memory-optimized concatenation failed, using standard method: {e}")
            master_df = pd.concat(standardized_dfs, ignore_index=True)
        
        logger.info(f"✅ Concatenation complete!")
        logger.info(f"   Final shape: {master_df.shape}")
        logger.info(f"   Total records: {len(master_df):,}")
        logger.info(f"   Files processed: {len(all_dataframes)}")
        
        return master_df
    
    def _load_files_sequential(self, files: List[str]) -> List[pd.DataFrame]:
        """Load files sequentially"""
        all_dataframes = []
        
        for file_name in files:
            df = self.load_file(file_name)
            
            if df is not None and not df.empty:
                # Filter columns
                df_filtered = self.filter_columns(df, file_name)
                
                if not df_filtered.empty:
                    all_dataframes.append(df_filtered)
                else:
                    logger.warning(f"⚠️ No data retained from {file_name} after filtering")
            else:
                logger.warning(f"⚠️ Could not load data from {file_name}")
        
        return all_dataframes
    
    def _load_files_parallel(self, files: List[str]) -> List[pd.DataFrame]:
        """Load files in parallel using ThreadPoolExecutor"""
        all_dataframes = []
        max_workers = getattr(self.config, 'max_workers', 4)
        
        def load_and_filter_file(file_name: str) -> Optional[pd.DataFrame]:
            """Load and filter a single file"""
            try:
                df = self.load_file(file_name)
                if df is not None and not df.empty:
                    df_filtered = self.filter_columns(df, file_name)
                    return df_filtered if not df_filtered.empty else None
                return None
            except Exception as e:
                logger.error(f"❌ Error in parallel loading of {file_name}: {e}")
                return None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all file loading tasks
            future_to_file = {executor.submit(load_and_filter_file, file_name): file_name 
                            for file_name in files}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file_name = future_to_file[future]
                try:
                    df = future.result()
                    if df is not None:
                        all_dataframes.append(df)
                        logger.info(f"✅ Parallel loading completed for {file_name}")
                    else:
                        logger.warning(f"⚠️ No data retained from {file_name}")
                except Exception as e:
                    logger.error(f"❌ Parallel loading failed for {file_name}: {e}")
        
        return all_dataframes

def create_data_loader(config=None) -> DataLoader:
    """Factory function to create a DataLoader instance"""
    return DataLoader(config)