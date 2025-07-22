"""
Data loading utilities for semantic data completion pipeline
Handles CSV, XLS, XLSX files with robust validation
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import logging

from ..config import settings
from ..utils import log_execution_time, default_logger


class DataValidator:
    """Validates loaded data for required columns and format"""

    REQUIRED_COLUMNS = {
        'equipment_type': ['Equipment Type', 'ObjectType', 'equipment_type', 'equip_type'],
        'equipment_description': ['Equipment Description', 'equipment_description', 'description', 'desc']
    }

    @classmethod
    def find_column_mapping(cls, df: pd.DataFrame) -> Dict[str, str]:
        """Find column mappings in dataframe"""
        column_mapping = {}
        df_columns_lower = [col.lower() for col in df.columns]

        for standard_name, possible_names in cls.REQUIRED_COLUMNS.items():
            found_column = None

            # Exact match first
            for possible_name in possible_names:
                if possible_name in df.columns:
                    found_column = possible_name
                    break

            # Case-insensitive match
            if not found_column:
                for possible_name in possible_names:
                    if possible_name.lower() in df_columns_lower:
                        idx = df_columns_lower.index(possible_name.lower())
                        found_column = df.columns[idx]
                        break

            if found_column:
                column_mapping[standard_name] = found_column

        return column_mapping

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame, file_type: str = "data") -> Tuple[bool, List[str]]:
        """Validate dataframe has required columns"""
        issues = []

        if df.empty:
            issues.append(f"{file_type} file is empty")
            return False, issues

        column_mapping = cls.find_column_mapping(df)

        # Check for equipment number (always required)
        if 'equipment_number' not in column_mapping:
            issues.append(f"No equipment number column found in {file_type} file")

        # For master file, both type and description should be available
        if file_type == "master":
            if 'equipment_type' not in column_mapping:
                issues.append("Master file missing equipment type column")
            if 'equipment_description' not in column_mapping:
                issues.append("Master file missing equipment description column")

        return len(issues) == 0, issues


class DataLoader:
    """Loads and validates data files in various formats"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or default_logger

    @log_execution_time()
    def load_file(self, file_path: str) -> pd.DataFrame:
        """Load file based on extension"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = path.suffix.lower()

        if file_extension not in settings.processing.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")

        self.logger.info(f"Loading file: {path.name}")

        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path, encoding='utf-8')
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unexpected file format: {file_extension}")

            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            self.logger.error(f"Failed to load {path.name}: {str(e)}")
            raise

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using validator mapping"""
        column_mapping = DataValidator.find_column_mapping(df)

        # Create reverse mapping for renaming
        rename_mapping = {}
        for standard_name, original_name in column_mapping.items():
            rename_mapping[original_name] = standard_name

        if rename_mapping:
            df = df.rename(columns=rename_mapping)
            self.logger.info(f"Standardized columns: {rename_mapping}")

        return df

    @log_execution_time()
    def load_master_data(self, master_file: str) -> pd.DataFrame:
        """Load and validate master data file"""
        df = self.load_file(master_file)

        # Validate master data
        is_valid, issues = DataValidator.validate_dataframe(df, "master")
        if not is_valid:
            raise ValueError(f"Master data validation failed: {', '.join(issues)}")

        # Standardize column names
        df = self.standardize_columns(df)

        # Remove rows with missing critical data in master
        initial_rows = len(df)
        df = df.dropna(subset=['equipment_number'])

        # For master data, we need either type or description
        df = df.dropna(subset=['equipment_type', 'equipment_description'], how='all')

        final_rows = len(df)
        if final_rows < initial_rows:
            self.logger.warning(f"Removed {initial_rows - final_rows} incomplete rows from master data")

        self.logger.info(f"Master data loaded: {len(df)} valid equipment records")
        return df

    @log_execution_time()
    def load_target_data(self, target_file: str) -> pd.DataFrame:
        """Load and validate target data file"""
        df = self.load_file(target_file)

        # Validate target data (less strict than master)
        is_valid, issues = DataValidator.validate_dataframe(df, "target")
        if not is_valid:
            raise ValueError(f"Target data validation failed: {', '.join(issues)}")

        # Standardize column names
        df = self.standardize_columns(df)

        # Ensure required columns exist (fill with NaN if missing)
        required_columns = ['equipment_number', 'equipment_type', 'equipment_description']
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
                self.logger.info(f"Added missing column '{col}' to target data")

        # Remove rows without equipment numbers
        initial_rows = len(df)
        df = df.dropna(subset=['equipment_number'])
        final_rows = len(df)

        if final_rows < initial_rows:
            self.logger.warning(f"Removed {initial_rows - final_rows} rows without equipment numbers")

        self.logger.info(f"Target data loaded: {len(df)} equipment records")
        return df
