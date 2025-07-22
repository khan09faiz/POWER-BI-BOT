"""
Central configuration for ONGC Equipment Data Pipeline
Clean, simple configuration management
"""
from pathlib import Path
from typing import List, Dict


class Config:
    """Simple configuration class"""

    # Directories
    RAW_DATA_DIR = "dataset/raw"
    OUTPUT_DIR = "dataset/processed"
    REPORTS_DIR = "reports"
    LOGS_DIR = "src/logs"

    # Required columns (as they appear in Excel files)
    REQUIRED_COLUMNS = [
        'Equipment',
        'Equipment description',
        'Created on',
        'Chngd On',
        'ObjectType',
        'Manufacturer of Asset'
    ]

    # Column name mappings
    COLUMN_MAPPINGS = {
        'Equipment': 'equipment_number',
        'Equipment description': 'Equipment Description',
        'Created on': 'created_on',
        'Chngd On': 'changed_on',
        'ObjectType': 'Equipment Type',
        'Manufacturer of Asset': 'manufacturer'
    }

    # Processing settings
    REMOVE_DUPLICATES = True
    REMOVE_EMPTY_ROWS = False
    OPTIMIZE_MEMORY = True
    SAVE_CSV = True
    SAVE_PARQUET = True

    @classmethod
    def create_directories(cls):
        """Create all required directories"""
        for directory in [cls.RAW_DATA_DIR, cls.OUTPUT_DIR, cls.REPORTS_DIR, cls.LOGS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_setup(cls) -> bool:
        """Validate that required files exist"""
        raw_dir = Path(cls.RAW_DATA_DIR)
        if not raw_dir.exists():
            return False

        excel_files = list(raw_dir.glob("*.xlsx"))
        return len(excel_files) > 0


# Global config instance
config = Config()
