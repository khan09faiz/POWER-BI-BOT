# Configuration settings for POWER-BI-BOT

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
EXPORTS_DATA_DIR = DATA_DIR / "exports"

# File paths
RAW_FILES = [
    "EQUI_01.04.2000 TO 31.03.2005.xlsx",
    "EQUI_01.04.2005 TO 31.03.2010.xlsx", 
    "EQUI_01.04.2010 TO 31.03.2015.xlsx",
    "EQUI_01.04.2015 TO 31.03.2020.xlsx",
    "EQUI_01.04.2020 TO 31.03.2025.xlsx"
]

# Processing settings
CHUNK_SIZE = 10000  # Process data in chunks for memory efficiency
MAX_MISSING_THRESHOLD = 0.6  # Drop columns with >60% missing values
REFERENCE_DATE = "2025-07-17"  # Fixed reference date for age calculations

# Feature engineering settings
CRITICAL_KEYWORDS = ['CRITICAL', 'ESSENTIAL', 'VITAL', 'EMERGENCY']
EQUIPMENT_TYPES = ['MIXER', 'MOTOR', 'PUMP', 'AERATOR', 'TANK', 'CHEMICAL', 'COMPRESSOR', 'GENERATOR']

# Dimensionality reduction settings
PCA_VARIANCE_THRESHOLD = 0.95  # Retain 95% of variance
CORRELATION_THRESHOLD = 0.9    # Remove features with correlation > 0.9

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Memory optimization and performance settings
LOW_MEMORY = True
OPTIMIZE_DTYPES = True
ENABLE_FEATURES = True  # Enable feature engineering
ENABLE_MONITORING = True  # Enable performance monitoring

# Export settings
EXPORT_FORMATS = ["csv", "parquet", "json"]
POWERBI_CHUNK_SIZE = 50000  # Chunk size for Power BI exports
