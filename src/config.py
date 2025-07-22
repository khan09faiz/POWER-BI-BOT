"""
Configuration Management for ONGC Equipment Data Processing Pipeline
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
import json

@dataclass
class PipelineConfig:
    """Configuration class for the data processing pipeline"""
    
    # Data directories
    raw_data_dir: str = "dataset/Raw"
    processed_data_dir: str = "dataset/Processed"
    reports_dir: str = "reports"
    logs_dir: str = "logs"
    
    # Required columns to keep from source data (EXACT column names from Excel files)
    required_columns: List[str] = field(default_factory=lambda: [
        'Equipment',  # Equipment number
        'Equipment description',  # Equipment description
        'Created on',  # Created on
        'Chngd On',  # Changed on
        'ObjectType',  # Object type
        'Description of technical object',  # Description of technical object
        'Material',  # Material
        'Manufacturer of Asset'  # Manufacturer of Asset
    ])
    
    # Column name mappings for standardization
    column_mappings: Dict[str, str] = field(default_factory=lambda: {
        'Equipment': 'equipment_number',
        'Equipment description': 'equipment_description',
        'Created on': 'created_on',
        'Chngd On': 'changed_on',
        'ObjectType': 'object_type',
        'Description of technical object': 'technical_description',
        'Material': 'material',
        'Manufacturer of Asset': 'manufacturer'
    })
    
    # Expected data files
    expected_files: List[str] = field(default_factory=lambda: [
        "EQUI_01.04.2000 TO 31.03.2005.xlsx",
        "EQUI_01.04.2005 TO 31.03.2010.xlsx", 
        "EQUI_01.04.2010 TO 31.03.2015.xlsx",
        "EQUI_01.04.2015 TO 31.03.2020.xlsx",
        "EQUI_01.04.2020 TO 31.03.2025.xlsx"
    ])
    
    # Processing settings
    memory_optimization: bool = True
    remove_duplicates: bool = True
    remove_empty_rows: bool = True
    auto_discover_files: bool = True
    
    # Data cleaning settings (no feature engineering)
    clean_only: bool = True
    preserve_missing_data: bool = True  # Keep missing data for future semantic search
    
    # Output settings
    save_csv: bool = True
    save_parquet: bool = True
    save_latest_version: bool = True
    
    # Performance settings
    chunk_size: int = 10000
    max_memory_usage_mb: int = 4096
    
    # Logging settings
    log_level: str = "INFO"
    enable_performance_logging: bool = True
    
    # Processing optimization settings
    enable_circuit_breaker: bool = True
    max_retry_attempts: int = 3
    
    # Memory optimization
    target_memory_reduction: float = 0.35  # Target 35% memory reduction
    use_nullable_dtypes: bool = True
    aggressive_categorization: bool = True
    
    # Circuit breaker settings
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: int = 60
    
    # Parallel processing settings
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        """Create directories after initialization"""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.raw_data_dir,
            self.processed_data_dir,
            self.reports_dir,
            self.logs_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary using dataclass fields"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        config_path = Path(file_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'PipelineConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Global configuration instance
_config = None

def get_config() -> PipelineConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = PipelineConfig()
    return _config

def set_config(config: PipelineConfig):
    """Set global configuration instance"""
    global _config
    _config = config

def load_config(config_path: str = None) -> PipelineConfig:
    """Load configuration from file or create default"""
    if config_path and Path(config_path).exists():
        config = PipelineConfig.load_from_file(config_path)
    else:
        config = PipelineConfig()
        
        # Save default config for future reference
        if config_path:
            config.save_to_file(config_path)
    
    set_config(config)
    return config