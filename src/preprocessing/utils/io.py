"""
I/O utilities for loading and saving data
Simple, reliable file operations
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def load_excel_file(file_path: Path) -> pd.DataFrame:
    """
    Load a single Excel file with error handling
    """
    try:
        df = pd.read_excel(file_path, dtype=str)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load {file_path.name}: {e}")


def discover_excel_files(directory: Path) -> List[Path]:
    """
    Find all Excel files in directory
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    excel_files = list(directory.glob("*.xlsx")) + list(directory.glob("*.xls"))

    if not excel_files:
        raise FileNotFoundError(f"No Excel files found in {directory}")

    return sorted(excel_files)


def save_csv(df: pd.DataFrame, output_path: Path) -> str:
    """
    Save DataFrame as CSV
    """
    df.to_csv(output_path, index=False)
    return str(output_path)


def save_parquet(df: pd.DataFrame, output_path: Path) -> str:
    """
    Save DataFrame as Parquet (requires pyarrow)
    """
    try:
        df.to_parquet(output_path, index=False)
        return str(output_path)
    except ImportError:
        raise ImportError("Parquet saving requires pyarrow: pip install pyarrow")


def save_json_report(data: Dict[str, Any], output_path: Path) -> str:
    """
    Save dictionary as JSON report
    """
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    return str(output_path)


def get_timestamp() -> str:
    """Get timestamp string for file naming"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_output_paths(base_dir: Path, prefix: str) -> Dict[str, Path]:
    """
    Create standardized output file paths
    """
    timestamp = get_timestamp()

    return {
        'csv': base_dir / f"{prefix}_{timestamp}.csv",
        'parquet': base_dir / f"{prefix}_{timestamp}.parquet",
        'report': base_dir / f"{prefix}_report_{timestamp}.json"
    }
