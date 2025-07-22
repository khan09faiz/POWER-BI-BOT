"""
Data Saving utilities for ONGC Equipment Data Processing Pipeline
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging
from .config import get_config
from .logger import monitor_performance

logger = logging.getLogger('ongc_pipeline')

class DataSaver:
    """Data saving and export utilities"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
    
    @monitor_performance
    def save_processed_data(self, df: pd.DataFrame, summary: Dict[str, Any]) -> Dict[str, str]:
        """Save processed data in multiple formats"""
        if df.empty:
            logger.error("❌ No data to save")
            return {}
        
        logger.info("💾 Saving processed data")
        
        # Create output directories
        processed_dir = Path(self.config.processed_data_dir)
        reports_dir = Path(self.config.reports_dir)
        
        processed_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save as CSV
        if self.config.save_csv:
            csv_path = processed_dir / f"ongc_equipment_data_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            saved_files["csv"] = str(csv_path)
            logger.info(f"   ✅ Saved CSV: {csv_path.name}")
        
        # Save as Parquet (more efficient)
        if self.config.save_parquet:
            try:
                parquet_path = processed_dir / f"ongc_equipment_data_{timestamp}.parquet"
                df.to_parquet(parquet_path, index=False)
                saved_files["parquet"] = str(parquet_path)
                logger.info(f"   ✅ Saved Parquet: {parquet_path.name}")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not save Parquet format: {e}")
        
        # Save latest versions (overwrite)
        if self.config.save_latest_version:
            if self.config.save_csv:
                latest_csv = processed_dir / "ongc_equipment_data_latest.csv"
                df.to_csv(latest_csv, index=False)
                saved_files["latest_csv"] = str(latest_csv)
                logger.info(f"   ✅ Saved latest CSV: {latest_csv.name}")
            
            if self.config.save_parquet:
                try:
                    latest_parquet = processed_dir / "ongc_equipment_data_latest.parquet"
                    df.to_parquet(latest_parquet, index=False)
                    saved_files["latest_parquet"] = str(latest_parquet)
                    logger.info(f"   ✅ Saved latest Parquet: {latest_parquet.name}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Could not save latest Parquet: {e}")
        
        # Save summary report
        summary_path = reports_dir / f"data_summary_{timestamp}.json"
        self.save_summary_report(summary, summary_path)
        saved_files["summary"] = str(summary_path)
        
        # Save data dictionary
        data_dict_path = reports_dir / f"data_dictionary_{timestamp}.json"
        data_dictionary = self.create_data_dictionary(df)
        self.save_data_dictionary(data_dictionary, data_dict_path)
        saved_files["data_dictionary"] = str(data_dict_path)
        
        logger.info(f"💾 Saved {len(saved_files)} files successfully")
        
        return saved_files
    
    @monitor_performance
    def save_summary_report(self, summary: Dict[str, Any], file_path: Path):
        """Save data summary report as JSON"""
        try:
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"   ✅ Saved summary report: {file_path.name}")
        except Exception as e:
            logger.error(f"   ❌ Failed to save summary report: {e}")
    
    def create_data_dictionary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a data dictionary describing the dataset"""
        data_dict = {
            "dataset_info": {
                "name": "ONGC Equipment Data",
                "description": "Processed equipment data from ONGC with standardized columns",
                "total_records": len(df),
                "total_columns": len(df.columns),
                "creation_date": datetime.now().isoformat()
            },
            "columns": {}
        }
        
        # Column descriptions
        column_descriptions = {
            "equipment_number": "Unique identifier for each piece of equipment",
            "equipment_description": "Descriptive name or title of the equipment",
            "created_on": "Date when the equipment record was created",
            "changed_on": "Date when the equipment record was last modified",
            "object_type": "Type or category of the equipment",
            "technical_description": "Detailed technical description of the equipment",
            "material": "Material composition or specification",
            "manufacturer": "Company that manufactured the equipment",
            "data_source_file": "Original file from which this record was extracted",
            "period_label": "Time period label extracted from the source filename"
        }
        
        # Build column information
        for col in df.columns:
            col_info = {
                "data_type": str(df[col].dtype),
                "non_null_count": int(df[col].count()),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": round(float(df[col].isnull().sum() / len(df) * 100), 2),
                "description": column_descriptions.get(col, "No description available")
            }
            
            # Add unique value count for categorical columns
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                col_info["unique_values"] = int(df[col].nunique())
                
                # Add sample values for categorical columns
                if df[col].nunique() <= 20:
                    sample_values = df[col].dropna().unique()[:10].tolist()
                    col_info["sample_values"] = [str(v) for v in sample_values]
            
            # Add statistics for numeric columns
            elif df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                if df[col].count() > 0:
                    col_info.update({
                        "min_value": float(df[col].min()) if pd.notna(df[col].min()) else None,
                        "max_value": float(df[col].max()) if pd.notna(df[col].max()) else None,
                        "mean_value": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                        "median_value": float(df[col].median()) if pd.notna(df[col].median()) else None
                    })
            
            data_dict["columns"][col] = col_info
        
        return data_dict
    
    @monitor_performance
    def save_data_dictionary(self, data_dict: Dict[str, Any], file_path: Path):
        """Save data dictionary as JSON"""
        try:
            with open(file_path, 'w') as f:
                json.dump(data_dict, f, indent=2, default=str)
            logger.info(f"   ✅ Saved data dictionary: {file_path.name}")
        except Exception as e:
            logger.error(f"   ❌ Failed to save data dictionary: {e}")
    
    @monitor_performance
    def create_processing_report(self, 
                               files_processed: List[str],
                               processing_summary: Dict[str, Any],
                               performance_metrics: Dict[str, Any],
                               saved_files: Dict[str, str]) -> Dict[str, Any]:
        """Create comprehensive processing report"""
        
        report = {
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_version": "2.0.0",
                "files_processed": files_processed,
                "total_files": len(files_processed)
            },
            "data_summary": processing_summary,
            "performance_metrics": performance_metrics,
            "output_files": saved_files,
            "configuration": self.config.to_dict()
        }
        
        return report
    
    @monitor_performance
    def save_processing_report(self, report: Dict[str, Any]) -> str:
        """Save comprehensive processing report"""
        reports_dir = Path(self.config.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"processing_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Saved processing report: {report_path.name}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save processing report: {e}")
            return ""

def create_data_saver(config=None) -> DataSaver:
    """Factory function to create a DataSaver instance"""
    return DataSaver(config)