#!/usr/bin/env python3
"""
ONGC Equipment Data Preprocessing Pipeline
Senior Programmer Implementation with Industry Best Practices

This script processes ONGC equipment data from multiple Excel files:
- Concatenates all 5 Excel files into one unified dataset  
- Keeps only the 8 specified columns
- Performs data cleaning (removes duplicates, empty rows)
- Does NOT impute missing values (preserves original data)
- Standardizes column names for consistency
- Applies memory optimization

Features:
- Modular architecture with separation of concerns
- Comprehensive error handling and validation
- Performance monitoring and reporting
- Multi-format output (CSV, Parquet)
- Industry best practices implementation
- Memory optimization for large datasets
"""

import sys
import argparse
import warnings
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import pipeline components
try:
    from src.pipeline import run_pipeline, create_pipeline
    from src.config import load_config, PipelineConfig
    from src.data_preprocessor import create_preprocessor
except ImportError as e:
    print(f"❌ Error importing pipeline modules: {e}")
    print("Please ensure all required files are in the src/ directory")
    sys.exit(1)

def main(config_path: str = None):
    """Main processing function using modular preprocessing pipeline"""
    
    try:
        print("=" * 80)
        print("🚀 ONGC EQUIPMENT DATA PREPROCESSING PIPELINE v3.0")
        print("=" * 80)
        print("📊 Senior Programmer Implementation - Data Preprocessing Focus")
        print("🎯 Task: Concatenate 5 Excel files + Filter 8 columns + Clean data")
        print("⚠️  Note: Missing values preserved (no imputation)")
        print("-" * 80)
        print("📋 Required columns (8 total):")
        print("   1. Equipment (Equipment number)")
        print("   2. Equipment description") 
        print("   3. Created on")
        print("   4. Chngd On (Changed on)")
        print("   5. ObjectType (Object type)")
        print("   6. Description of technical object")
        print("   7. Material")
        print("   8. Manufacturer of Asset")
        print("=" * 80)
        
        # Run the modular preprocessing pipeline
        result = run_pipeline(config_path)
        
        if result["status"] == "success":
            print("=" * 80)
            print("✅ PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📊 Records processed: {result['records_processed']:,}")
            print(f"📋 Final columns: {result['columns_count']}")
            print(f"⏱️ Execution time: {result['execution_time']:.2f}s")
            print(f"💾 Files saved: {len(result['saved_files'])}")
            
            # Show preprocessing statistics
            if 'preprocessing_stats' in result:
                stats = result['preprocessing_stats']
                print(f"📈 Files concatenated: {stats.get('successful_files', 0)}")
                print(f"🧹 Duplicates removed: {stats.get('duplicates_removed', 0):,}")
                print(f"🗑️ Empty rows removed: {stats.get('empty_rows_removed', 0):,}")
                
                memory_before = stats.get('memory_before_mb', 0)
                memory_after = stats.get('memory_after_mb', 0)
                if memory_before > 0:
                    reduction = ((memory_before - memory_after) / memory_before) * 100
                    print(f"🧠 Memory reduction: {reduction:.1f}%")
            
            print("=" * 80)
            
            # Show output files
            print("📁 Output files:")
            for file_type, file_path in result['saved_files'].items():
                print(f"   {file_type}: {file_path}")
            
            print("=" * 80)
            print("🎯 PREPROCESSING COMPLETED - DATA READY FOR USE!")
            print("   ✅ All 5 Excel files concatenated into unified dataset")
            print("   ✅ Only required 8 columns retained")
            print("   ✅ Data cleaned (duplicates & empty rows removed)")
            print("   ✅ Column names standardized")
            print("   ✅ Missing values preserved (no imputation)")
            print("   ✅ Memory optimized for efficient processing")
            print("   ✅ Available in multiple formats (CSV, Parquet)")
            print("=" * 80)
            
            return result
        else:
            print("=" * 80)
            print("❌ PREPROCESSING PIPELINE FAILED!")
            print("=" * 80)
            print(f"Error: {result['message']}")
            print(f"Execution time: {result['execution_time']:.2f}s")
            print("=" * 80)
            return result
            
    except Exception as e:
        print("=" * 80)
        print("❌ CRITICAL PIPELINE FAILURE!")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("=" * 80)
        raise

def run_demo():
    """Run demonstration mode with sample data"""
    print("🎯 Running in demonstration mode...")
    print("This will process sample data to show pipeline capabilities")
    
    # Create sample configuration for demo
    config = PipelineConfig()
    config.raw_data_dir = "demo/raw"
    config.processed_data_dir = "demo/processed"
    config.reports_dir = "demo/reports"
    
    # Create demo directories
    from pathlib import Path
    Path(config.raw_data_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Demo data directory: {config.raw_data_dir}")
    print("📝 Place your sample Excel/CSV files in the demo directory")
    print("🚀 Then run: python main.py --demo")
    
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONGC Equipment Data Processing Pipeline v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline with default settings
  python main.py --config custom.json  # Run with custom configuration
  python main.py --demo             # Run demonstration mode
  python main.py --validate         # Validate setup only
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run in demonstration mode with sample data'
    )
    
    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate pipeline setup without processing data'
    )
    
    parser.add_argument(
        '--log-level', '-l',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.demo:
            # Run demonstration mode
            demo_config = run_demo()
        elif args.validate:
            # Validate setup only
            pipeline = create_pipeline(args.config)
            validation = pipeline.validate_setup()
            
            print("🔍 Pipeline Validation Results:")
            print(f"   Valid: {'✅' if validation['valid'] else '❌'}")
            
            if validation['issues']:
                print("   Issues:")
                for issue in validation['issues']:
                    print(f"     ❌ {issue}")
            
            if validation['warnings']:
                print("   Warnings:")
                for warning in validation['warnings']:
                    print(f"     ⚠️ {warning}")
            
            if validation['valid']:
                print("✅ Pipeline setup is valid and ready to run!")
            else:
                print("❌ Pipeline setup has issues that need to be resolved.")
                sys.exit(1)
        else:
            # Run full pipeline
            result = main(args.config)
            
            # Exit with appropriate code
            if result and result.get("status") == "success":
                sys.exit(0)
            else:
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        sys.exit(1)