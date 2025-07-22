#!/usr/bin/env python3
"""
ONGC Equipment Data Preprocessing Pipeline - Entry Point
Simple, clean CLI for running the complete preprocessing pipeline
"""
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import config, logger
from steps.preprocess_main import run_preprocessing_pipeline, get_pipeline_info


def main():
    """Main entry point for the preprocessing pipeline"""
    try:
        # Run the complete preprocessing pipeline
        results = run_preprocessing_pipeline()

        if results['status'] == 'success':
            print("\n" + "="*60)
            print("  PREPROCESSING COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"  Final Records: {results['final_records']:,}")
            print(f"  Final Columns: {results['final_columns']}")

            # Show saved files
            saved_files = results['statistics'].get('saving', {}).get('saved_files', {})
            if saved_files:
                print("\n  Output Files:")
                for file_type, file_path in saved_files.items():
                    print(f"    {file_type.upper()}: {Path(file_path).name}")

            print("="*60)
            return 0
        else:
            print(f"\nERROR: {results['message']}")
            return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"UNEXPECTED ERROR: {e}")
        return 1


def validate_setup():
    """Validate pipeline setup and configuration"""
    print("="*60)
    print("  SETUP VALIDATION")
    print("="*60)

    # Check configuration
    if not config.validate_setup():
        print("❌ Setup validation failed")
        print(f"   - Check raw data directory: {config.RAW_DATA_DIR}")
        print("   - Ensure Excel files are present")
        return False

    # Check raw data files
    raw_dir = Path(config.RAW_DATA_DIR)
    excel_files = list(raw_dir.glob("*.xlsx"))

    print(f"✅ Raw data directory: {raw_dir}")
    print(f"✅ Excel files found: {len(excel_files)}")

    for file_path in excel_files[:5]:  # Show first 5 files
        print(f"   - {file_path.name}")

    if len(excel_files) > 5:
        print(f"   ... and {len(excel_files) - 5} more files")

    # Check output directories
    config.create_directories()
    print(f"✅ Output directories ready")
    print(f"   - Processed data: {config.OUTPUT_DIR}")
    print(f"   - Reports: {config.REPORTS_DIR}")
    print(f"   - Logs: {config.LOGS_DIR}")

    print("✅ Setup validation passed")
    return True


def show_info():
    """Display pipeline information"""
    info = get_pipeline_info()

    print("="*60)
    print("  PIPELINE INFORMATION")
    print("="*60)
    print(f"Version: {info['version']}")
    print(f"Required Columns: {len(info['required_columns'])}")

    for i, col in enumerate(info['required_columns'], 1):
        print(f"  {i:2d}. {col}")

    print(f"\nProcessing Options:")
    options = info['processing_options']
    for key, value in options.items():
        status = "✅" if value else "❌"
        print(f"  {status} {key.replace('_', ' ').title()}")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ONGC Equipment Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_preprocessing.py              # Run full pipeline
  python run_preprocessing.py --validate   # Validate setup only
  python run_preprocessing.py --info       # Show pipeline info
        """
    )

    parser.add_argument(
        '--validate', '-v',
        action='store_true',
        help='Validate setup without running pipeline'
    )

    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Show pipeline configuration information'
    )

    args = parser.parse_args()

    try:
        if args.validate:
            # Validate setup only
            if validate_setup():
                sys.exit(0)
            else:
                sys.exit(1)
        elif args.info:
            # Show pipeline information
            show_info()
            sys.exit(0)
        else:
            # Run full pipeline
            exit_code = main()
            sys.exit(exit_code)

    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        print(f"CLI ERROR: {e}")
        sys.exit(1)
