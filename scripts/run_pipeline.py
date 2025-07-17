"""
Run the complete data processing pipeline
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from main import run_data_pipeline

if __name__ == "__main__":
    print("Starting POWER-BI-BOT Data Processing Pipeline...")
    print("="*60)
    
    try:
        # Run the pipeline
        result_df = run_data_pipeline()
        
        print("Pipeline completed successfully!")
        print(f"Final dataset shape: {result_df.shape}")
        print(f"Columns: {result_df.columns.tolist()}")
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
