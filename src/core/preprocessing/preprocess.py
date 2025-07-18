# automate_processing.py

import pandas as pd
import numpy as np
import yaml
import re
import os

def clean_initial_names(df):
    """Cleans the initial messy column names from the raw file."""
    cols = [re.sub(r'[^a-z0-9_]+', '_', col.lower()).strip('_') for col in df.columns]
    df.columns = [re.sub(r'_+', '_', col) for col in cols]
    return df

def run_automated_pipeline(config_path='src\config\preprocessing\preprocess.yaml'):
    """
    Runs a full, automated pre-processing pipeline based on a config file.
    """
    print("🚀 Starting Automated Pre-processing Pipeline...")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- Load Data and Configs ---
    print(f"   - Loading data from '{cfg['files']['input_file']}'...")
    df = pd.read_csv(cfg['files']['input_file'], low_memory=False)
    df = clean_initial_names(df)
    
    rules = cfg['processing_rules']
    rename_map = cfg['column_rename_map']
    transforms = cfg['transformations']

    # --- Step 1: Automatically Drop Columns with High Missing Values ---
    print("\n--- Step 1: Removing Sparse Columns ---")
    threshold = rules['missing_value_threshold']
    missing_ratios = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratios[missing_ratios > threshold].index.tolist()
    
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"   - Dropped {len(cols_to_drop)} columns with >{threshold:.0%} missing data.")
    print(f"   - {df.shape[1]} columns remain.")

    # --- Step 2: Rename Remaining Columns to Logical Names ---
    print("\n--- Step 2: Applying Logical Column Names ---")
    df.rename(columns=rename_map, inplace=True)
    print("   - Columns renamed.")

    # --- Step 3: Apply Data Type Corrections and Transformations ---
    print("\n--- Step 3: Cleaning and Transforming Data ---")

    # Convert specified columns to category
    for col in transforms['force_as_category']:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"   - Converted '{col}' to category.")

    # Apply log transform to specified skewed columns
    for col in transforms['apply_log_transform']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
            print(f"   - Applied log transform to '{col}'.")

    # Engineer features from date columns
    for col in transforms['engineer_from_date']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df.drop(columns=[col], inplace=True) # Drop original date column
            print(f"   - Engineered Year/Month features from '{col}'.")

    # --- Final Step: Save Processed Data ---
    output_path = cfg['files']['output_file']
    df.to_csv(output_path, index=False)
    print(f"\n✅ Pipeline complete! Final processed data saved to '{output_path}'.")
    print(f"   - Final dataset shape: {df.shape}")

if __name__ == "__main__":
    # Update this path to where you save your new config file
    run_automated_pipeline(config_path='src\config\preprocessing\preprocess.yaml')