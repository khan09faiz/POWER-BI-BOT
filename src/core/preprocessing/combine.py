import pandas as pd
import yaml
import glob
import os

def combine_files():
    print("🚀 Starting file combination...")
    with open('src\config\preprocessing\combine.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    file_pattern = os.path.join(cfg['data_folder_path'], cfg['file_extension'])
    file_list = glob.glob(file_pattern)

    if not file_list:
        print(f"❌ No files found matching '{file_pattern}'. Check combine.yaml.")
        return

    df_list = []
    for filename in file_list:
        print(f"  - Reading: {os.path.basename(filename)}")
        try:
            if cfg['file_extension'].endswith('.xlsx'):
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {filename}: {e}")
            continue

    if not df_list:
        print("❌ No data was loaded. Aborting.")
        return

    print("\n🔗 Concatenating dataframes...")
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"✅ Combination complete! Total rows: {len(combined_df)}")

    # Save the combined dataframe
    print(f"\n💾 Saving to CSV: {cfg['output_csv']}")
    combined_df.to_csv(cfg['output_csv'], index=False)
    
    print(f"💾 Saving to Parquet: {cfg['output_parquet']}")
    try:
        combined_df.to_parquet(cfg['output_parquet'], index=False)
    except ImportError:
        print("   ⚠️ Could not save to Parquet. Install 'pyarrow': pip install pyarrow")
    
    print("\n✨ Combination script finished!")

if __name__ == "__main__":
    combine_files()