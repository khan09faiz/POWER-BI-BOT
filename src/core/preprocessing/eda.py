# src/core/preprocessing/eda.py

import pandas as pd
import numpy as np
import yaml
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

def clean_col_names(df):
    """Cleans column names to be Python-friendly."""
    cols = [re.sub(r'[^a-z0-9_]+', '_', col.lower()).strip('_') for col in df.columns]
    df.columns = [re.sub(r'_+', '_', col) for col in cols]
    return df

def run_eda():
    """
    Performs a focused EDA on important features and visualizes their relationships.
    """
    print("🚀 Starting Focused EDA...")
    
    # Corrected file path to avoid SyntaxWarning
    with open('src/config/preprocessing/eda.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # --- Setup ---
    plt.style.use(cfg['plot_style'])
    plot_dir = cfg['plot_output_directory']
    os.makedirs(plot_dir, exist_ok=True)
    print(f"   - Plots will be saved to '{plot_dir}'.")

    # --- Load Data ---
    try:
        # Using low_memory=False can help with mixed data types in large files
        df = pd.read_csv(cfg['data_file_path'], low_memory=False)
        df = clean_col_names(df)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # --- 1. Filter for Important Features ---
    print("\n--- Filtering for Important Features ---")
    missing_threshold = cfg['analysis_thresholds']['missing_value']
    missing_ratios = df.isnull().sum() / len(df)
    cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index
    
    df_filtered = df.drop(columns=cols_to_drop)
    print(f"   - Dropped {len(cols_to_drop)} columns with >{missing_threshold:.0%} missing values.")
    print(f"   - {len(df_filtered.columns)} columns remaining for analysis.")

    numerical_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- 2. Univariate Analysis (Understanding Individual Features) ---
    print("\n--- Analyzing Individual Feature Distributions ---")
    for col in numerical_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df_filtered[col], kde=True).set_title(f'Distribution of {col}')
        plt.savefig(os.path.join(plot_dir, f'univariate_hist_{col}.png'))
        plt.close()

    # --- 3. Bivariate Analysis (Understanding Feature Relationships) ---
    print("\n--- Analyzing Feature Relationships ---")

    # Correlation Heatmap (Numerical vs. Numerical)
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_filtered[numerical_cols].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Numerical Feature Correlation')
        plt.savefig(os.path.join(plot_dir, 'bivariate_correlation_heatmap.png'), bbox_inches='tight')
        plt.close()

    # Pair Plot (Detailed look at a few key numerical features)
    if cfg['relationship_plots']['enable_pairplot']:
        pairplot_cols = [col for col in cfg['relationship_plots']['pairplot_features'] if col in df_filtered.columns]
        if len(pairplot_cols) > 1:
            sns.pairplot(df_filtered[pairplot_cols].dropna())
            plt.savefig(os.path.join(plot_dir, 'bivariate_pairplot.png'))
            plt.close()
            
    # Box Plots (Categorical vs. Numerical)
    if numerical_cols and categorical_cols:
        num_col_example = 'weight' if 'weight' in numerical_cols else numerical_cols[0]
        cat_col_example = 'plnt' if 'plnt' in categorical_cols else categorical_cols[0]
        
        if cat_col_example in df_filtered.columns and num_col_example in df_filtered.columns:
            plt.figure(figsize=(12, 8))
            order = df_filtered[cat_col_example].value_counts().nlargest(15).index
            sns.boxplot(x=cat_col_example, y=num_col_example, data=df_filtered[df_filtered[cat_col_example].isin(order)])
            plt.xticks(rotation=45)
            plt.title(f'"{num_col_example}" across Top 15 Categories of "{cat_col_example}"')
            plt.savefig(os.path.join(plot_dir, f'bivariate_{cat_col_example}_vs_{num_col_example}.png'), bbox_inches='tight')
            plt.close()

    # Crosstab Heatmap (Categorical vs. Categorical)
    if 'plnt' in df_filtered.columns and 'pgrp' in df_filtered.columns:
        print("   - Generating crosstab heatmap for Plant vs. Purchasing Group...")
        top_plnt = df_filtered['plnt'].value_counts().nlargest(15).index
        top_pgrp = df_filtered['pgrp'].value_counts().nlargest(15).index
        
        filtered_for_crosstab = df_filtered[df_filtered['plnt'].isin(top_plnt) & df_filtered['pgrp'].isin(top_pgrp)]
        
        if not filtered_for_crosstab.empty:
            crosstab = pd.crosstab(filtered_for_crosstab['plnt'], filtered_for_crosstab['pgrp'])
            plt.figure(figsize=(14, 10))
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='viridis')
            plt.title('Relationship between Top Plants and Purchasing Groups')
            plt.savefig(os.path.join(plot_dir, 'bivariate_crosstab_plnt_vs_pgrp.png'), bbox_inches='tight')
            plt.close()

    print(f"\n✨ Focused EDA script finished. Check the '{plot_dir}' folder.")

if __name__ == "__main__":
    run_eda()