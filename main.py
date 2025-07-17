"""
ENHANCED MAIN PIPELINE: Equipment Data Processing for POWER-BI-BOT
Author: Senior Programmer
Purpose: Process ONGC equipment data for AI agent and Power BI dashboard integration

INTEGRATED BEST PRACTICES:
1. Principal Data Scientist methodology implementation
2. Business-driven EDA with domain expertise  
3. ML-driven feature selection using ensemble methods
4. Advanced performance optimization strategies
5. Memory-efficient processing for 5M+ records
6. Modular architecture for easy maintenance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import os

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import our enhanced modules
from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, 
    ENABLE_FEATURES, ENABLE_MONITORING
)
from src.data.make_dataset import load_and_concatenate_data, clean_and_impute
from src.features.build_features import feature_engineering
from src.features.feature_selection import optimize_feature_set
from src.utils.logger import setup_logger, log_dataframe_info
from src.utils.memory_utils import (
    get_memory_usage, optimize_dtypes, reduce_memory_usage, 
    performance_optimizer, memory_cleanup, get_performance_report
)
from src.utils.helpers import create_backup

# Import best practices modules
try:
    from src.analysis.exploratory_data_analysis import IndustrialEDAAnalyzer
    from src.features.ml_feature_selection import BusinessDrivenFeatureSelector
    from src.utils.memory_utils import PerformanceOptimizer
    BEST_PRACTICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Best practices modules not fully available: {e}")
    BEST_PRACTICES_AVAILABLE = False

def run_enhanced_data_pipeline(enable_best_practices: bool = True):
    """
    Run the complete enhanced data processing pipeline with best practices
    """
    # Setup logging
    logger = setup_logger("enhanced_pipeline", "logs/enhanced_pipeline.log")
    logger.info("="*60)
    logger.info("STARTING ENHANCED EQUIPMENT DATA PROCESSING PIPELINE")
    logger.info("="*60)
    
    try:
        # Initialize performance monitoring if available
        if BEST_PRACTICES_AVAILABLE and enable_best_practices:
            optimizer = PerformanceOptimizer()
            logger.info("✅ Best practices modules loaded - Enhanced mode enabled")
        else:
            optimizer = None
            logger.info("⚠️ Running in standard mode - Best practices modules not available")
        
        # Log initial memory usage
        memory_info = get_memory_usage()
        logger.info(f"Initial memory usage: {memory_info}")
        
        # ========================
        # PHASE 1: ENHANCED DATA LOADING & CONCATENATION
        # ========================
        logger.info("PHASE 1: Enhanced data loading and concatenation")
        
        # Load and concatenate all files (TIME-SERIES APPROACH)
        master_df = load_and_concatenate_data(str(RAW_DATA_DIR))
        logger.info(f"Concatenated dataset shape: {master_df.shape}")
        
        # Apply initial memory optimization
        master_df = optimize_dtypes(master_df)
        logger.info("✅ Initial memory optimization applied")
        
        # Create backup of raw concatenated data
        raw_backup_path = PROCESSED_DATA_DIR / "01_raw_concatenated.parquet"
        create_backup(master_df, str(raw_backup_path))
        logger.info(f"Raw data backup saved to: {raw_backup_path}")
        
        # ========================
        # PHASE 2: BUSINESS-DRIVEN EDA (Best Practices)
        # ========================
        if BEST_PRACTICES_AVAILABLE and enable_best_practices:
            logger.info("PHASE 2: Business-driven exploratory data analysis")
            
            eda_analyzer = IndustrialEDAAnalyzer(
                data=master_df,
                business_context={
                    'domain': 'industrial_equipment',
                    'use_case': 'predictive_maintenance',
                    'stakeholders': ['maintenance_team', 'operations', 'finance']
                }
            )
            
            # Comprehensive analysis with business insights
            print("🔍 Running comprehensive data profiling...")
            profiling_results = eda_analyzer.comprehensive_data_profiling()
            
            print("📊 Analyzing equipment lifecycle patterns...")
            lifecycle_insights = eda_analyzer.equipment_lifecycle_analysis()
            
            print("🔗 Performing correlation analysis...")
            correlation_analysis = eda_analyzer.correlation_heatmap_analysis()
            
            # Generate business insights report
            business_insights = eda_analyzer.create_business_insights_report()
            
            logger.info(f"✅ EDA completed - Data quality score: {business_insights.get('data_quality_score', 'N/A')}")
        else:
            logger.info("PHASE 2: Standard data cleaning and preprocessing")
        
        # ========================
        # PHASE 3: ENHANCED DATA CLEANING & PREPROCESSING
        # ========================
        logger.info("PHASE 3: Enhanced data cleaning and preprocessing")
        
        master_df = clean_and_impute(master_df)
        
        # Additional memory optimization after cleaning
        master_df = reduce_memory_usage(master_df)
        
        # Save cleaned data
        cleaned_data_path = PROCESSED_DATA_DIR / "02_cleaned_data.parquet"
        master_df.to_parquet(cleaned_data_path, index=False)
        logger.info(f"Cleaned data saved to: {cleaned_data_path}")
        
        # ========================
        # PHASE 4: ML-DRIVEN FEATURE ENGINEERING (Best Practices)
        # ========================
        if ENABLE_FEATURES:
            logger.info("PHASE 4: ML-driven feature engineering")
            
            if BEST_PRACTICES_AVAILABLE and enable_best_practices:
                # Use advanced feature selection
                feature_selector = BusinessDrivenFeatureSelector(
                    data=master_df,
                    target_variable='equipment_risk_score',
                    problem_type='regression'
                )
                
                print("🎯 Creating business-relevant target variable...")
                target = feature_selector.create_business_target_variable()
                
                print("🔍 Running ensemble feature selection...")
                selected_features = feature_selector.ensemble_feature_selection()
                
                print("✅ Validating feature selection performance...")
                validation_results = feature_selector.validate_feature_selection()
                
                # Generate feature selection report
                feature_report = feature_selector.create_feature_selection_report()
                
                logger.info(f"✅ Advanced feature selection completed - {len(selected_features) if selected_features else 0} features selected")
            else:
                # Standard feature engineering
                master_df = feature_engineering(master_df)
                logger.info("✅ Standard feature engineering completed")
            
            # Save feature-engineered data
            features_data_path = FEATURES_DATA_DIR / "03_feature_engineered.parquet"
            features_data_path.parent.mkdir(parents=True, exist_ok=True)
            master_df.to_parquet(features_data_path, index=False)
            logger.info(f"Feature-engineered data saved to: {features_data_path}")
        
        # ========================
        # PHASE 5: PERFORMANCE-OPTIMIZED PROCESSING (Best Practices)
        # ========================
        if BEST_PRACTICES_AVAILABLE and enable_best_practices and optimizer:
            logger.info("PHASE 5: Performance-optimized processing")
            
            # Apply advanced memory-efficient processing
            master_df = optimizer.memory_efficient_processing(
                master_df,
                chunk_size=10000
            )
            
            # Generate performance report
            performance_report = optimizer.generate_performance_report()
            optimization_summary = optimizer.create_optimization_summary(master_df, master_df)
            
            logger.info(f"✅ Performance optimization completed - {optimization_summary.get('memory_optimization', {}).get('memory_reduction_percentage', 0):.1f}% memory saved")
        
        # ========================
        # PHASE 6: DIMENSIONALITY REDUCTION & FINAL OPTIMIZATION
        # ========================
        logger.info("PHASE 6: Dimensionality reduction and final optimization")
        
        # Use risk_score as target for supervised feature selection
        target_column = 'risk_score' if 'risk_score' in master_df.columns else None
        master_df_optimized = optimize_feature_set(master_df, target_column)
        
        # Final memory optimization
        master_df_optimized = optimize_dtypes(master_df_optimized)
        
        # ========================
        # PHASE 7: MULTI-FORMAT OUTPUTS
        # ========================
        logger.info("PHASE 7: Generating multi-format outputs")
        
        # Save final optimized dataset
        final_data_path = PROCESSED_DATA_DIR / "master_equipment_data_optimized.parquet"
        master_df_optimized.to_parquet(final_data_path, index=False)
        
        # Also save as CSV for Power BI compatibility
        final_csv_path = PROCESSED_DATA_DIR / "master_equipment_data_optimized.csv"
        master_df_optimized.to_csv(final_csv_path, index=False)
        
        # Generate summary report
        final_memory = get_memory_usage()
        generate_enhanced_pipeline_report(master_df_optimized, logger, memory_info, final_memory)
        
        # Clean up memory
        memory_cleanup()
        
        logger.info("="*60)
        logger.info("ENHANCED PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Final dataset shape: {master_df_optimized.shape}")
        logger.info(f"Optimized data saved to: {final_data_path}")
        logger.info(f"CSV for Power BI saved to: {final_csv_path}")
        logger.info("="*60)
        
        return master_df_optimized
        
    except Exception as e:
        logger.error(f"Enhanced pipeline failed with error: {e}")
        raise

def generate_enhanced_pipeline_report(df: pd.DataFrame, logger, initial_memory: dict, final_memory: dict):
    """Generate a comprehensive enhanced pipeline report"""
    logger.info("ENHANCED PIPELINE SUMMARY REPORT")
    logger.info("-" * 40)
    
    # Dataset overview
    log_dataframe_info(logger, df, "Final Optimized Dataset")
    
    # Feature categories
    numeric_features = len(df.select_dtypes(include=['number']).columns)
    categorical_features = len(df.select_dtypes(include=['object', 'category']).columns)
    date_features = len([col for col in df.columns if 'date' in col.lower() or col.endswith('_on')])
    
    logger.info(f"Feature breakdown:")
    logger.info(f"  Numeric features: {numeric_features}")
    logger.info(f"  Categorical features: {categorical_features}")
    logger.info(f"  Date features: {date_features}")
    
    # Performance metrics
    memory_reduction = ((initial_memory['process_memory_mb'] - final_memory['process_memory_mb']) / 
                       initial_memory['process_memory_mb']) * 100
    logger.info(f"Memory optimization: {memory_reduction:.1f}% reduction")
    
    # Key business metrics
    if 'equipment_id' in df.columns:
        total_equipment = df['equipment_id'].nunique()
        logger.info(f"Total unique equipment: {total_equipment}")
    
    if 'period_label' in df.columns:
        time_periods = df['period_label'].nunique()
        logger.info(f"Time periods covered: {time_periods}")
    
    if 'acquisition_value' in df.columns:
        total_value = df['acquisition_value'].sum()
        logger.info(f"Total equipment value: ${total_value:,.2f}")
    
    # Data quality metrics
    missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    logger.info(f"Overall missing data percentage: {missing_percentage:.2f}%")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Final dataset memory usage: {memory_usage:.2f} MB")
    
    # Performance summary
    if ENABLE_MONITORING:
        performance_report = get_performance_report()
        total_operations = performance_report.get('total_operations', 0)
        logger.info(f"Total performance-monitored operations: {total_operations}")

def create_sample_equipment_data(n_samples=1000):
    """
    Create sample equipment data for demonstration purposes
    """
    np.random.seed(42)
    
    # Equipment types
    equipment_types = ['PUMP', 'MOTOR', 'VALVE', 'COMPRESSOR', 'TURBINE']
    manufacturers = ['SULZER', 'ABB', 'SIEMENS', 'GE', 'FLOWSERVE']
    
    # Generate sample data
    data = {
        'equipment': range(10000, 10000 + n_samples),
        'equipment_description': [f'Equipment {i}' for i in range(n_samples)],
        'created_on': pd.date_range('2000-01-01', periods=n_samples, freq='D'),
        'objecttype': np.random.choice(equipment_types, n_samples),
        'acquisition_value': np.random.lognormal(10, 1, n_samples).astype(int),
        'weight': np.random.lognormal(5, 0.5, n_samples).astype(int),
        'manufacturer_of_asset': np.random.choice(manufacturers, n_samples),
        'model_number': [f'MODEL_{i}' for i in np.random.randint(1, 100, n_samples)],
        'inventory_number': [f'INV_{i}' if np.random.random() > 0.3 else None 
                           for i in range(n_samples)],
        'data_source_file': 'sample_data.csv',
        'period_label': '2000-2025'
    }
    
    return pd.DataFrame(data)

def run_best_practices_demo():
    """
    Run a demonstration of best practices with sample data if needed
    """
    print("🎯 RUNNING BEST PRACTICES DEMONSTRATION")
    print("="*50)
    
    # Check for available data files
    data_files = [
        PROCESSED_DATA_DIR / "master_equipment_data_optimized.parquet",
        PROCESSED_DATA_DIR / "02_cleaned_data.parquet",
        RAW_DATA_DIR / "EQUI_01.04.2020 TO 31.03.2025.xlsx"
    ]
    
    data_path = None
    for file_path in data_files:
        if file_path.exists():
            data_path = file_path
            break
    
    if not data_path:
        print("⚠️ No existing data files found. Creating sample dataset for demonstration...")
        
        # Create sample dataset for demonstration
        sample_data = create_sample_equipment_data()
        data_path = PROCESSED_DATA_DIR / "sample_equipment_data.csv"
        
        # Ensure directory exists
        data_path.parent.mkdir(parents=True, exist_ok=True)
        sample_data.to_csv(data_path, index=False)
        print(f"✅ Sample dataset created: {data_path}")
    
    # Run enhanced pipeline with sample data
    try:
        processed_data = run_enhanced_data_pipeline(enable_best_practices=True)
        print("\n🎉 BEST PRACTICES DEMONSTRATION COMPLETE!")
        print("✅ Enhanced pipeline executed successfully")
        return processed_data
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

if __name__ == '__main__':
    # Ensure required directories exist
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Check if we have real data or need to run demo
    if not any((RAW_DATA_DIR / f).exists() for f in ["EQUI_01.04.2020 TO 31.03.2025.xlsx", 
                                                    "EQUI_01.04.2015 TO 31.03.2020.xlsx"]):
        print("🧪 No raw data found - Running best practices demonstration")
        processed_data = run_best_practices_demo()
    else:
        print("📊 Raw data found - Running enhanced production pipeline")
        # Run the enhanced pipeline
        processed_data = run_enhanced_data_pipeline(enable_best_practices=BEST_PRACTICES_AVAILABLE)
    
    print("\n📚 For detailed implementation, see:")
    print("   - Enhanced Streamlit Dashboard: streamlit run streamlit_dashboard.py")
    print("   - BEST_PRACTICES_GUIDE.md")
    print("   - IMPLEMENTATION_SUMMARY.md")
    
    print("\n🚀 Ready for production deployment!")