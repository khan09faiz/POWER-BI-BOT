import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
from pathlib import Path
import sys

# Import our custom utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import REFERENCE_DATE, CRITICAL_KEYWORDS, EQUIPMENT_TYPES
from src.utils.logger import setup_logger, log_dataframe_info, log_processing_step
from src.utils.memory_utils import optimize_dtypes
from src.utils.helpers import safe_convert_numeric

# Setup logger
logger = setup_logger(__name__, "logs/feature_engineering.log")

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    COMPREHENSIVE: Advanced feature engineering for equipment data analysis and AI agent queries
    """
    logger.info("Starting comprehensive feature engineering")
    initial_shape = df.shape
    
    # Make a copy to avoid modifying original
    df_features = df.copy()
    
    # === 1. TEMPORAL FEATURES ===
    df_features = create_temporal_features(df_features)
    
    # === 2. EQUIPMENT CLASSIFICATION FEATURES ===
    df_features = create_equipment_classification_features(df_features)
    
    # === 3. FINANCIAL FEATURES ===
    df_features = create_financial_features(df_features)
    
    # === 4. MAINTENANCE & LIFECYCLE FEATURES ===
    df_features = create_maintenance_features(df_features)
    
    # === 5. TEXT & CATEGORICAL FEATURES ===
    df_features = create_text_features(df_features)
    
    # === 6. AGGREGATION FEATURES ===
    df_features = create_aggregation_features(df_features)
    
    # === 7. RISK & PRIORITY FEATURES ===
    df_features = create_risk_priority_features(df_features)
    
    # Final optimization
    df_features = optimize_dtypes(df_features)
    
    final_shape = df_features.shape
    log_processing_step(logger, "Comprehensive Feature Engineering", initial_shape, final_shape)
    log_dataframe_info(logger, df_features, "Feature-Engineered Dataset")
    
    return df_features


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive temporal features for time-series analysis"""
    logger.info("Creating temporal features")
    
    reference_date = pd.to_datetime(REFERENCE_DATE)
    
    # Parse key date columns
    date_columns = ['created_on', 'changed_on', 'acquisition_date', 'warranty_end', 
                   'delivery_date', 'start_date', 'end_of_use_date']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Equipment age and lifecycle features
    if 'created_on' in df.columns:
        df['asset_age_years'] = (reference_date - df['created_on']).dt.days / 365.25
        df['asset_age_months'] = (reference_date - df['created_on']).dt.days / 30.44
        df['creation_year'] = df['created_on'].dt.year
        df['creation_month'] = df['created_on'].dt.month
        df['creation_quarter'] = df['created_on'].dt.quarter
        df['creation_day_of_year'] = df['created_on'].dt.dayofyear
        
        # Lifecycle stage based on age
        df['lifecycle_stage'] = pd.cut(df['asset_age_years'], 
                                     bins=[-np.inf, 1, 5, 10, 20, np.inf],
                                     labels=['New', 'Young', 'Mature', 'Aging', 'Legacy'])
    
    # Warranty analysis
    if 'warranty_end' in df.columns:
        df['warranty_status'] = np.where(df['warranty_end'] > reference_date, 'Active', 'Expired')
        df['warranty_days_remaining'] = (df['warranty_end'] - reference_date).dt.days
        df['warranty_duration_years'] = (df['warranty_end'] - df['created_on']).dt.days / 365.25
    
    # Time between key events
    if 'delivery_date' in df.columns and 'created_on' in df.columns:
        df['delivery_to_creation_days'] = (df['created_on'] - df['delivery_date']).dt.days
    
    # Period-based features (for time-series analysis)
    if 'period_label' in df.columns:
        df['period_numeric'] = df['period_label'].str.extract(r'(\d{4})').astype(float)
    
    return df


def create_equipment_classification_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create equipment classification and categorization features"""
    logger.info("Creating equipment classification features")
    
    if 'equipment_description' in df.columns:
        desc_col = df['equipment_description'].astype(str).str.upper()
        
        # Equipment type classification
        for eq_type in EQUIPMENT_TYPES:
            df[f'is_{eq_type.lower()}'] = desc_col.str.contains(eq_type, na=False).astype(int)
        
        # General equipment type
        equipment_pattern = '|'.join(EQUIPMENT_TYPES)
        df['equipment_type'] = desc_col.str.extract(f'({equipment_pattern})', expand=False).fillna('OTHER')
        
        # Size/capacity indicators
        df['has_capacity_info'] = desc_col.str.contains(r'\d+\s*(HP|KW|CFM|GPM|TON)', na=False).astype(int)
        df['description_length'] = df['equipment_description'].str.len()
        df['description_word_count'] = df['equipment_description'].str.split().str.len()
    
    # Class-based features
    if 'class_code' in df.columns:
        df['equipment_class_group'] = df['class_code'].astype(str).str[:2]  # First 2 digits
    
    return df


def create_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create financial analysis features"""
    logger.info("Creating financial features")
    
    # Convert financial columns to numeric
    financial_cols = ['acquisition_value', 'replacement_value', 'provision_fee', 'unload_costs']
    for col in financial_cols:
        if col in df.columns:
            df[col] = safe_convert_numeric(df[col])
    
    if 'acquisition_value' in df.columns:
        # Value categories
        df['value_category'] = pd.cut(df['acquisition_value'],
                                    bins=[0, 1000, 10000, 100000, np.inf],
                                    labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Cost per year of life
        if 'asset_age_years' in df.columns:
            df['cost_per_year'] = df['acquisition_value'] / np.maximum(df['asset_age_years'], 0.1)
    
    # Replacement analysis
    if 'replacement_value' in df.columns and 'acquisition_value' in df.columns:
        df['replacement_ratio'] = df['replacement_value'] / np.maximum(df['acquisition_value'], 1)
        df['value_appreciation'] = df['replacement_value'] - df['acquisition_value']
    
    return df


def create_maintenance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create maintenance and operational features"""
    logger.info("Creating maintenance features")
    
    # Determine the equipment ID column name
    equipment_col = None
    for col in ['equipment_id', 'equipment', 'equi_id']:
        if col in df.columns:
            equipment_col = col
            break
    
    # Maintenance plan analysis
    if 'maintenance_plan' in df.columns:
        df['has_maintenance_plan'] = (~df['maintenance_plan'].isnull()).astype(int)
        df['maintenance_plan_type'] = df['maintenance_plan'].astype(str).str[:3]  # First 3 chars
    
    # Plant and location features
    if 'field' in df.columns and equipment_col:
        df['plant_equipment_count'] = df.groupby('field')[equipment_col].transform('count')
        if 'asset_age_years' in df.columns:
            df['plant_avg_age'] = df.groupby('field')['asset_age_years'].transform('mean')
    
    # Manufacturer reliability indicators - check for actual column name
    manufacturer_col = None
    for col in ['manufacturer', 'manufacturer_of_asset']:
        if col in df.columns:
            manufacturer_col = col
            break
    
    if manufacturer_col and equipment_col:
        df['manufacturer_equipment_count'] = df.groupby(manufacturer_col)[equipment_col].transform('count')
        if 'asset_age_years' in df.columns:
            df['manufacturer_avg_age'] = df.groupby(manufacturer_col)['asset_age_years'].transform('mean')
        
        # Top manufacturers flag
        top_manufacturers = df[manufacturer_col].value_counts().nlargest(10).index
        df['is_top_manufacturer'] = df[manufacturer_col].isin(top_manufacturers).astype(int)
    
    return df


def create_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from text analysis"""
    logger.info("Creating text analysis features")
    
    if 'equipment_description' in df.columns:
        desc_col = df['equipment_description'].astype(str).str.upper()
        
        # Criticality indicators
        critical_pattern = '|'.join(CRITICAL_KEYWORDS)
        df['is_critical'] = desc_col.str.contains(critical_pattern, na=False).astype(int)
        
        # Safety indicators
        safety_keywords = ['SAFETY', 'EMERGENCY', 'FIRE', 'ALARM', 'PROTECTION']
        safety_pattern = '|'.join(safety_keywords)
        df['is_safety_equipment'] = desc_col.str.contains(safety_pattern, na=False).astype(int)
        
        # Environmental indicators
        env_keywords = ['ENVIRONMENTAL', 'WASTE', 'EMISSION', 'POLLUTION']
        env_pattern = '|'.join(env_keywords)
        df['is_environmental'] = desc_col.str.contains(env_pattern, na=False).astype(int)
        
        # Automation level
        automation_keywords = ['AUTOMATIC', 'AUTO', 'CONTROL', 'PLC', 'SCADA']
        automation_pattern = '|'.join(automation_keywords)
        df['automation_level'] = desc_col.str.contains(automation_pattern, na=False).astype(int)
    
    return df


def create_aggregation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create aggregation-based features for group analysis"""
    logger.info("Creating aggregation features")
    
    # Determine the equipment ID column name
    equipment_col = None
    for col in ['equipment_id', 'equipment', 'equi_id']:
        if col in df.columns:
            equipment_col = col
            break
    
    if not equipment_col:
        logger.warning("No equipment ID column found, skipping aggregation features")
        return df
    
    # Equipment density by location
    if 'field' in df.columns:
        df['field_equipment_density'] = df.groupby('field')[equipment_col].transform('count')
        if 'acquisition_value' in df.columns:
            df['field_total_value'] = df.groupby('field')['acquisition_value'].transform('sum')
    
    # Time-based aggregations
    if 'creation_year' in df.columns:
        df['year_equipment_count'] = df.groupby('creation_year')[equipment_col].transform('count')
        if 'acquisition_value' in df.columns:
            df['year_total_investment'] = df.groupby('creation_year')['acquisition_value'].transform('sum')
    
    # Manufacturer market share
    if 'manufacturer_of_asset' in df.columns:
        total_equipment = len(df)
        df['manufacturer_market_share'] = (df.groupby('manufacturer_of_asset')[equipment_col].transform('count') / total_equipment * 100)
    
    return df


def create_risk_priority_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create risk assessment and priority features for AI agent queries"""
    logger.info("Creating risk and priority features")
    
    # Risk score calculation
    risk_factors = []
    
    if 'asset_age_years' in df.columns:
        # Age risk (older equipment = higher risk)
        age_risk = pd.cut(df['asset_age_years'], 
                         bins=[-np.inf, 5, 10, 15, np.inf],
                         labels=[0, 1, 2, 3]).astype(float)
        risk_factors.append(age_risk)
    
    if 'is_critical' in df.columns:
        # Critical equipment risk
        risk_factors.append(df['is_critical'] * 2)  # Weight critical equipment higher
    
    if 'warranty_status' in df.columns:
        # Warranty risk (expired = higher risk)
        warranty_risk = (df['warranty_status'] == 'Expired').astype(int)
        risk_factors.append(warranty_risk)
    
    # Combine risk factors (convert to float to handle categorical)
    if risk_factors:
        # Convert all factors to float to avoid categorical issues
        converted_factors = []
        for factor in risk_factors:
            if hasattr(factor, 'astype'):
                converted_factors.append(factor.astype(float))
            else:
                converted_factors.append(pd.Series(factor, dtype=float))
        
        # Sum the factors
        if len(converted_factors) == 1:
            df['risk_score'] = converted_factors[0]
        else:
            df['risk_score'] = converted_factors[0]
            for factor in converted_factors[1:]:
                df['risk_score'] = df['risk_score'] + factor
        
        df['risk_category'] = pd.cut(df['risk_score'],
                                   bins=[-np.inf, 1, 3, 5, np.inf],
                                   labels=['Low', 'Medium', 'High', 'Critical'])
    
    # Priority score for maintenance scheduling
    priority_factors = []
    
    if 'value_category' in df.columns:
        value_priority = df['value_category'].map({'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4})
        priority_factors.append(value_priority.fillna(1))  # Fill NaN with 1
    
    if 'is_safety_equipment' in df.columns:
        priority_factors.append(df['is_safety_equipment'] * 3)  # Safety is high priority
    
    # Combine priority factors (similar approach)
    if priority_factors:
        # Convert all factors to float to avoid categorical issues
        converted_priority_factors = []
        for factor in priority_factors:
            if hasattr(factor, 'astype'):
                converted_priority_factors.append(factor.astype(float))
            else:
                converted_priority_factors.append(pd.Series(factor, dtype=float))
        
        # Sum the factors
        if len(converted_priority_factors) == 1:
            df['maintenance_priority'] = converted_priority_factors[0]
        else:
            df['maintenance_priority'] = converted_priority_factors[0]
            for factor in converted_priority_factors[1:]:
                df['maintenance_priority'] = df['maintenance_priority'] + factor
        
        df['maintenance_urgency'] = pd.cut(df['maintenance_priority'],
                                         bins=[-np.inf, 2, 4, 6, np.inf],
                                         labels=['Routine', 'Scheduled', 'Urgent', 'Immediate'])
    
    return df