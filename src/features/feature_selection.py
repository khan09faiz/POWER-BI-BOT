"""
Feature selection and dimensionality reduction for equipment data
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from pathlib import Path
import sys
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import our custom utilities
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import PCA_VARIANCE_THRESHOLD, CORRELATION_THRESHOLD
from src.utils.logger import setup_logger, log_dataframe_info, log_processing_step
from src.utils.memory_utils import optimize_dtypes

# Setup logger
logger = setup_logger(__name__, "logs/feature_selection.log")


class FeatureSelector:
    """
    Comprehensive feature selection and dimensionality reduction for equipment data
    """
    
    def __init__(self, 
                 correlation_threshold: float = CORRELATION_THRESHOLD,
                 pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD):
        self.correlation_threshold = correlation_threshold
        self.pca_variance_threshold = pca_variance_threshold
        self.selected_features = []
        self.feature_importance_scores = {}
        self.pca_model = None
        self.scaler = None
        
    def select_features(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Main feature selection pipeline
        
        Args:
            df: Input DataFrame with features
            target_column: Optional target column for supervised selection
            
        Returns:
            DataFrame with selected features
        """
        logger.info("Starting comprehensive feature selection")
        initial_shape = df.shape
        
        # 1. Remove low-variance features
        df_selected = self.remove_low_variance_features(df)
        
        # 2. Remove highly correlated features
        df_selected = self.remove_correlated_features(df_selected)
        
        # 3. Feature importance ranking (if target provided)
        if target_column and target_column in df_selected.columns:
            df_selected = self.select_important_features(df_selected, target_column)
        
        # 4. Remove redundant categorical features
        df_selected = self.remove_redundant_categorical(df_selected)
        
        # 5. Final optimization
        df_selected = optimize_dtypes(df_selected)
        
        final_shape = df_selected.shape
        log_processing_step(logger, "Feature Selection", initial_shape, final_shape)
        
        return df_selected
    
    def remove_low_variance_features(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with very low variance"""
        logger.info("Removing low-variance features")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        low_variance_cols = []
        
        for col in numeric_cols:
            if df[col].var() < threshold:
                low_variance_cols.append(col)
        
        # Also remove constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        
        cols_to_remove = list(set(low_variance_cols + constant_cols))
        
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            logger.info(f"Removed {len(cols_to_remove)} low-variance/constant features: {cols_to_remove}")
        
        return df
    
    def remove_correlated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features"""
        logger.info(f"Removing features with correlation > {self.correlation_threshold}")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return df
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > self.correlation_threshold:
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        # Remove features with highest average correlation
        features_to_remove = set()
        for col1, col2, corr_val in high_corr_pairs:
            if col1 not in features_to_remove and col2 not in features_to_remove:
                # Keep the feature with lower average correlation
                col1_avg_corr = corr_matrix[col1].mean()
                col2_avg_corr = corr_matrix[col2].mean()
                
                if col1_avg_corr > col2_avg_corr:
                    features_to_remove.add(col1)
                else:
                    features_to_remove.add(col2)
        
        if features_to_remove:
            df = df.drop(columns=list(features_to_remove))
            logger.info(f"Removed {len(features_to_remove)} highly correlated features: {list(features_to_remove)}")
        
        return df
    
    def select_important_features(self, df: pd.DataFrame, target_column: str, 
                                top_k: int = 50) -> pd.DataFrame:
        """Select top-k important features using multiple methods"""
        logger.info(f"Selecting top {top_k} important features")
        
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols]
        y = df[target_column]
        
        # Prepare features for analysis
        X_numeric = self._prepare_features_for_selection(X)
        
        # Method 1: Random Forest Feature Importance
        rf_scores = self._get_random_forest_importance(X_numeric, y)
        
        # Method 2: Mutual Information
        mi_scores = self._get_mutual_information_scores(X_numeric, y)
        
        # Method 3: Statistical tests
        stat_scores = self._get_statistical_scores(X_numeric, y)
        
        # Combine scores (average ranking)
        combined_scores = self._combine_importance_scores([rf_scores, mi_scores, stat_scores])
        
        # Select top features
        top_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        selected_feature_names = [feat for feat, score in top_features]
        
        # Include target column
        final_columns = selected_feature_names + [target_column]
        
        self.feature_importance_scores = dict(top_features)
        logger.info(f"Selected {len(selected_feature_names)} features based on importance")
        
        return df[final_columns]
    
    def apply_pca(self, df: pd.DataFrame, exclude_columns: List[str] = None) -> pd.DataFrame:
        """Apply PCA for dimensionality reduction"""
        logger.info(f"Applying PCA with {self.pca_variance_threshold} variance threshold")
        
        exclude_columns = exclude_columns or []
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_columns]
        
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for PCA")
            return df
        
        X = df[numeric_cols].fillna(0)
        
        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        self.pca_model = PCA(n_components=self.pca_variance_threshold)
        X_pca = self.pca_model.fit_transform(X_scaled)
        
        # Create PCA DataFrame
        pca_columns = [f'PC_{i+1}' for i in range(X_pca.shape[1])]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        # Combine with non-numeric columns
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        df_final = pd.concat([df[non_numeric_cols], df_pca], axis=1)
        
        logger.info(f"PCA reduced {len(numeric_cols)} features to {len(pca_columns)} components")
        logger.info(f"Explained variance ratio: {self.pca_model.explained_variance_ratio_.sum():.3f}")
        
        return df_final
    
    def remove_redundant_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove redundant categorical features"""
        logger.info("Removing redundant categorical features")
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        redundant_cols = []
        
        for col in categorical_cols:
            # Remove columns with only one unique value (after cleaning)
            if df[col].nunique() <= 1:
                redundant_cols.append(col)
            
            # Remove columns that are mostly 'Unknown' or 'Other'
            elif df[col].value_counts().iloc[0] / len(df) > 0.95:
                top_value = df[col].value_counts().index[0]
                if top_value.lower() in ['unknown', 'other', 'none', 'nan']:
                    redundant_cols.append(col)
        
        if redundant_cols:
            df = df.drop(columns=redundant_cols)
            logger.info(f"Removed {len(redundant_cols)} redundant categorical features: {redundant_cols}")
        
        return df
    
    def _prepare_features_for_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for selection algorithms"""
        X_prepared = X.copy()
        
        # Handle categorical variables
        categorical_cols = X_prepared.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            # Label encode categorical variables
            le = LabelEncoder()
            X_prepared[col] = le.fit_transform(X_prepared[col].astype(str))
        
        # Handle missing values
        X_prepared = X_prepared.fillna(0)
        
        return X_prepared
    
    def _get_random_forest_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get feature importance using Random Forest"""
        try:
            if y.dtype in ['object', 'category'] or y.nunique() < 10:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            return dict(zip(X.columns, model.feature_importances_))
        except:
            return {col: 0 for col in X.columns}
    
    def _get_mutual_information_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get mutual information scores"""
        try:
            scores = mutual_info_classif(X, y, random_state=42)
            return dict(zip(X.columns, scores))
        except:
            return {col: 0 for col in X.columns}
    
    def _get_statistical_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Get statistical test scores"""
        try:
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X, y)
            return dict(zip(X.columns, selector.scores_))
        except:
            return {col: 0 for col in X.columns}
    
    def _combine_importance_scores(self, score_dicts: List[Dict[str, float]]) -> Dict[str, float]:
        """Combine multiple importance scores using average ranking"""
        all_features = set()
        for score_dict in score_dicts:
            all_features.update(score_dict.keys())
        
        combined_scores = {}
        
        for feature in all_features:
            ranks = []
            for score_dict in score_dicts:
                if feature in score_dict:
                    # Convert scores to ranks (higher score = lower rank number)
                    sorted_features = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
                    rank = next(i for i, (f, s) in enumerate(sorted_features) if f == feature) + 1
                    ranks.append(rank)
            
            # Average rank (lower is better, so we invert for final score)
            avg_rank = np.mean(ranks) if ranks else len(all_features)
            combined_scores[feature] = 1.0 / avg_rank
        
        return combined_scores
    
    def get_feature_report(self) -> Dict[str, Any]:
        """Generate a comprehensive feature selection report"""
        return {
            'selected_features': self.selected_features,
            'feature_importance_scores': self.feature_importance_scores,
            'pca_components': self.pca_model.n_components_ if self.pca_model else None,
            'pca_explained_variance': self.pca_model.explained_variance_ratio_.sum() if self.pca_model else None
        }


def optimize_feature_set(df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
    """
    Main function to optimize feature set for the equipment dataset
    
    Args:
        df: Input DataFrame
        target_column: Optional target column for supervised selection
        
    Returns:
        Optimized DataFrame with selected features
    """
    logger.info("Starting feature set optimization for equipment data")
    
    # Initialize feature selector
    selector = FeatureSelector()
    
    # Apply feature selection
    df_optimized = selector.select_features(df, target_column)
    
    # Generate report
    report = selector.get_feature_report()
    logger.info(f"Feature optimization complete. Report: {report}")
    
    return df_optimized
