"""
Enterprise-grade ML-driven feature selection
Focus: Business-relevant target variables and interpretable models
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, RFE
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class BusinessDrivenFeatureSelector:
    """
    Principal Data Scientist approach to feature selection:
    - Business target variable alignment
    - Model interpretability preservation
    - Performance validation
    """
    
    def __init__(self, data, target_variable, problem_type='regression'):
        self.data = data.copy()
        self.target_variable = target_variable
        self.problem_type = problem_type
        self.selected_features = []
        self.feature_importance_scores = {}
        self.validation_results = {}
        
    def create_business_target_variable(self):
        """
        DO: Create meaningful target variables based on business goals
        
        For equipment data, relevant targets might be:
        - Equipment failure risk score
        - Maintenance cost prediction
        - Equipment replacement priority
        """
        print("🎯 CREATING BUSINESS-RELEVANT TARGET VARIABLE")
        print("=" * 50)
        
        if self.target_variable == 'equipment_risk_score':
            # Create comprehensive risk score based on multiple factors
            risk_score = self.calculate_equipment_risk_score()
            self.data[self.target_variable] = risk_score
            
        elif self.target_variable == 'maintenance_priority':
            # Create maintenance priority classification
            priority_class = self.calculate_maintenance_priority()
            self.data[self.target_variable] = priority_class
            
        elif self.target_variable == 'replacement_urgency':
            # Create replacement urgency based on age, value, and condition
            urgency_score = self.calculate_replacement_urgency()
            self.data[self.target_variable] = urgency_score
        
        elif self.target_variable not in self.data.columns:
            # Create default risk score if target doesn't exist
            print(f"Target variable '{self.target_variable}' not found. Creating default risk score.")
            risk_score = self.calculate_equipment_risk_score()
            self.data[self.target_variable] = risk_score
            
        print(f"✅ Target variable '{self.target_variable}' created successfully")
        print(f"Target distribution:\n{self.data[self.target_variable].value_counts()}")
        
        return self.data[self.target_variable]
    
    def calculate_equipment_risk_score(self):
        """
        DO: Create composite business metrics
        """
        risk_factors = []
        
        # Age-based risk (older equipment = higher risk)
        if 'equipment_age_years' in self.data.columns:
            age_risk = np.clip(self.data['equipment_age_years'] / 20, 0, 1)
            risk_factors.append(age_risk)
        elif 'created_on' in self.data.columns:
            # Calculate age from created_on
            current_date = pd.Timestamp.now()
            equipment_age = (current_date - pd.to_datetime(self.data['created_on'])).dt.days / 365.25
            age_risk = np.clip(equipment_age / 20, 0, 1)
            risk_factors.append(age_risk)
        
        # Value-based risk (higher value = higher impact)
        if 'acquisition_value' in self.data.columns:
            # Handle missing values and normalize
            values = self.data['acquisition_value'].fillna(self.data['acquisition_value'].median())
            max_value = values.max()
            if max_value > 0:
                value_normalized = values / max_value
                risk_factors.append(value_normalized)
        
        # Missing data risk (more missing data = higher uncertainty)
        missing_ratio = self.data.isnull().sum(axis=1) / len(self.data.columns)
        risk_factors.append(missing_ratio)
        
        # Weight-based risk (heavier equipment often more critical)
        if 'weight' in self.data.columns:
            weights = self.data['weight'].fillna(self.data['weight'].median())
            max_weight = weights.max()
            if max_weight > 0:
                weight_normalized = weights / max_weight
                risk_factors.append(weight_normalized * 0.5)  # Lower weight factor
        
        # Combine risk factors
        if risk_factors:
            risk_score = np.mean(risk_factors, axis=0)
            return (risk_score * 100).round(2)  # Scale to 0-100
        else:
            # Fallback: random scores for demonstration
            return np.random.uniform(20, 80, len(self.data)).round(2)
    
    def calculate_maintenance_priority(self):
        """
        Create maintenance priority classification
        """
        # Use risk score as base
        risk_scores = self.calculate_equipment_risk_score()
        
        # Create priority classes
        priority_classes = pd.cut(
            risk_scores,
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return priority_classes
    
    def calculate_replacement_urgency(self):
        """
        Create replacement urgency score
        """
        # Similar to risk score but focused on replacement needs
        return self.calculate_equipment_risk_score()
    
    def random_forest_feature_selection(self, n_features=20):
        """
        DO: Use Random Forest for feature importance with business interpretation
        """
        print("🌲 RANDOM FOREST FEATURE SELECTION")
        print("=" * 50)
        
        # Prepare features (exclude target and non-predictive columns)
        exclude_columns = [
            self.target_variable, 'equipment', 'data_source_file', 
            'period_label', 'equipment_description'
        ]
        feature_columns = [
            col for col in self.data.columns 
            if col not in exclude_columns
        ]
        
        # Handle case where we have no valid features
        if len(feature_columns) == 0:
            print("⚠️ No valid feature columns found")
            return []
        
        # Handle categorical variables
        X = self.prepare_features_for_ml(feature_columns)
        y = self.data[self.target_variable].copy()
        
        # Remove samples with missing target
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(y) < 10:
            print("⚠️ Insufficient data for analysis")
            return []
        
        # Check target variable variance
        if self.problem_type == 'classification':
            if len(pd.Series(y).unique()) < 2:
                print("⚠️ Target variable has insufficient variation for classification")
                return []
        else:
            if pd.Series(y).var() == 0:
                print("⚠️ Target variable has no variance for regression")
                return []
        
        # Choose appropriate model
        if self.problem_type == 'classification':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=max(2, len(y) // 50)
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=max(2, len(y) // 50)
            )
        
        try:
            # Fit model and get feature importance
            model.fit(X, y)
            
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Cross-validation for model validation
            cv_scores = cross_val_score(
                model, X, y, cv=min(5, len(y) // 10), 
                scoring='neg_mean_squared_error' if self.problem_type == 'regression' else 'accuracy'
            )
            
            print(f"Model Performance (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            print(f"\nTop {min(n_features, len(feature_columns))} Most Important Features:")
            print(feature_importance.head(min(n_features, len(feature_columns))))
            
            # Select top features
            selected_features = feature_importance.head(min(n_features, len(feature_columns)))['feature'].tolist()
            
            # Business interpretation
            self.interpret_feature_importance(feature_importance.head(min(n_features, len(feature_columns))))
            
            self.feature_importance_scores['random_forest'] = feature_importance
            return selected_features
            
        except Exception as e:
            print(f"⚠️ Error in Random Forest feature selection: {str(e)}")
            return feature_columns[:min(n_features, len(feature_columns))]
    
    def mutual_information_selection(self, n_features=15):
        """
        DO: Use mutual information for non-linear relationships
        """
        print("🔗 MUTUAL INFORMATION FEATURE SELECTION")
        print("=" * 50)
        
        exclude_columns = [
            self.target_variable, 'equipment', 'data_source_file', 
            'period_label', 'equipment_description'
        ]
        feature_columns = [
            col for col in self.data.columns 
            if col not in exclude_columns
        ]
        
        if len(feature_columns) == 0:
            print("⚠️ No valid feature columns found")
            return []
        
        X = self.prepare_features_for_ml(feature_columns)
        y = self.data[self.target_variable].copy()
        
        # Remove missing values
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(y) < 10:
            print("⚠️ Insufficient data for mutual information analysis")
            return feature_columns[:min(n_features, len(feature_columns))]
        
        try:
            # Calculate mutual information
            if self.problem_type == 'classification':
                mi_scores = mutual_info_classif(X, y, random_state=42)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=42)
            
            mi_results = pd.DataFrame({
                'feature': feature_columns,
                'mutual_info': mi_scores
            }).sort_values('mutual_info', ascending=False)
            
            print(f"Top {min(n_features, len(feature_columns))} Features by Mutual Information:")
            print(mi_results.head(min(n_features, len(feature_columns))))
            
            selected_features = mi_results.head(min(n_features, len(feature_columns)))['feature'].tolist()
            self.feature_importance_scores['mutual_information'] = mi_results
            
            return selected_features
            
        except Exception as e:
            print(f"⚠️ Error in Mutual Information selection: {str(e)}")
            return feature_columns[:min(n_features, len(feature_columns))]
    
    def recursive_feature_elimination(self, n_features=10):
        """
        DO: Use RFE for optimal feature subset selection
        """
        print("♻️ RECURSIVE FEATURE ELIMINATION")
        print("=" * 50)
        
        exclude_columns = [
            self.target_variable, 'equipment', 'data_source_file', 
            'period_label', 'equipment_description'
        ]
        feature_columns = [
            col for col in self.data.columns 
            if col not in exclude_columns
        ]
        
        if len(feature_columns) == 0:
            print("⚠️ No valid feature columns found")
            return []
        
        X = self.prepare_features_for_ml(feature_columns)
        y = self.data[self.target_variable].copy()
        
        valid_indices = ~pd.isna(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(y) < 10:
            print("⚠️ Insufficient data for RFE analysis")
            return feature_columns[:min(n_features, len(feature_columns))]
        
        try:
            # Base estimator
            if self.problem_type == 'classification':
                estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # RFE
            n_features_to_select = min(n_features, len(feature_columns))
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
            rfe.fit(X, y)
            
            selected_features = [
                feature_columns[i] for i, selected in enumerate(rfe.support_) 
                if selected
            ]
            
            feature_rankings = pd.DataFrame({
                'feature': feature_columns,
                'ranking': rfe.ranking_,
                'selected': rfe.support_
            }).sort_values('ranking')
            
            print(f"Selected Features by RFE:")
            print(feature_rankings[feature_rankings['selected']])
            
            self.feature_importance_scores['rfe'] = feature_rankings
            return selected_features
            
        except Exception as e:
            print(f"⚠️ Error in RFE selection: {str(e)}")
            return feature_columns[:min(n_features, len(feature_columns))]
    
    def ensemble_feature_selection(self):
        """
        DO: Combine multiple selection methods for robust results
        """
        print("🎭 ENSEMBLE FEATURE SELECTION")
        print("=" * 50)
        
        # Run multiple selection methods
        rf_features = self.random_forest_feature_selection(n_features=20)
        mi_features = self.mutual_information_selection(n_features=15)
        rfe_features = self.recursive_feature_elimination(n_features=10)
        
        # Create voting system
        all_features = set(rf_features + mi_features + rfe_features)
        feature_votes = {}
        
        for feature in all_features:
            votes = 0
            if feature in rf_features:
                votes += 3  # Random Forest gets higher weight
            if feature in mi_features:
                votes += 2  # Mutual Info gets medium weight
            if feature in rfe_features:
                votes += 3  # RFE gets higher weight
            
            feature_votes[feature] = votes
        
        # Select features with highest votes
        sorted_features = sorted(
            feature_votes.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        final_features = [feature for feature, votes in sorted_features[:15]]
        
        print(f"Final Selected Features (Top 15):")
        for feature, votes in sorted_features[:15]:
            print(f"  {feature}: {votes} votes")
        
        self.selected_features = final_features
        return final_features
    
    def validate_feature_selection(self):
        """
        DO: Validate feature selection with business metrics
        """
        if not self.selected_features:
            print("⚠️ No features selected yet. Run ensemble_feature_selection() first.")
            return
        
        print("✅ FEATURE SELECTION VALIDATION")
        print("=" * 50)
        
        # Compare performance: all features vs selected features
        exclude_columns = [
            self.target_variable, 'equipment', 'data_source_file', 
            'period_label', 'equipment_description'
        ]
        feature_columns = [
            col for col in self.data.columns 
            if col not in exclude_columns
        ]
        
        if len(feature_columns) == 0:
            print("⚠️ No valid feature columns for validation")
            return
        
        X_all = self.prepare_features_for_ml(feature_columns)
        X_selected = self.prepare_features_for_ml(self.selected_features)
        y = self.data[self.target_variable].copy()
        
        valid_indices = ~pd.isna(y)
        X_all = X_all[valid_indices]
        X_selected = X_selected[valid_indices]
        y = y[valid_indices]
        
        if len(y) < 10:
            print("⚠️ Insufficient data for validation")
            return
        
        try:
            # Model comparison
            if self.problem_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                scoring = 'neg_mean_squared_error'
            
            # Performance with all features
            cv_all = cross_val_score(model, X_all, y, cv=min(5, len(y) // 10), scoring=scoring)
            
            # Performance with selected features
            cv_selected = cross_val_score(model, X_selected, y, cv=min(5, len(y) // 10), scoring=scoring)
            
            print(f"All Features ({len(feature_columns)}): {cv_all.mean():.4f} ± {cv_all.std():.4f}")
            print(f"Selected Features ({len(self.selected_features)}): {cv_selected.mean():.4f} ± {cv_selected.std():.4f}")
            
            # Feature reduction benefit
            reduction_percentage = (1 - len(self.selected_features) / len(feature_columns)) * 100
            print(f"\n📊 Feature Reduction: {reduction_percentage:.1f}%")
            
            # Performance difference
            performance_diff = cv_selected.mean() - cv_all.mean()
            if abs(performance_diff) < 0.01:  # Negligible difference
                print(f"✅ Maintained performance with {reduction_percentage:.1f}% fewer features")
            elif performance_diff > 0:
                print(f"✅ Improved performance by {performance_diff:.4f}")
            else:
                print(f"⚠️ Performance decreased by {abs(performance_diff):.4f}")
            
            self.validation_results = {
                'all_features_performance': cv_all.mean(),
                'selected_features_performance': cv_selected.mean(),
                'feature_reduction_percentage': reduction_percentage,
                'performance_difference': performance_diff
            }
            
            return self.validation_results
            
        except Exception as e:
            print(f"⚠️ Error in validation: {str(e)}")
            return None
    
    def prepare_features_for_ml(self, feature_columns):
        """
        DO: Properly handle mixed data types for ML
        """
        X = self.data[feature_columns].copy()
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if X[col].dtype.name == 'category':
                # Already categorical
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna('Unknown'))
            else:
                # Convert object to category
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].fillna('Unknown').astype(str))
        
        # Handle datetime columns
        datetime_columns = X.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            X[col] = pd.to_datetime(X[col]).astype('int64') // 10**9  # Convert to Unix timestamp
        
        # Fill remaining missing values with median for numeric, mode for others
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'int8']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 0)
        
        return X.values
    
    def interpret_feature_importance(self, feature_importance_df):
        """
        DO: Provide business interpretation of feature importance
        """
        print("\n💡 BUSINESS INTERPRETATION")
        print("-" * 30)
        
        top_features = feature_importance_df.head(5)
        
        for _, row in top_features.iterrows():
            feature_name = row['feature']
            importance = row['importance']
            
            # Business context interpretation
            if 'age' in feature_name.lower():
                interpretation = "Equipment age is a critical factor - supports lifecycle management strategy"
            elif 'value' in feature_name.lower() or 'acquisition' in feature_name.lower():
                interpretation = "Asset value drives decisions - aligns with financial risk management"
            elif 'manufacturer' in feature_name.lower():
                interpretation = "Manufacturer quality affects outcomes - supports vendor management"
            elif 'weight' in feature_name.lower():
                interpretation = "Equipment size/complexity impacts maintenance requirements"
            elif 'created' in feature_name.lower():
                interpretation = "Installation timing affects performance - supports planning strategies"
            elif 'object' in feature_name.lower():
                interpretation = "Equipment type classification - supports category-based maintenance"
            else:
                interpretation = "Domain expert review recommended for business context"
            
            print(f"• {feature_name} ({importance:.4f}): {interpretation}")
    
    def create_feature_selection_report(self):
        """
        Create comprehensive feature selection report
        """
        print("📋 FEATURE SELECTION REPORT")
        print("=" * 50)
        
        report = {
            'target_variable': self.target_variable,
            'problem_type': self.problem_type,
            'selected_features': self.selected_features,
            'feature_importance_scores': self.feature_importance_scores,
            'validation_results': self.validation_results,
            'business_recommendations': self.generate_business_recommendations()
        }
        
        return report
    
    def generate_business_recommendations(self):
        """
        Generate business-focused recommendations
        """
        recommendations = [
            "Focus on equipment age and value for risk assessment",
            "Implement manufacturer-based quality scoring",
            "Use temporal features for predictive maintenance scheduling",
            "Consider equipment complexity (weight/size) in resource planning",
            "Establish data quality monitoring for missing value patterns"
        ]
        
        if self.validation_results:
            if self.validation_results.get('feature_reduction_percentage', 0) > 50:
                recommendations.append("Significant feature reduction achieved - consider simplified models")
            
            if self.validation_results.get('performance_difference', 0) > 0:
                recommendations.append("Feature selection improved model performance - deploy selected features")
        
        return recommendations
