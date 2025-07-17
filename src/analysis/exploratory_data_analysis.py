"""
Comprehensive EDA module for industrial equipment data analysis
Focuses on business value discovery rather than just statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class IndustrialEDAAnalyzer:
    """
    Principal Data Scientist approach to EDA:
    - Business-driven hypothesis testing
    - Statistical significance validation
    - Performance impact assessment
    """
    
    def __init__(self, data, business_context=None):
        self.data = data.copy()
        self.business_context = business_context or {}
        self.insights = {}
        self.feature_candidates = []
        
    def comprehensive_data_profiling(self):
        """
        DO: Systematic data profiling with business context
        """
        print("🔍 COMPREHENSIVE DATA PROFILING")
        print("=" * 50)
        
        # Data Shape and Structure
        print(f"Dataset Dimensions: {self.data.shape}")
        print(f"Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Business-Critical Missing Data Analysis
        missing_analysis = self.analyze_missing_patterns()
        
        # Data Type Optimization Opportunities
        dtype_optimization = self.analyze_dtype_efficiency()
        
        # Temporal Pattern Analysis (Critical for Equipment Data)
        temporal_insights = self.analyze_temporal_patterns()
        
        return {
            'missing_patterns': missing_analysis,
            'dtype_optimization': dtype_optimization,
            'temporal_insights': temporal_insights
        }
    
    def analyze_missing_patterns(self):
        """
        DO: Understand missing data in business context
        Missing data in equipment records often indicates:
        - Equipment lifecycle stages
        - Data collection system changes
        - Business process variations
        """
        missing_df = pd.DataFrame({
            'column': self.data.columns,
            'missing_count': self.data.isnull().sum(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data)) * 100
        }).sort_values('missing_percentage', ascending=False)
        
        # Business Impact Classification
        critical_missing = missing_df[missing_df['missing_percentage'] > 90]
        moderate_missing = missing_df[
            (missing_df['missing_percentage'] > 10) & 
            (missing_df['missing_percentage'] <= 90)
        ]
        
        print(f"🚨 Critical Missing (>90%): {len(critical_missing)} columns")
        print(f"⚠️  Moderate Missing (10-90%): {len(moderate_missing)} columns")
        
        return {
            'critical_missing_columns': critical_missing['column'].tolist(),
            'moderate_missing_columns': moderate_missing['column'].tolist(),
            'missing_summary': missing_df
        }
    
    def analyze_temporal_patterns(self):
        """
        DO: Deep dive into time-series patterns for equipment data
        """
        temporal_cols = self.data.select_dtypes(include=['datetime64']).columns
        insights = {}
        
        for col in temporal_cols:
            if self.data[col].notna().sum() > 100:  # Sufficient data
                # Equipment creation patterns
                yearly_pattern = self.data[col].dt.year.value_counts().sort_index()
                monthly_pattern = self.data[col].dt.month.value_counts().sort_index()
                
                insights[col] = {
                    'yearly_trend': yearly_pattern.to_dict(),
                    'monthly_seasonality': monthly_pattern.to_dict(),
                    'date_range': f"{self.data[col].min()} to {self.data[col].max()}"
                }
        
        return insights
    
    def analyze_dtype_efficiency(self):
        """
        DO: Identify data type optimization opportunities
        """
        optimization_suggestions = {}
        
        # Analyze integer columns
        int_cols = self.data.select_dtypes(include=['int64']).columns
        for col in int_cols:
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            
            if col_min >= -128 and col_max <= 127:
                optimization_suggestions[col] = 'int8'
            elif col_min >= -32768 and col_max <= 32767:
                optimization_suggestions[col] = 'int16'
            elif col_min >= -2147483648 and col_max <= 2147483647:
                optimization_suggestions[col] = 'int32'
        
        # Analyze float columns
        float_cols = self.data.select_dtypes(include=['float64']).columns
        for col in float_cols:
            if not self.data[col].isnull().all():
                # Check if values can fit in float32
                max_val = abs(self.data[col]).max()
                if max_val < 3.4e38:  # float32 range
                    optimization_suggestions[col] = 'float32'
        
        # Analyze object columns for categorical conversion
        obj_cols = self.data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            unique_ratio = self.data[col].nunique() / len(self.data[col])
            if unique_ratio < 0.5:  # Less than 50% unique values
                optimization_suggestions[col] = 'category'
        
        return optimization_suggestions
    
    def equipment_lifecycle_analysis(self):
        """
        DO: Business-specific analysis for equipment domain
        """
        print("🏭 EQUIPMENT LIFECYCLE ANALYSIS")
        print("=" * 50)
        
        results = {}
        
        # Equipment Age Distribution
        if 'created_on' in self.data.columns:
            current_date = pd.Timestamp.now()
            self.data['equipment_age_years'] = (
                current_date - pd.to_datetime(self.data['created_on'])
            ).dt.days / 365.25
            
            age_stats = self.data['equipment_age_years'].describe()
            print(f"Equipment Age Statistics:\n{age_stats}")
            
            # Age-based risk categorization
            self.data['age_risk_category'] = pd.cut(
                self.data['equipment_age_years'],
                bins=[0, 5, 10, 15, float('inf')],
                labels=['New', 'Mature', 'Aging', 'Critical']
            )
            
            results['age_distribution'] = age_stats.to_dict()
        
        # Equipment Value Analysis
        if 'acquisition_value' in self.data.columns:
            value_analysis = self.analyze_equipment_value_patterns()
            results['value_patterns'] = value_analysis
            
        # Manufacturer Performance Analysis
        if 'manufacturer_of_asset' in self.data.columns:
            manufacturer_insights = self.analyze_manufacturer_patterns()
            results['manufacturer_insights'] = manufacturer_insights
            
        return results
    
    def analyze_equipment_value_patterns(self):
        """
        Analyze equipment value distributions and patterns
        """
        value_col = 'acquisition_value'
        if value_col not in self.data.columns:
            return None
            
        # Basic statistics
        value_stats = self.data[value_col].describe()
        
        # Value tiers
        value_quantiles = self.data[value_col].quantile([0.25, 0.5, 0.75, 0.9])
        
        # High-value equipment analysis
        high_value_threshold = value_quantiles[0.9]
        high_value_equipment = self.data[self.data[value_col] > high_value_threshold]
        
        return {
            'basic_stats': value_stats.to_dict(),
            'quantiles': value_quantiles.to_dict(),
            'high_value_count': len(high_value_equipment),
            'high_value_percentage': len(high_value_equipment) / len(self.data) * 100
        }
    
    def analyze_manufacturer_patterns(self):
        """
        Analyze manufacturer distribution and performance
        """
        manufacturer_col = 'manufacturer_of_asset'
        if manufacturer_col not in self.data.columns:
            return None
            
        # Manufacturer frequency
        manufacturer_counts = self.data[manufacturer_col].value_counts()
        
        # Top manufacturers
        top_manufacturers = manufacturer_counts.head(10)
        
        # Manufacturer diversity
        total_manufacturers = len(manufacturer_counts)
        top_5_share = manufacturer_counts.head(5).sum() / len(self.data) * 100
        
        return {
            'total_manufacturers': total_manufacturers,
            'top_manufacturers': top_manufacturers.to_dict(),
            'top_5_market_share': top_5_share,
            'manufacturer_diversity_index': 1 - (manufacturer_counts / len(self.data)).pow(2).sum()
        }
    
    def correlation_heatmap_analysis(self, target_variable=None):
        """
        DO: Focus correlation analysis on business-relevant features
        """
        # Select only numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("⚠️ Insufficient numeric columns for correlation analysis")
            return None
        
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Create interactive heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title='Equipment Data Correlation Heatmap',
            width=800,
            height=600
        )
        
        # Identify high correlations (business insights)
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'high_correlations': high_correlations,
            'visualization': fig
        }
    
    def distribution_analysis(self):
        """
        DO: Understand data distributions for modeling decisions
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        distribution_insights = {}
        
        for col in numeric_cols[:10]:  # Analyze top 10 numeric columns
            if self.data[col].notna().sum() > 50:  # Sufficient data
                # Statistical tests for normality
                try:
                    from scipy import stats
                    
                    # Remove outliers for cleaner analysis
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    clean_data = self.data[col][
                        (self.data[col] >= lower_bound) & 
                        (self.data[col] <= upper_bound)
                    ].dropna()
                    
                    if len(clean_data) > 10:
                        skewness = stats.skew(clean_data)
                        kurtosis = stats.kurtosis(clean_data)
                        
                        distribution_insights[col] = {
                            'skewness': skewness,
                            'kurtosis': kurtosis,
                            'outlier_percentage': (len(self.data[col]) - len(clean_data)) / len(self.data[col]) * 100,
                            'transformation_needed': abs(skewness) > 1 or abs(kurtosis) > 3
                        }
                except ImportError:
                    # Fallback if scipy is not available
                    distribution_insights[col] = {
                        'skewness': None,
                        'kurtosis': None,
                        'outlier_percentage': None,
                        'transformation_needed': False
                    }
        
        return distribution_insights
    
    def create_business_insights_report(self):
        """
        Generate comprehensive business insights report
        """
        print("📊 BUSINESS INSIGHTS REPORT")
        print("=" * 50)
        
        # Compile all analysis results
        profiling_results = self.comprehensive_data_profiling()
        lifecycle_insights = self.equipment_lifecycle_analysis()
        correlation_analysis = self.correlation_heatmap_analysis()
        distribution_insights = self.distribution_analysis()
        
        # Generate actionable insights
        insights = {
            'data_quality_score': self.calculate_data_quality_score(profiling_results),
            'business_readiness': self.assess_business_readiness(lifecycle_insights),
            'feature_engineering_opportunities': self.identify_feature_opportunities(correlation_analysis),
            'modeling_recommendations': self.generate_modeling_recommendations(distribution_insights)
        }
        
        # Print summary
        print(f"Data Quality Score: {insights['data_quality_score']:.1f}/100")
        print(f"Business Readiness: {insights['business_readiness']}")
        print(f"Feature Opportunities: {len(insights['feature_engineering_opportunities'])} identified")
        print(f"Modeling Recommendations: {len(insights['modeling_recommendations'])} provided")
        
        return insights
    
    def calculate_data_quality_score(self, profiling_results):
        """Calculate overall data quality score"""
        missing_patterns = profiling_results['missing_patterns']
        
        # Penalize for missing data
        critical_missing_penalty = len(missing_patterns['critical_missing_columns']) * 10
        moderate_missing_penalty = len(missing_patterns['moderate_missing_columns']) * 5
        
        base_score = 100
        quality_score = max(0, base_score - critical_missing_penalty - moderate_missing_penalty)
        
        return quality_score
    
    def assess_business_readiness(self, lifecycle_insights):
        """Assess readiness for business applications"""
        readiness_factors = []
        
        if 'age_distribution' in lifecycle_insights:
            readiness_factors.append("Age Analysis Available")
        
        if 'value_patterns' in lifecycle_insights:
            readiness_factors.append("Value Analysis Available")
            
        if 'manufacturer_insights' in lifecycle_insights:
            readiness_factors.append("Manufacturer Analysis Available")
        
        if len(readiness_factors) >= 2:
            return "High - Ready for Advanced Analytics"
        elif len(readiness_factors) == 1:
            return "Medium - Requires Additional Data"
        else:
            return "Low - Needs Data Enhancement"
    
    def identify_feature_opportunities(self, correlation_analysis):
        """Identify feature engineering opportunities"""
        opportunities = []
        
        if correlation_analysis and correlation_analysis['high_correlations']:
            opportunities.extend([
                "High correlation features identified - consider interaction terms",
                "Feature combination opportunities available",
                "Dimensionality reduction may be beneficial"
            ])
        
        opportunities.extend([
            "Equipment age-based features",
            "Value tier classifications",
            "Manufacturer performance metrics",
            "Missing data indicators",
            "Temporal pattern features"
        ])
        
        return opportunities
    
    def generate_modeling_recommendations(self, distribution_insights):
        """Generate modeling approach recommendations"""
        recommendations = []
        
        transformation_needed = sum(
            1 for insight in distribution_insights.values() 
            if insight.get('transformation_needed', False)
        )
        
        if transformation_needed > len(distribution_insights) * 0.5:
            recommendations.append("Consider data transformations for normal distribution")
        
        recommendations.extend([
            "Random Forest suitable for mixed data types",
            "Gradient Boosting for complex interactions",
            "Linear models after feature scaling",
            "Ensemble methods for robust predictions"
        ])
        
        return recommendations
