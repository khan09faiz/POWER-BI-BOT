"""
STREAMLIT DASHBOARD: Equipment Data Analysis with Enhanced Performance
Integrated with Principal Data Scientist best practices
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.memory_utils import (
    optimize_dtypes, 
    reduce_memory_usage, 
    get_memory_usage, 
    get_performance_report,
    performance_optimizer,
    memory_cleanup
)
from config.settings import PROCESSED_DATA_DIR, FEATURES_DATA_DIR

def load_optimized_data():
    """Load the optimized dataset with performance monitoring"""
    try:
        # Check multiple data sources in order of preference
        data_sources = [
            ("Optimized Parquet", PROCESSED_DATA_DIR / "master_equipment_data_optimized.parquet"),
            ("Optimized CSV", PROCESSED_DATA_DIR / "master_equipment_data_optimized.csv"),
            ("Cleaned Data", PROCESSED_DATA_DIR / "02_cleaned_data.parquet"),
            ("Feature Engineered", FEATURES_DATA_DIR / "03_feature_engineered.parquet"),
            ("Raw Concatenated", PROCESSED_DATA_DIR / "01_raw_concatenated.parquet")
        ]
        
        for data_name, data_path in data_sources:
            if data_path.exists():
                try:
                    if data_path.suffix == '.parquet':
                        df = pd.read_parquet(data_path)
                    else:
                        df = pd.read_csv(data_path)
                    
                    df = optimize_dtypes(df)  # Optimize data types
                    st.success(f"✅ Loaded {data_name}: {df.shape}")
                    st.info(f"📁 Data source: {data_path.name}")
                    return df, data_name
                except Exception as e:
                    st.warning(f"⚠️ Could not load {data_name}: {str(e)}")
                    continue
        
        # If no processed data, show message
        st.warning("⚠️ No processed data found. Please run the main pipeline first.")
        st.code("python main.py", language="bash")
        return None, None
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None

def show_data_cleaning_summary(df: pd.DataFrame, data_name: str):
    """Show comprehensive data cleaning and preprocessing summary"""
    
    st.subheader("🧹 Data Cleaning & Preprocessing Summary")
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Data Quality", "📈 Statistics", "🧮 Column Analysis"])
    
    with tab1:
        st.write(f"**Dataset Type:** {data_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Total Records", f"{len(df):,}")
        
        with col2:
            st.metric("📊 Total Columns", len(df.columns))
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB")
        
        with col4:
            data_quality = ((df.notnull().sum().sum() / (len(df) * len(df.columns))) * 100)
            st.metric("✅ Data Completeness", f"{data_quality:.1f}%")
        
        # Data types breakdown
        st.write("**📋 Data Types Summary:**")
        dtype_summary = df.dtypes.value_counts()
        col1, col2 = st.columns(2)
        
        with col1:
            for dtype, count in dtype_summary.items():
                st.write(f"• **{dtype}:** {count} columns")
        
        with col2:
            # Show optimization status
            optimized_types = ['int8', 'int16', 'int32', 'float32', 'category']
            optimized_count = sum(1 for dtype in df.dtypes if str(dtype) in optimized_types)
            optimization_pct = (optimized_count / len(df.columns)) * 100
            
            st.metric("🔧 Memory Optimized Columns", f"{optimized_count}/{len(df.columns)}")
            st.progress(optimization_pct / 100)
            st.caption(f"{optimization_pct:.1f}% of columns optimized")
    
    with tab2:
        st.write("**🔍 Data Quality Assessment:**")
        
        # Missing values analysis
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        
        if missing_data.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing %': missing_pct.values
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
            
            st.write(f"**Columns with Missing Values ({len(missing_df)} out of {len(df.columns)}):**")
            st.dataframe(missing_df, use_container_width=True)
            
            # Visual representation
            if len(missing_df) > 0:
                fig = px.bar(missing_df.head(10), x='Column', y='Missing %', 
                           title="Top 10 Columns with Missing Values")
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 **Excellent!** No missing values found in the dataset!")
        
        # Duplicate analysis
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            st.warning(f"⚠️ Found {duplicates:,} duplicate rows ({(duplicates/len(df)*100):.2f}%)")
        else:
            st.success("✅ No duplicate rows found!")
        
        # Data consistency checks
        st.write("**� Data Consistency Checks:**")
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        consistency_issues = []
        
        for col in numeric_cols:
            # Check for negative values in value/cost columns
            if any(keyword in col.lower() for keyword in ['value', 'cost', 'price', 'amount']):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    consistency_issues.append(f"❌ {col}: {negative_count} negative values")
                else:
                    consistency_issues.append(f"✅ {col}: No negative values")
        
        if consistency_issues:
            for issue in consistency_issues[:10]:  # Show top 10
                st.write(issue)
        else:
            st.info("ℹ️ No specific consistency checks performed")
    
    with tab3:
        st.write("**📈 Statistical Summary:**")
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Summary:**")
            
            # Let user select columns to analyze
            selected_numeric = st.multiselect(
                "Select numeric columns to analyze:",
                numeric_cols.tolist(),
                default=numeric_cols[:5].tolist() if len(numeric_cols) > 5 else numeric_cols.tolist()
            )
            
            if selected_numeric:
                summary_stats = df[selected_numeric].describe()
                st.dataframe(summary_stats, use_container_width=True)
                
                # Show distribution plots
                if len(selected_numeric) <= 4:  # Avoid too many plots
                    fig = make_subplots(
                        rows=1, cols=len(selected_numeric),
                        subplot_titles=selected_numeric
                    )
                    
                    for i, col in enumerate(selected_numeric):
                        fig.add_trace(
                            go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                            row=1, col=i+1
                        )
                    
                    fig.update_layout(title="Distribution of Selected Numeric Columns")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.write("**Categorical Columns Summary:**")
            
            cat_summary = []
            for col in categorical_cols:
                unique_count = df[col].nunique()
                most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                cat_summary.append({
                    'Column': col,
                    'Unique Values': unique_count,
                    'Most Frequent': str(most_frequent)[:50] + "..." if len(str(most_frequent)) > 50 else str(most_frequent)
                })
            
            cat_df = pd.DataFrame(cat_summary)
            st.dataframe(cat_df, use_container_width=True)
    
    with tab4:
        st.write("**🧮 Detailed Column Analysis:**")
        
        # Column selector
        selected_col = st.selectbox("Select a column for detailed analysis:", df.columns.tolist())
        
        if selected_col:
            col_data = df[selected_col]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Column: {selected_col}**")
                st.write(f"• **Data Type:** {col_data.dtype}")
                st.write(f"• **Non-null Count:** {col_data.count():,}")
                st.write(f"• **Null Count:** {col_data.isnull().sum():,}")
                st.write(f"• **Unique Values:** {col_data.nunique():,}")
                
                if col_data.dtype in ['object', 'category']:
                    st.write(f"• **Memory Usage:** {col_data.memory_usage(deep=True) / 1024:.1f} KB")
                
                if col_data.dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'int8']:
                    st.write(f"• **Min Value:** {col_data.min()}")
                    st.write(f"• **Max Value:** {col_data.max()}")
                    st.write(f"• **Mean:** {col_data.mean():.2f}")
            
            with col2:
                if col_data.dtype in ['int64', 'float64', 'int32', 'float32', 'int16', 'int8']:
                    # Histogram for numeric columns
                    fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Bar chart for categorical columns
                    value_counts = col_data.value_counts().head(10)
                    if len(value_counts) > 0:
                        fig = px.bar(x=value_counts.values, y=value_counts.index, 
                                   orientation='h', title=f"Top 10 values in {selected_col}")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Show sample values
            st.write("**Sample Values:**")
            non_null_values = col_data.dropna()
            if len(non_null_values) > 0:
                sample_values = non_null_values.sample(min(10, len(non_null_values))).tolist()
                for i, value in enumerate(sample_values, 1):
                    st.write(f"{i}. {value}")
            else:
                st.write("No non-null values to display")

def create_visualizations(df: pd.DataFrame):
    """Create interactive visualizations"""
    
    st.subheader("📊 Data Visualizations")
    
    # Select columns for visualization
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_columns) > 0:
        st.write("**Numeric Data Distribution**")
        
        # Select column for histogram
        selected_numeric = st.selectbox("Select numeric column for distribution:", numeric_columns)
        
        if selected_numeric:
            fig = px.histogram(df, x=selected_numeric, title=f"Distribution of {selected_numeric}")
            st.plotly_chart(fig, use_container_width=True)
    
    if len(categorical_columns) > 0 and len(categorical_columns) < 20:  # Avoid too many categories
        st.write("**Categorical Data Overview**")
        
        selected_categorical = st.selectbox("Select categorical column:", categorical_columns)
        
        if selected_categorical:
            value_counts = df[selected_categorical].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values, 
                        title=f"Top 10 values in {selected_categorical}")
            st.plotly_chart(fig, use_container_width=True)
    
    # Time series if date columns exist
    date_columns = [col for col in df.columns if any(date_indicator in col.lower() 
                   for date_indicator in ['date', 'time', '_on', 'period'])]
    
    if date_columns and len(numeric_columns) > 0:
        st.write("**Time Series Analysis**")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.selectbox("Select date column:", date_columns)
        with col2:
            selected_metric = st.selectbox("Select metric:", numeric_columns)
        
        if selected_date and selected_metric:
            # Sample data if too large
            if len(df) > 10000:
                df_sample = df.sample(n=10000)
                st.info("📊 Showing sample of 10,000 records for performance")
            else:
                df_sample = df
            
            try:
                # Convert to datetime if needed
                if df_sample[selected_date].dtype == 'object':
                    df_sample[selected_date] = pd.to_datetime(df_sample[selected_date], errors='coerce')
                
                # Create time series plot
                df_ts = df_sample.groupby(selected_date)[selected_metric].mean().reset_index()
                fig = px.line(df_ts, x=selected_date, y=selected_metric, 
                            title=f"{selected_metric} over time")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"⚠️ Could not create time series plot: {str(e)}")

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="ONGC Equipment Analysis Dashboard",
        page_icon="🛢️",
        layout="wide"
    )
    
    st.title("🛢️ ONGC Equipment Analysis Dashboard")
    st.markdown("**Enhanced with Principal Data Scientist Performance Optimization**")
    
    # Sidebar for performance monitoring
    st.sidebar.title("📊 Performance Monitor")
    
    # Initial memory usage
    initial_memory = get_memory_usage()
    st.sidebar.metric("System Memory", f"{initial_memory['process_memory_mb']:.1f} MB")
    st.sidebar.metric("System Memory %", f"{initial_memory['system_memory_percent']:.1f}%")
    
    # Load data
    with st.spinner("🔄 Loading optimized data..."):
        result = load_optimized_data()
        if result[0] is not None:
            df, data_name = result
        else:
            df, data_name = None, None
    
    if df is not None:
        # Memory optimization options
        st.sidebar.subheader("🔧 Memory Optimization")
        
        if st.sidebar.button("🗂️ Optimize Data Types"):
            with st.spinner("Optimizing data types..."):
                df = optimize_dtypes(df)
                st.sidebar.success("Data types optimized!")
        
        if st.sidebar.button("📉 Reduce Memory Usage"):
            with st.spinner("Reducing memory usage..."):
                df = reduce_memory_usage(df)
                st.sidebar.success("Memory usage reduced!")
        
        if st.sidebar.button("🧹 Memory Cleanup"):
            memory_cleanup()
            st.sidebar.success("Memory cleaned up!")
        
        # Main content - Show data cleaning summary
        show_data_cleaning_summary(df, data_name)
        
        # Data preview
        st.subheader("🔍 Data Preview")
        
        # Allow user to select number of rows to display
        num_rows = st.slider("Number of rows to display:", 5, 100, 10)
        st.dataframe(df.head(num_rows))
        
        # Data info
        with st.expander("ℹ️ Data Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Column Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Data Type': df.dtypes.values
                })
                st.dataframe(dtype_df)
            
            with col2:
                st.write("**Missing Values:**")
                missing_df = pd.DataFrame({
                    'Column': df.columns,
                    'Missing Count': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0]
                if len(missing_df) > 0:
                    st.dataframe(missing_df)
                else:
                    st.success("✅ No missing values found!")
        
        # Visualizations
        create_visualizations(df)
        
        # Performance report
        if st.sidebar.button("📈 Generate Performance Report"):
            report = get_performance_report()
            
            st.sidebar.subheader("🎯 Performance Summary")
            st.sidebar.json(report)
            
            # Show detailed metrics
            if report['performance_metrics']:
                st.subheader("⚡ Detailed Performance Metrics")
                
                metrics_df = pd.DataFrame.from_dict(report['performance_metrics'], orient='index')
                st.dataframe(metrics_df)
        
        # Update memory usage
        current_memory = get_memory_usage()
        st.sidebar.metric(
            "Current Memory", 
            f"{current_memory['process_memory_mb']:.1f} MB",
            delta=f"{current_memory['process_memory_mb'] - initial_memory['process_memory_mb']:.1f} MB"
        )
        
    else:
        st.info("👨‍💻 To get started:")
        st.markdown("""
        1. Run the main data pipeline: `python main.py`
        2. This will process your raw data and create optimized datasets
        3. Refresh this dashboard to view the results
        """)
        
        # Show available files
        st.subheader("📁 Available Data Files")
        
        # Check all possible data directories
        data_directories = [
            ("Processed Data", PROCESSED_DATA_DIR),
            ("Features Data", FEATURES_DATA_DIR),
        ]
        
        files_found = False
        for dir_name, dir_path in data_directories:
            if dir_path.exists():
                # Look for data files
                data_files = (list(dir_path.glob("*.parquet")) + 
                            list(dir_path.glob("*.csv")) + 
                            list(dir_path.glob("*.xlsx")))
                
                if data_files:
                    files_found = True
                    st.write(f"**{dir_name} ({dir_path.name}):**")
                    
                    for file_path in data_files:
                        file_size = file_path.stat().st_size / (1024*1024)  # MB
                        st.write(f"📄 {file_path.name} ({file_size:.1f} MB)")
                        
                        # Add button to load this specific file
                        if st.button(f"🔄 Load {file_path.name}", key=f"load_{file_path.name}"):
                            try:
                                if file_path.suffix == '.parquet':
                                    temp_df = pd.read_parquet(file_path)
                                elif file_path.suffix == '.csv':
                                    temp_df = pd.read_csv(file_path)
                                else:
                                    st.warning(f"Unsupported file format: {file_path.suffix}")
                                    continue
                                
                                st.success(f"✅ Loaded {file_path.name}: {temp_df.shape}")
                                st.dataframe(temp_df.head())
                                
                            except Exception as e:
                                st.error(f"❌ Error loading {file_path.name}: {str(e)}")
                    
                    st.write("---")
        
        if not files_found:
            st.write("No processed files found")
            
            # Show instructions to create data
            st.info("💡 **How to create processed data:**")
            st.code("python main.py", language="bash")
            st.write("This will:")
            st.write("• Load and concatenate raw Excel files")
            st.write("• Clean and preprocess the data")
            st.write("• Apply feature engineering")
            st.write("• Create optimized output files")

if __name__ == "__main__":
    main()
