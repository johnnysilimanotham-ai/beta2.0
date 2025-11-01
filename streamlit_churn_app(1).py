"""
Customer Churn Prediction App
Save this as: streamlit_app.py
Run with: streamlit run streamlit_app.py
"""

import io
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Churn Prediction Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = None
if 'production_model' not in st.session_state:
    st.session_state.production_model = None
if 'feature_schema' not in st.session_state:
    st.session_state.feature_schema = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# ==================== HELPER FUNCTIONS ====================

def normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Normalize target to 0/1 format"""
    if target_col not in df.columns:
        return df
    
    s = df[target_col].copy()
    
    # If it's already numeric, just ensure it's 0/1
    if pd.api.types.is_numeric_dtype(s):
        df[target_col] = pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).astype(int)
    # If it's object/string type
    elif s.dtype == "O":
        def to_bin(x):
            if x is None or (isinstance(x, float) and pd.isna(x)): 
                return 0
            xs = str(x).strip().lower()
            if xs in {"yes", "true", "1", "1.0", "y", "t"}: 
                return 1
            if xs in {"no", "false", "0", "0.0", "n", "f"}: 
                return 0
            return 0
        df[target_col] = s.map(to_bin).astype(int)
    else:
        df[target_col] = s.astype(int)
    
    return df

def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Clean and prepare data"""
    df = df.copy()
    
    # Drop ID columns first
    id_cols = [c for c in df.columns if 'id' in c.lower()]
    if id_cols:
        df = df.drop(columns=id_cols)
    
    # Normalize target column
    df = normalize_target(df, target_col)
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Verify we have both classes
    unique_classes = df[target_col].unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target column must have at least 2 classes. Found only: {unique_classes}")
    
    return df

def build_preprocessor(df: pd.DataFrame, target_col: str):
    """Build preprocessing pipeline"""
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]
    
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    
    schema = {
        "target": target_col,
        "numeric_features": num_cols,
        "categorical_features": cat_cols
    }
    
    return preprocessor, schema

def train_models(df: pd.DataFrame, target_col: str, test_size: float = 0.2):
    """Train all models and return results"""
    df = df.copy()
    
    # Ensure target is int type
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    
    # Check for class balance before proceeding
    class_counts = y.value_counts()
    st.info(f"📊 Class distribution: {class_counts.to_dict()}")
    
    if len(class_counts) < 2:
        st.error(f"❌ Only found class {y.unique()[0]} in the data. Cannot train models.")
        st.stop()
    
    preprocessor, schema = build_preprocessor(df, target_col)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models_def = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (name, model) in enumerate(models_def.items()):
        status_text.text(f"Training {name}...")
        
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
        }
        results.append(metrics)
        
        trained_models[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        
        progress_bar.progress((idx + 1) / len(models_def))
    
    status_text.empty()
    progress_bar.empty()
    
    results_df = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)
    
    return results_df, trained_models, schema, X_test

# ==================== MAIN APP ====================

st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox(
    "🔍 Navigation",
    ["📤 Upload Data", "🧹 Data Cleaning", "📊 Data Visualization", "🤖 Train Models", "📈 Model Comparison", "🔮 Make Predictions"]
)

# ==================== PAGE 1: UPLOAD DATA ====================
if page == "📤 Upload Data":
    st.header("📤 Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data",
        type=['csv'],
        help="Your dataset should include customer features and a churn column"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df_raw = df
        
        st.success(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📊 Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True)

# ==================== PAGE 2: DATA CLEANING ====================
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning & Preparation")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        # Use cleaned data if available, otherwise start with raw
        if st.session_state.df_clean is not None:
            df = st.session_state.df_clean.copy()
            st.info("📋 Working with previously cleaned data. You can make additional changes below.")
        else:
            df = st.session_state.df_raw.copy()
        
        st.subheader("⚙️ Target Column Selection")
        
        # Target column selection
        target_col = st.selectbox(
            "Select Target Column (Churn)",
            options=df.columns.tolist(),
            index=df.columns.tolist().index(st.session_state.target_col) if st.session_state.target_col and st.session_state.target_col in df.columns else (len(df.columns)-1 if 'churn' in str(df.columns).lower() else 0)
        )
        
        st.session_state.target_col = target_col
        
        # Show current data quality issues
        st.subheader("🔍 Data Quality Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())
        
        # Missing values details
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.subheader("⚠️ Columns with Missing Values")
            st.dataframe(missing_df, use_container_width=True)
        
        # Granular cleaning options
        st.subheader("🛠️ Cleaning Operations")
        
        with st.expander("🗑️ Remove Duplicate Rows", expanded=False):
            if df.duplicated().sum() > 0:
                st.write(f"Found **{df.duplicated().sum()}** duplicate rows")
                if st.button("Remove Duplicates"):
                    df = df.drop_duplicates()
                    st.session_state.df_clean = df
                    st.success(f"✅ Removed duplicates! Now {len(df)} rows.")
                    st.rerun()
            else:
                st.info("No duplicates found")
        
        with st.expander("🔢 Handle Missing Values (Column-by-Column)", expanded=True):
            if missing_df.empty:
                st.info("No missing values found!")
            else:
                st.write("💡 **Tip:** Different strategies work better for different features. Try different approaches and check visualizations!")
                
                cols_with_missing = missing_df['Column'].tolist()
                selected_col = st.selectbox("Select column to handle", cols_with_missing)
                
                if selected_col:
                    col_data = df[selected_col]
                    st.write(f"**{selected_col}**: {col_data.isnull().sum()} missing values ({(col_data.isnull().sum()/len(df)*100):.1f}%)")
                    
                    # Determine if numeric or categorical
                    is_numeric = pd.api.types.is_numeric_dtype(col_data)
                    
                    if is_numeric:
                        strategy = st.radio(
                            f"Strategy for {selected_col}",
                            ["Keep missing (let model handle)", "Fill with Mean", "Fill with Median", "Fill with Zero", "Drop rows with missing"],
                            key=f"strategy_{selected_col}"
                        )
                    else:
                        strategy = st.radio(
                            f"Strategy for {selected_col}",
                            ["Keep missing (let model handle)", "Fill with Most Frequent", "Fill with 'Unknown'", "Drop rows with missing"],
                            key=f"strategy_{selected_col}"
                        )
                    
                    if st.button(f"Apply to {selected_col}", type="primary"):
                        if strategy == "Keep missing (let model handle)":
                            st.info("✅ Keeping missing values. The model preprocessing will handle them.")
                        elif strategy == "Fill with Mean":
                            df[selected_col].fillna(df[selected_col].mean(), inplace=True)
                            st.success(f"✅ Filled {selected_col} with mean value: {df[selected_col].mean():.2f}")
                        elif strategy == "Fill with Median":
                            df[selected_col].fillna(df[selected_col].median(), inplace=True)
                            st.success(f"✅ Filled {selected_col} with median value: {df[selected_col].median():.2f}")
                        elif strategy == "Fill with Zero":
                            df[selected_col].fillna(0, inplace=True)
                            st.success(f"✅ Filled {selected_col} with zeros")
                        elif strategy == "Fill with Most Frequent":
                            most_frequent = df[selected_col].mode()[0]
                            df[selected_col].fillna(most_frequent, inplace=True)
                            st.success(f"✅ Filled {selected_col} with most frequent value: {most_frequent}")
                        elif strategy == "Fill with 'Unknown'":
                            df[selected_col].fillna('Unknown', inplace=True)
                            st.success(f"✅ Filled {selected_col} with 'Unknown'")
                        elif strategy == "Drop rows with missing":
                            before = len(df)
                            df = df.dropna(subset=[selected_col])
                            st.success(f"✅ Dropped {before - len(df)} rows with missing {selected_col}")
                        
                        st.session_state.df_clean = df
                        st.rerun()
        
        with st.expander("🎯 Clean Target Column", expanded=False):
            st.write(f"Current target: **{target_col}**")
            st.write(f"Unique values: {df[target_col].unique()}")
            st.write(f"Value counts:\n{df[target_col].value_counts()}")
            
            if st.button("Normalize Target to 0/1"):
                try:
                    df = normalize_target(df, target_col)
                    df = df.dropna(subset=[target_col])
                    st.session_state.df_clean = df
                    st.success("✅ Target normalized!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with st.expander("🗂️ Remove ID Columns", expanded=False):
            id_cols = [c for c in df.columns if 'id' in c.lower()]
            if id_cols:
                st.write(f"Found ID columns: {id_cols}")
                if st.button("Remove ID Columns"):
                    df = df.drop(columns=id_cols)
                    st.session_state.df_clean = df
                    st.success(f"✅ Removed {len(id_cols)} ID columns")
                    st.rerun()
            else:
                st.info("No ID columns detected")
        
        # Quick clean all button
        st.subheader("⚡ Quick Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Auto-Clean All", help="Automatically clean: remove IDs, duplicates, normalize target"):
                try:
                    with st.spinner("Auto-cleaning..."):
                        df_clean = clean_data(df, target_col)
                        st.session_state.df_clean = df_clean
                        st.session_state.target_col = target_col
                    st.success("✅ Auto-cleaning complete!")
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("↩️ Reset to Original Data"):
                st.session_state.df_clean = None
                st.session_state.target_col = None
                st.info("🔄 Reset to original data")
                st.rerun()
        
        # Show current state
        st.subheader("📊 Current Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show target distribution if available
        if target_col in df.columns:
            st.subheader("🎯 Target Distribution")
            try:
                churn_counts = df[target_col].value_counts().sort_index()
                
                col1, col2 = st.columns(2)
                with col1:
                    labels = [str(v) for v in churn_counts.index]
                    fig = px.pie(
                        values=churn_counts.values,
                        names=labels,
                        title=f"{target_col} Distribution",
                        color_discrete_sequence=['#667eea', '#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write("**Value Counts:**")
                    for idx, count in churn_counts.items():
                        st.metric(str(idx), count)
            except:
                st.write(df[target_col].value_counts())

# ==================== PAGE 3: DATA VISUALIZATION ====================
elif page == "📊 Data Visualization":
    st.header("📊 Data Visualization & Exploratory Analysis")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        # Use cleaned data if available, otherwise use raw
        df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        data_type = "Cleaned" if st.session_state.df_clean is not None else "Raw"
        
        st.info(f"📋 Analyzing **{data_type} Data** with {len(df)} rows and {len(df.columns)} columns")
        
        # Key insights section
        st.subheader("🔍 Key Data Insights")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Features", len(df.columns))
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            st.metric("Numeric Features", len(numeric_cols))
        with col3:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            st.metric("Categorical Features", len(categorical_cols))
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
            st.metric("Missing Data %", f"{missing_pct:.1f}%")
        
        # Missing data analysis
        st.subheader("⚠️ Missing Data Analysis")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Percent': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing'] > 0].sort_values('Missing', ascending=False)
        
        if len(missing_data) > 0:
            col1, col2 = st.columns([2, 1])
            with col1:
                fig = px.bar(
                    missing_data,
                    x='Column',
                    y='Percent',
                    title="Missing Data by Feature",
                    labels={'Percent': 'Missing %'},
                    color='Percent',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(missing_data, use_container_width=True)
            
            st.info("💡 **Insight:** Features with >50% missing data might need to be dropped. Features with <5% missing can often be imputed.")
        else:
            st.success("✅ No missing data found!")
        
        # Numeric features distribution
        if numeric_cols:
            st.subheader("📈 Numeric Features Distribution")
            st.write("💡 **Insight:** Look for skewed distributions that might benefit from transformation, and outliers that might need handling.")
            
            selected_numeric = st.multiselect(
                "Select numeric features to visualize",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
            
            if selected_numeric:
                cols = st.columns(2)
                for idx, col in enumerate(selected_numeric):
                    with cols[idx % 2]:
                        fig = px.histogram(
                            df, 
                            x=col, 
                            title=f"Distribution of {col}",
                            color_discrete_sequence=['#667eea'],
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Stats summary
                        with st.expander(f"📊 Statistics for {col}"):
                            stats_df = pd.DataFrame({
                                'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
                                'Value': [
                                    df[col].mean(),
                                    df[col].median(),
                                    df[col].std(),
                                    df[col].min(),
                                    df[col].max(),
                                    df[col].skew()
                                ]
                            })
                            st.dataframe(stats_df, use_container_width=True)
        
        # Categorical features
        if categorical_cols:
            st.subheader("📊 Categorical Features Analysis")
            st.write("💡 **Insight:** High cardinality (many unique values) might require encoding strategies or grouping rare categories.")
            
            selected_cat = st.selectbox("Select categorical feature", categorical_cols)
            
            if selected_cat:
                unique_count = df[selected_cat].nunique()
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    value_counts = df[selected_cat].value_counts().head(15)
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        labels={'x': selected_cat, 'y': 'Count'},
                        title=f"Top 15 values in {selected_cat}",
                        color_discrete_sequence=['#764ba2']
                    )
                    fig.update_xaxis(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.metric("Unique Values", unique_count)
                    st.metric("Most Common", value_counts.index[0])
                    st.metric("Frequency", value_counts.values[0])
                    
                    if unique_count > 20:
                        st.warning(f"⚠️ High cardinality: {unique_count} unique values")
                    else:
                        st.success("✅ Manageable cardinality")
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.subheader("🔥 Correlation Analysis")
            st.write("💡 **Insight:** Highly correlated features (>0.9) might be redundant. Target correlation shows feature importance.")
            
            corr_matrix = df[numeric_cols].corr()
            
            # Show full heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                title="Feature Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect="auto",
                zmin=-1,
                zmax=1
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight strong correlations
            with st.expander("🔍 Strong Correlations (>0.7 or <-0.7)"):
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > 0.7:
                            strong_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': corr_matrix.iloc[i, j]
                            })
                
                if strong_corr:
                    st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
                    st.warning("⚠️ Consider removing one feature from highly correlated pairs")
                else:
                    st.success("✅ No strong multicollinearity detected")
        
        # Target relationship analysis (if target is available)
        if st.session_state.target_col and st.session_state.target_col in df.columns:
            st.subheader("🎯 Features vs Target Relationship")
            st.write("💡 **Insight:** Features showing clear separation between classes are likely to be predictive.")
            
            target_col = st.session_state.target_col
            
            # Correlation with target for numeric features
            if numeric_cols and target_col in numeric_cols:
                target_corr = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
                target_corr = target_corr[target_corr.index != target_col]
                
                if len(target_corr) > 0:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(
                            x=target_corr.index[:10],
                            y=target_corr.values[:10],
                            title="Top 10 Features by Target Correlation",
                            labels={'x': 'Feature', 'y': 'Absolute Correlation'},
                            color=target_corr.values[:10],
                            color_continuous_scale='Viridis'
                        )
                        fig.update_xaxis(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.write("**Top Correlated Features:**")
                        for feat, corr in target_corr.head(5).items():
                            st.metric(feat, f"{corr:.3f}")
            
            # Box plots for numeric features
            if numeric_cols:
                selected_feature = st.selectbox(
                    "Select feature to compare with target",
                    [col for col in numeric_cols if col != target_col]
                )
                
                if selected_feature:
                    fig = px.box(
                        df,
                        x=target_col,
                        y=selected_feature,
                        title=f"{selected_feature} by {target_col}",
                        color=target_col,
                        color_discrete_sequence=['#667eea', '#764ba2']
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations section
        st.subheader("💡 Recommendations for Model Training")
        
        recommendations = []
        
        # Check for missing data
        if missing_pct > 10:
            recommendations.append("⚠️ **High missing data** - Consider imputation strategies or removing features with >50% missing")
        
        # Check for imbalanced target
        if st.session_state.target_col and st.session_state.target_col in df.columns:
            target_dist = df[st.session_state.target_col].value_counts(normalize=True)
            if len(target_dist) >= 2 and target_dist.min() < 0.2:
                recommendations.append("⚠️ **Imbalanced target** - Consider using stratified sampling or SMOTE for model training")
        
        # Check for high cardinality
        high_card_features = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_card_features:
            recommendations.append(f"⚠️ **High cardinality features** - {', '.join(high_card_features[:3])} - Consider target encoding or feature hashing")
        
        # Check for skewed features
        if numeric_cols:
            skewed_features = [col for col in numeric_cols if abs(df[col].skew()) > 2]
            if skewed_features:
                recommendations.append(f"⚠️ **Skewed features** - {', '.join(skewed_features[:3])} - Consider log transformation")
        
        # Model suggestions
        if len(numeric_cols) > len(categorical_cols):
            recommendations.append("✅ **Mostly numeric data** - SVM and Logistic Regression likely to perform well")
        else:
            recommendations.append("✅ **Mixed/Categorical data** - Random Forest and Gradient Boosting recommended")
        
        if not recommendations:
            recommendations.append("✅ Data looks good! Ready for model training.")
        
        for rec in recommendations:
            st.write(rec)
        
        # Navigation hint
        st.info("💡 **Next Steps:** Review these insights, go back to Data Cleaning to make adjustments if needed, or proceed to Train Models when ready!")

# ==================== PAGE 4: TRAIN MODELS ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please clean your data first!")
    else:
        df_clean = st.session_state.df_clean
        target_col = st.session_state.target_col
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
        with col2:
            st.metric("Training Samples", int(len(df_clean) * (1 - test_size)))
        
        if st.button("🚀 Train All Models", type="primary"):
            with st.spinner("Training models... This may take a minute."):
                results, models, schema, X_test = train_models(df_clean, target_col, test_size)
                
                st.session_state.results = results
                st.session_state.models = models
                st.session_state.feature_schema = schema
                st.session_state.X_test = X_test
            
            st.success("✅ All models trained successfully!")
            st.balloons()
            
            st.subheader("📊 Training Results")
            st.dataframe(results.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

# ==================== PAGE 5: MODEL COMPARISON ====================
elif page == "📈 Model Comparison":
    st.header("📈 Model Performance Comparison")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please train models first!")
    else:
        results = st.session_state.results
        models = st.session_state.models
        
        # Metrics comparison
        st.subheader("📈 Performance Metrics")
        
        fig = go.Figure()
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            fig.add_trace(go.Bar(
                name=metric.upper(),
                x=results.index,
                y=results[metric],
                text=results[metric].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            barmode='group',
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model
        best_model = results.index[0]
        st.subheader("🏆 Best Model")
        st.success(f"**{best_model}** with F1 Score: {results.loc[best_model, 'f1']:.4f}")
        
        # Confusion matrices
        st.subheader("🔢 Confusion Matrices")
        cols = st.columns(3)
        
        for idx, (name, model_data) in enumerate(models.items()):
            with cols[idx]:
                cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="Actual"),
                    x=['No Churn', 'Churn'],
                    y=['No Churn', 'Churn'],
                    title=name,
                    color_continuous_scale='Purples'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Deploy to production
        st.subheader("🚀 Deploy Model")
        selected_model = st.selectbox("Select model for production", results.index.tolist())
        
        if st.button("Deploy to Production", type="primary"):
            with st.spinner(f"Retraining {selected_model} on full dataset..."):
                # Get the model class
                df_clean = st.session_state.df_clean
                target_col = st.session_state.target_col
                
                y = df_clean[target_col].astype(int)
                X = df_clean.drop(columns=[target_col])
                
                # Rebuild preprocessor and get model
                preprocessor, schema = build_preprocessor(df_clean, target_col)
                
                models_def = {
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
                    "SVM": SVC(kernel="rbf", probability=True, random_state=42)
                }
                
                # Create pipeline with selected model and train on FULL dataset
                model = models_def[selected_model]
                full_pipeline = Pipeline([("prep", preprocessor), ("model", model)])
                full_pipeline.fit(X, y)
                
                st.session_state.production_model = full_pipeline
                st.session_state.feature_schema = schema
            
            st.success(f"✅ {selected_model} retrained on full dataset and deployed to production!")
            st.info(f"📊 Model trained on all {len(df_clean)} samples")

# ==================== PAGE 6: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Predict Customer Churn")
    
    if st.session_state.production_model is None:
        st.warning("⚠️ Please deploy a model first!")
    else:
        pipe = st.session_state.production_model
        schema = st.session_state.feature_schema
        
        # Add prediction mode selector
        prediction_mode = st.radio(
            "Select Prediction Mode",
            ["Single Customer Prediction", "Batch Prediction (Upload CSV)"],
            horizontal=True
        )
        
        if prediction_mode == "Single Customer Prediction":
            st.subheader("🔍 Enter Customer Information")
            
            input_data = {}
            
            # Create input fields based on schema
            num_cols = schema['numeric_features']
            cat_cols = schema['categorical_features']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numeric Features**")
                for col in num_cols:
                    input_data[col] = st.number_input(f"{col}", value=0.0)
            
            with col2:
                st.markdown("**Categorical Features**")
                for col in cat_cols:
                    # Get unique values from cleaned data if available
                    if st.session_state.df_clean is not None and col in st.session_state.df_clean.columns:
                        options = st.session_state.df_clean[col].unique().tolist()
                        input_data[col] = st.selectbox(f"{col}", options)
                    else:
                        input_data[col] = st.text_input(f"{col}")
            
            if st.button("🎯 Predict Churn", type="primary"):
                # Create DataFrame with correct column order
                X = pd.DataFrame([input_data], columns=num_cols + cat_cols)
                
                prediction = pipe.predict(X)[0]
                proba = pipe.predict_proba(X)[0, 1]
                
                st.markdown("---")
                st.subheader("🔮 Prediction Result")
                
                if prediction == 1:
                    st.error("⚠️ **Customer will CHURN**")
                    st.markdown(f"### Churn Probability: {proba*100:.1f}%")
                else:
                    st.success("✅ **Customer will NOT churn**")
                    st.markdown(f"### Retention Probability: {(1-proba)*100:.1f}%")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={'text': "Churn Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if proba > 0.5 else "darkgreen"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        else:  # Batch Prediction
            st.subheader("📁 Upload CSV File for Batch Predictions")
            st.info("💡 Upload a CSV file with customer data. The file should NOT include the target column.")
            
            batch_file = st.file_uploader(
                "Upload CSV for batch predictions",
                type=['csv'],
                key="batch_upload"
            )
            
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                
                st.success(f"✅ File loaded: {len(batch_df)} customers")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("🎯 Predict All", type="primary"):
                    try:
                        with st.spinner("Making predictions..."):
                            # Ensure columns match expected features
                            num_cols = schema['numeric_features']
                            cat_cols = schema['categorical_features']
                            expected_cols = num_cols + cat_cols
                            
                            # Reorder columns to match training
                            X_batch = batch_df[expected_cols]
                            
                            # Make predictions
                            predictions = pipe.predict(X_batch)
                            probabilities = pipe.predict_proba(X_batch)[:, 1]
                            
                            # Add predictions to dataframe
                            result_df = batch_df.copy()
                            result_df['Churn_Prediction'] = predictions
                            result_df['Churn_Probability'] = probabilities
                            result_df['Risk_Level'] = pd.cut(
                                probabilities,
                                bins=[0, 0.33, 0.66, 1.0],
                                labels=['Low', 'Medium', 'High']
                            )
                        
                        st.success("✅ Predictions completed!")
                        
                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(result_df))
                        with col2:
                            st.metric("Predicted Churns", (predictions == 1).sum())
                        with col3:
                            churn_rate = (predictions == 1).mean() * 100
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        
                        # Risk distribution
                        st.subheader("📊 Risk Distribution")
                        risk_counts = result_df['Risk_Level'].value_counts()
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Customer Risk Distribution",
                            color_discrete_sequence=['#667eea', '#FFA500', '#764ba2']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.subheader("📋 Prediction Results")
                        st.dataframe(
                            result_df.sort_values('Churn_Probability', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.info("💡 Make sure your CSV has all the required features with correct names.")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 App Status")
if st.session_state.df_raw is not None:
    st.sidebar.success("✅ Data Uploaded")
else:
    st.sidebar.info("⏳ Awaiting Data")

if st.session_state.df_clean is not None:
    st.sidebar.success("✅ Data Cleaned")
else:
    st.sidebar.info("⏳ Awaiting Cleaning")

if st.session_state.results is not None:
    st.sidebar.success("✅ Models Trained")
else:
    st.sidebar.info("⏳ Awaiting Training")

if st.session_state.production_model is not None:
    st.sidebar.success("✅ Model Deployed")
else:
    st.sidebar.info("⏳ Awaiting Deployment")
