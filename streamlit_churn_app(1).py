"""
Customer Churn Prediction App
Save this as: streamlit_churn_app.py
Run with: streamlit run streamlit_churn_app.py
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
    page_title="Customer Retention Analysis",
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
if 'production_model_name' not in st.session_state:
    st.session_state.production_model_name = None
if 'feature_schema' not in st.session_state:
    st.session_state.feature_schema = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
# --- Cleaning Configuration ---
if 'imputation_strategies' not in st.session_state:
    st.session_state.imputation_strategies = {}
if 'columns_to_drop' not in st.session_state:
    st.session_state.columns_to_drop = []

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

def apply_custom_cleaning(df: pd.DataFrame, target_col: str, drop_cols: list, impute_strategies: dict) -> pd.DataFrame:
    """Apply all saved custom cleaning instructions."""
    df_clean = df.copy()
    
    # 1. Drop Unwanted Columns
    cols_to_drop = [col for col in drop_cols if col in df_clean.columns and col != target_col]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop, errors='ignore')
        st.write(f"✅ Dropped {len(cols_to_drop)} columns.")

    # 2. Handle Duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    if initial_rows != len(df_clean):
        st.write(f"✅ Removed {initial_rows - len(df_clean)} duplicate rows. Now {len(df_clean)} rows.")
    else:
        st.write("✅ No duplicates found or removed.")


    # 3. Normalize Target and Drop Rows with Missing Target
    target_missing_before = df_clean[target_col].isnull().sum() if target_col in df_clean.columns else 0
    try:
        df_clean = normalize_target(df_clean, target_col)
        df_clean = df_clean.dropna(subset=[target_col])
        if target_missing_before > 0:
            st.write("✅ Target normalized to 0/1 and rows with missing target dropped.")
        else:
             st.write("✅ Target column normalized.")
    except Exception as e:
         st.error(f"Target Cleaning Error: {str(e)}")
         st.stop()


    # 4. Handle Missing Values based on strategy
    st.write("---")
    st.write("### Applying Missing Value Strategies:")
    for col, strategy in impute_strategies.items():
        # Check if column still exists after dropping, and if it has missing values
        if col in df_clean.columns and df_clean[col].isnull().any():
            missing_count = df_clean[col].isnull().sum()
            
            if strategy == "Drop rows with missing":
                df_clean = df_clean.dropna(subset=[col])
                st.write(f"🗑️ Dropped {missing_count} rows due to missing **{col}**.")
            elif strategy == "Fill with Mean":
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                st.write(f"🔢 Filled **{col}** with Mean.")
            elif strategy == "Fill with Median":
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                st.write(f"🔢 Filled **{col}** with Median.")
            elif strategy == "Fill with Zero":
                df_clean[col].fillna(0, inplace=True)
                st.write(f"🔢 Filled **{col}** with Zero.")
            elif strategy == "Fill with Most Frequent":
                try:
                    most_frequent = df_clean[col].mode()[0]
                    df_clean[col].fillna(most_frequent, inplace=True)
                    st.write(f"🔠 Filled **{col}** with Most Frequent ({most_frequent}).")
                except IndexError:
                    st.warning(f"⚠️ Could not find mode for **{col}**, skipping imputation.")
            elif strategy == "Fill with 'Unknown'":
                df_clean[col].fillna('Unknown', inplace=True)
                st.write(f"🔠 Filled **{col}** with 'Unknown'.")
            elif strategy == "Keep missing (let model handle)":
                st.write(f"⏭️ Skipping **{col}** - missing values will be handled by the ML preprocessing pipeline.")
            else:
                 st.write(f"⚠️ **{col}**: Unknown strategy '{strategy}' skipped.")
        elif col in df_clean.columns:
            st.write(f"✅ **{col}** has no missing values.")
    
    # Final check for two classes
    if target_col in df_clean.columns:
        unique_classes = df_clean[target_col].unique()
        if len(unique_classes) < 2:
            raise ValueError(f"Target column must have at least 2 classes. Found only: {unique_classes}")
    
    return df_clean

def get_feature_influence(pipe: Pipeline, X_single_row: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Extracts top features influencing a single prediction for interpretability."""
    
    model = pipe.named_steps['model']
    preprocessor = pipe.named_steps['prep']
    
    # 1. Get the names of the encoded features (after preprocessor)
    try:
        # Access the OneHotEncoder step within the categorical transformer
        cat_transformer = preprocessor.named_transformers_['cat']
        ohe_names = cat_transformer.named_steps['encoder'].get_feature_names_out(schema['categorical_features'])
        feature_names = schema['numeric_features'] + ohe_names.tolist()
    except:
        # Fallback if preprocessing is simpler or structure is different
        feature_names = schema['numeric_features'] + schema['categorical_features']

    # 2. Extract importance based on model type
    importances = None
    influence_method = "Unknown"
    
    if isinstance(model, LogisticRegression):
        # We need the preprocessed data to calculate feature driver score (Coefficient * Feature_Value)
        importances = model.coef_[0]
        influence_method = "Logistic Regression Coefficients"
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        influence_method = "Built-in Feature Importances (MDI)"

    if importances is None:
        # st.warning("⚠️ Feature influence for this model type cannot be calculated.")
        return pd.DataFrame()
        
    st.info(f"💡 **Feature Influence Method:** {influence_method} used for interpretation.")
        
    # 3. Create a DataFrame for importance
    importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Influence_Score': importances
    }).sort_values(by='Influence_Score', key=lambda x: x.abs(), ascending=False)
    
    # 4. Calculate prediction-specific driver score for Logistic Regression 
    if isinstance(model, LogisticRegression):
        # Preprocess the single row to get the numerical feature vector
        X_processed = preprocessor.transform(X_single_row)
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Calculate the driver score (Coefficient * Preprocessed Value)
        # Ensure alignment: only features in importance_df are considered
        # Note: Features with zero coefficient might not be in the feature_names list from the pipeline if OHE dropped them,
        # but here we rely on the importance_df features which should align with coef_
        
        # We need to map the coefficients (importances) and the processed features correctly
        # The indices of feature_names match the indices of importances and X_processed
        
        # Re-index X_processed_df to match feature_names order, then grab the values for the features in importance_df
        processed_values = X_processed_df.iloc[0].loc[importance_df['Feature']].values
        
        importance_df['Prediction_Driver'] = importance_df['Influence_Score'] * processed_values
        
        # Keep only features where the driver score is not near zero (i.e., the feature was actually present)
        importance_df = importance_df[abs(importance_df['Prediction_Driver']) > 1e-4]
        
        return importance_df.sort_values(by='Prediction_Driver', key=lambda x: x.abs(), ascending=False).head(5)

    # For other models, we'll just show the top 5 *global* importances as a proxy
    return importance_df.head(5)


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
    
    if target_col not in df.columns:
        st.error(f"❌ Target column '{target_col}' not found in the dataset.")
        st.stop()
        
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
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, max_iter=2000)
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
    ["📤 Upload Data", "📊 Data Visualization", "🧹 Data Cleaning", "🤖 Train Models", "📈 Model Comparison", "🔮 Make Predictions"]
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
        
        # ------------------------------------------------------------------
        # Target Column Selection
        st.subheader("🎯 Select Target Variable")
        
        # Calculate a sensible default index
        column_names = df.columns.tolist()
        default_index = len(column_names) - 1 # Fallback to last column
        if st.session_state.target_col and st.session_state.target_col in column_names:
            default_index = column_names.index(st.session_state.target_col)
        elif 'churn' in str(column_names).lower():
            try:
                # Find the first column containing 'churn' (case-insensitive)
                default_index = next(i for i, col in enumerate(column_names) if 'churn' in col.lower())
            except StopIteration:
                pass
            
        selected_target = st.selectbox(
            "Which column represents customer churn (The variable you want to predict)?",
            options=column_names,
            index=default_index,
            key="upload_target_select"
        )
        
        # Save the selection to session state
        if selected_target != st.session_state.target_col:
            st.session_state.target_col = selected_target
            # Clear previous cleaning/training results if target changes
            st.session_state.df_clean = None
            st.session_state.results = None
            st.session_state.production_model = None
            st.session_state.production_model_name = None
            # Also reset cleaning instructions if data/target changes
            st.session_state.imputation_strategies = {}
            st.session_state.columns_to_drop = []
            st.warning("Target column changed. Please re-run cleaning and training steps.")
        
        st.info(f"Target column set to: **{st.session_state.target_col}**")
        # ------------------------------------------------------------------
        
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), width='stretch')
        
        st.subheader("📊 Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Non-Null Count": df.count().values,
            "Null Count": df.isnull().sum().values
        })
        st.dataframe(info_df, width='stretch')

# ==================== PAGE 2: DATA VISUALIZATION ====================
elif page == "📊 Data Visualization":
    st.header("📊 Data Visualization & Exploratory Analysis")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
    elif st.session_state.target_col is None:
        st.warning("⚠️ Please select a target column on the 'Upload Data' page first!")
    else:
        # Use cleaned data if available, otherwise use raw
        df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        data_type = "Cleaned" if st.session_state.df_clean is not None else "Raw"
        target_col = st.session_state.target_col
        
        if target_col not in df.columns:
             st.error(f"❌ Target column '{target_col}' not found in the current dataset.")
             st.stop()
             
        st.info(f"📋 Analyzing **{data_type} Data** with {len(df)} rows and {len(df.columns)} columns. Target: **{target_col}**")
        
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
                st.plotly_chart(fig, width='stretch')
            with col2:
                st.dataframe(missing_data, width='stretch')
            
            st.info("💡 **Insight:** Features with >50% missing data might need to be dropped. Features with <5% missing can often be imputed.")
        else:
            st.success("✅ No missing data found!")
        
        # Numeric features distribution
        if numeric_cols:
            st.subheader("📈 Numeric Features Distribution")
            st.write("💡 **Insight:** Look for skewed distributions that might benefit from transformation, and outliers that might need handling.")
            
            # Exclude target from numeric features for visualization selection if it's numeric
            vis_numeric_cols = [c for c in numeric_cols if c != target_col]
            
            selected_numeric = st.multiselect(
                "Select numeric features to visualize",
                vis_numeric_cols,
                default=vis_numeric_cols[:min(4, len(vis_numeric_cols))]
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
                        st.plotly_chart(fig, width='stretch')
                        
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
                            st.dataframe(stats_df, width='stretch')
        
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
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.metric("Unique Values", unique_count)
                    st.metric("Most Common", value_counts.index[0])
                    st.metric("Frequency", value_counts.values[0])
                    
                    if unique_count > 20:
                        st.warning(f"⚠️ High cardinality: {unique_count} unique values")
                    else:
                        st.success("✅ Manageable cardinality")
        
        # Correlation analysis
        numeric_cols_no_target = [c for c in numeric_cols if c != target_col]
        if len(numeric_cols_no_target) >= 1: # Now checks if we have at least one numeric feature to correlate
            st.subheader("🔥 Correlation Analysis")
            st.write("💡 **Insight:** Highly correlated features (>0.9) might be redundant. Target correlation shows feature importance.")
            
            # --- Feature-Feature Correlation Heatmap ---
            if len(numeric_cols_no_target) > 1:
                corr_matrix = df[numeric_cols_no_target].corr()
                
                # Show full heatmap
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    title="Feature Correlation Heatmap (Excluding Target)",
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    zmin=-1,
                    zmax=1
                )
                st.plotly_chart(fig, width='stretch')
                
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
                        st.dataframe(pd.DataFrame(strong_corr), width='stretch')
                        st.warning("⚠️ Consider removing one feature from highly correlated pairs")
                    else:
                        st.success("✅ No strong multicollinearity detected")
            
            # --- Target relationship analysis (if target is available) ---
            st.subheader("🎯 Features vs Target Relationship")
            st.write("💡 **Insight:** Features showing clear separation between classes are likely to be predictive.")
            
            # Correlation with target for numeric features
            if numeric_cols_no_target:
                # Ensure target column is numeric (0/1) for correlation calculation
                df_temp = df.copy()
                df_temp = normalize_target(df_temp, target_col)
                
                target_corr = df_temp[numeric_cols_no_target + [target_col]].corrwith(df_temp[target_col]).abs().sort_values(ascending=False)
                target_corr = target_corr.drop(target_col, errors='ignore')
                
                if len(target_corr) > 0:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(
                            x=target_corr.index[:10],
                            y=target_corr.values[:10],
                            title="Top 10 Numeric Features by Target Correlation",
                            labels={'x': 'Feature', 'y': 'Absolute Correlation'},
                            color=target_corr.values[:10],
                            color_continuous_scale='Viridis'
                        )
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        st.write("**Top Correlated Features:**")
                        for feat, corr in target_corr.head(5).items():
                            st.metric(feat, f"{corr:.3f}")
            
            # Box plots for numeric features
            if numeric_cols_no_target:
                selected_feature = st.selectbox(
                    "Select numeric feature to compare with target",
                    numeric_cols_no_target
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
                    st.plotly_chart(fig, width='stretch')
        
        # Recommendations section
        st.subheader("💡 Recommendations for Model Training")
        
        recommendations = []
        
        # Check for missing data
        if missing_pct > 10:
            recommendations.append("⚠️ **High missing data** - Consider imputation strategies or removing features with >50% missing")
        
        # Check for imbalanced target
        target_dist = df[target_col].value_counts(normalize=True)
        if len(target_dist) >= 2 and target_dist.min() < 0.2:
            recommendations.append("⚠️ **Imbalanced target** - Consider using stratified sampling or SMOTE for model training")
        
        # Check for high cardinality
        high_card_features = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_card_features:
            recommendations.append(f"⚠️ **High cardinality features** - {', '.join(high_card_features[:3])} - Consider target encoding or feature hashing")
        
        # Check for skewed features
        if numeric_cols_no_target:
            skewed_features = [col for col in numeric_cols_no_target if abs(df[col].skew()) > 2]
            if skewed_features:
                recommendations.append(f"⚠️ **Skewed features** - {', '.join(skewed_features[:3])} - Consider log transformation")
        
        # Model suggestions
        if len(numeric_cols_no_target) > len(categorical_cols):
            recommendations.append("✅ **Mostly numeric data** - SVM and Logistic Regression likely to perform well")
        else:
            recommendations.append("✅ **Mixed/Categorical data** - Random Forest and Gradient Boosting recommended")
        
        if not recommendations:
            recommendations.append("✅ Data looks good! Ready for model training.")
        
        for rec in recommendations:
            st.write(rec)
        
        # Navigation hint
        st.info("💡 **Next Steps:** Review these insights, go back to Data Cleaning to make adjustments if needed, or proceed to Train Models when ready!")

# ==================== PAGE 3: DATA CLEANING ====================
elif page == "🧹 Data Cleaning":
    st.header("🧹 Data Cleaning & Preparation")
    
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please upload a dataset first!")
        st.stop()
    
    if st.session_state.target_col is None:
        st.warning("⚠️ Please select a target column on the 'Upload Data' page first!")
        st.stop()
        
    target_col = st.session_state.target_col
    
    # Use raw data for configuration, the cleaning will be applied on button click
    df = st.session_state.df_raw.copy()
    
    st.info(f"📋 Configuring cleaning steps for **Raw Data** with {len(df)} rows. Target: **{target_col}**")

    # Display Target Column (No longer a selectbox, just a display)
    st.subheader("⚙️ Target Column (Set on Upload Page)")
    st.write(f"The current target column is: **{target_col}**")
    
    if target_col not in df.columns:
        st.error(f"❌ Selected target column '{target_col}' is missing from the dataset. Please go back to 'Upload Data'.")
        st.stop()
    
    # Show current data quality issues
    st.subheader("🔍 Raw Data Quality Overview")
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
        'Percent': (df.isnull().sum().values / len(df) * 100).round(2),
        'DType': df.dtypes.astype(str)
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percent', ascending=False)
    
    # ------------------------------------------------------------------
    # 1. Column Dropping
    st.subheader("🗑️ 1. Drop Columns")
    current_cols = [c for c in df.columns if c != target_col]
    
    # Preserve current state for the multiselect if possible
    default_drop_cols = [c for c in st.session_state.columns_to_drop if c in current_cols]
    
    st.session_state.columns_to_drop = st.multiselect(
        "Select columns to permanently **drop** from the dataset (e.g., ID columns, features with too many missing values, or non-predictive features):",
        options=current_cols,
        default=default_drop_cols
    )
    
    st.info(f"Columns selected to drop: **{len(st.session_state.columns_to_drop)}**")
    
    # ------------------------------------------------------------------
    # 2. Missing Value Imputation
    st.subheader("🩹 2. Missing Value Imputation / Handling")
    
    cols_with_missing = missing_df['Column'].tolist()
    
    if not cols_with_missing:
        st.success("✅ No missing values found in the raw dataset!")
    else:
        st.write("Configure how to handle missing values for the following columns:")
        
        # Filter for columns that are not selected for dropping
        cols_for_imputation = [c for c in cols_with_missing if c not in st.session_state.columns_to_drop]
        
        if cols_for_imputation:
            # Display configuration for each column
            col_types = df[cols_for_imputation].dtypes
            
            for col in cols_for_imputation:
                dtype = col_types[col]
                default_strategy = st.session_state.imputation_strategies.get(col, "Keep missing (let model handle)")
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    strategies = ["Keep missing (let model handle)", "Drop rows with missing", "Fill with Mean", "Fill with Median", "Fill with Zero"]
                    tooltip = "Use Mean/Median for normal distributions, Zero for count-like features."
                else: # Categorical/Object
                    strategies = ["Keep missing (let model handle)", "Drop rows with missing", "Fill with Most Frequent", "Fill with 'Unknown'"]
                    tooltip = "Use Most Frequent or 'Unknown' for categorical features."
                
                # Update strategy in session state
                st.session_state.imputation_strategies[col] = st.selectbox(
                    f"**{col}** (Type: *{dtype}* | Missing: {df[col].isnull().sum()} rows)",
                    options=strategies,
                    index=strategies.index(default_strategy),
                    key=f"impute_select_{col}",
                    help=tooltip
                )
        else:
            st.info("No columns with missing data remaining after drop selection.")

    # ------------------------------------------------------------------
    # 3. Apply Cleaning Button
    st.subheader("🚀 3. Apply Cleaning Steps")
    
    if st.button("✨ Apply and View Cleaned Data", key="apply_cleaning_button"):
        try:
            with st.spinner("Applying cleaning steps..."):
                # Pass the raw data, target, configured drops, and imputation strategies
                df_clean = apply_custom_cleaning(
                    df=st.session_state.df_raw,
                    target_col=target_col,
                    drop_cols=st.session_state.columns_to_drop,
                    impute_strategies=st.session_state.imputation_strategies
                )
                
                st.session_state.df_clean = df_clean
                
                # Clear model results as data has changed
                st.session_state.results = None
                st.session_state.models = {}
                st.session_state.production_model = None
                st.session_state.production_model_name = None
                
                st.success("✅ Data cleaning complete!")
                
                st.subheader("🧹 Cleaned Data Preview")
                st.write(f"Final Data Shape: {df_clean.shape}")
                
                target_counts = df_clean[target_col].value_counts().sort_index()
                st.write("Target Class Distribution (0=No Churn, 1=Churn):")
                st.dataframe(target_counts, use_container_width=True)
                
                st.dataframe(df_clean.head(), use_container_width=True)
                
        except ValueError as e:
            st.error(f"Cleaning Error: {e}")
        except Exception as e:
            st.exception(e)
    
    # Show cleaned data if it exists
    if st.session_state.df_clean is not None:
        st.subheader("✅ Currently Cleaned Data (Ready for Training)")
        st.write(f"Current Shape: {st.session_state.df_clean.shape}")
        st.info("💡 Changes in configuration will only apply after clicking the 'Apply Cleaning' button.")
        
# ==================== PAGE 4: TRAIN MODELS ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.df_clean is None:
        st.warning("⚠️ Please complete the 'Data Cleaning' step first!")
        st.stop()
    
    df_clean = st.session_state.df_clean
    target_col = st.session_state.target_col
    
    st.info(f"🎯 Training models on **{len(df_clean)} rows** using **{len(df_clean.columns)} features**. Target: **{target_col}**")
    
    st.subheader("⚙️ Training Configuration")
    test_size = st.slider("Test Set Size (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
    
    if st.button("🚀 Train All Models", key="train_models_button"):
        try:
            with st.spinner("Training models: Logistic Regression, Random Forest, and SVM... This may take a moment."):
                
                # Check for two classes again
                if df_clean[target_col].nunique() < 2:
                     st.error(f"❌ Target column '{target_col}' has less than two unique classes after cleaning. Cannot train.")
                     st.stop()
                
                results_df, trained_models, feature_schema, X_test = train_models(df_clean, target_col, test_size)
                
                st.session_state.results = results_df
                st.session_state.models = trained_models
                st.session_state.feature_schema = feature_schema
                st.session_state.X_test = X_test # Store test data for later prediction/analysis
                
                st.success("✅ Model training complete!")
                st.balloons()
        except Exception as e:
            st.error("An error occurred during training.")
            st.exception(e)
            
    if st.session_state.results is not None:
        st.subheader("🏆 Model Performance Summary")
        
        # Highlight best F1 score
        def highlight_max(s):
            is_max = s == s.max()
            return ['background-color: #667eea; color: white' if v else '' for v in is_max]
            
        styled_results = st.session_state.results.style.apply(highlight_max, subset=['f1']).format({
            'accuracy': "{:.4f}",
            'precision': "{:.4f}",
            'recall': "{:.4f}",
            'f1': "{:.4f}",
            'roc_auc': "{:.4f}"
        })
        
        st.dataframe(styled_results, use_container_width=True)
        
        # Selection for Production Model
        best_model_name = st.session_state.results.index[0]
        
        st.subheader("🥇 Select Production Model")
        
        production_model_name = st.selectbox(
            f"Choose a model to deploy for predictions (Best based on F1: **{best_model_name}**)",
            options=st.session_state.results.index.tolist(),
            index=0,
            key="prod_model_select"
        )
        
        # Save the selected pipeline and schema
        st.session_state.production_model = st.session_state.models[production_model_name]['pipeline']
        st.session_state.production_model_name = production_model_name # Store name
        
        st.success(f"Production Model set to: **{production_model_name}**")
        
        # ROC Curve Plot
        st.subheader("📈 ROC Curve Comparison")
        fig = go.Figure()
        
        for name, data in st.session_state.models.items():
            if data['y_proba'] is not None:
                fpr, tpr, _ = roc_curve(data['y_test'], data['y_proba'])
                auc = st.session_state.results.loc[name, 'roc_auc']
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode='lines', name=f"{name} (AUC: {auc:.3f})"
                ))
        
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='grey'), name='Random Guess (AUC: 0.5)'))
        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Recall)',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
# ==================== PAGE 5: MODEL COMPARISON ====================
elif page == "📈 Model Comparison":
    st.header("📈 Model Comparison & Diagnostics")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please train the models first on the 'Train Models' page!")
        st.stop()
        
    st.subheader("📊 Performance Metrics")
    
    # Display the results table again
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #667eea; color: white' if v else '' for v in is_max]
            
    styled_results = st.session_state.results.style.apply(highlight_max, subset=['f1']).format({
        'accuracy': "{:.4f}",
        'precision': "{:.4f}",
        'recall': "{:.4f}",
        'f1': "{:.4f}",
        'roc_auc': "{:.4f}"
    })
    st.dataframe(styled_results, use_container_width=True)
    
    st.subheader("📉 Detailed Model Diagnostics")
    
    selected_model = st.selectbox(
        "Select Model for Detailed View",
        options=st.session_state.results.index.tolist()
    )
    
    if selected_model:
        model_data = st.session_state.models[selected_model]
        y_test = model_data['y_test']
        y_pred = model_data['y_pred']
        
        st.markdown(f"### Diagnostics for **{selected_model}**")
        
        col1, col2 = st.columns(2)
        
        # Confusion Matrix
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=['Actual 0 (No Churn)', 'Actual 1 (Churn)'], columns=['Predicted 0', 'Predicted 1'])
            
            fig_cm = px.imshow(
                cm_df, 
                text_auto=True, 
                color_continuous_scale='Blues',
                labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
            tn, fp, fn, tp = cm.ravel()
            
            st.markdown(f"""
            - **True Negatives (TN):** {tn} (Correctly predicted No Churn)
            - **False Positives (FP):** {fp} (Incorrectly predicted Churn - **Type I Error**)
            - **False Negatives (FN):** {fn} (Incorrectly predicted No Churn - **Type II Error**)
            - **True Positives (TP):** {tp} (Correctly predicted Churn)
            """)

        # Feature Importance
        with col2:
            st.subheader("Feature Importance / Influence")
            
            model_pipe = st.session_state.models[selected_model]['pipeline']
            schema = st.session_state.feature_schema
            
            # Feature Importance (only possible for models with feature_importances_ or coef_)
            if hasattr(model_pipe.named_steps['model'], 'feature_importances_') or isinstance(model_pipe.named_steps['model'], LogisticRegression):
                 
                # Use a dummy single row from X_test for coefficient-based interpretation if LR
                X_single = st.session_state.X_test.iloc[[0]] 
                
                importance_df = get_feature_influence(
                    pipe=model_pipe, 
                    X_single_row=X_single,
                    schema=schema
                )
                
                if not importance_df.empty:
                    # Determine which column to use for the plot (Prediction_Driver for LR, Influence_Score for others)
                    influence_col = 'Prediction_Driver' if 'Prediction_Driver' in importance_df.columns else 'Influence_Score'
                    
                    if influence_col == 'Prediction_Driver':
                        importance_df = importance_df.sort_values(influence_col, key=lambda x: x.abs(), ascending=False)
                        plot_color_col = importance_df[influence_col] > 0
                        plot_title = "Prediction Driver Score (Top 10 - Prediction Specific)"
                        x_label = "Impact Score (Coefficient * Value)"
                        color_scale = 'RdBu_r'
                    else:
                        importance_df = importance_df.sort_values(influence_col, ascending=False)
                        plot_color_col = importance_df[influence_col]
                        plot_title = "Global Feature Importance (Top 10 - Model Wide)"
                        x_label = "Feature Importance (MDI)"
                        color_scale = 'Viridis'
                        
                    importance_df = importance_df.head(10)
                    
                    fig_imp = px.bar(
                        importance_df.sort_values(influence_col, ascending=True),
                        x=influence_col,
                        y='Feature',
                        orientation='h',
                        title=plot_title,
                        labels={influence_col: x_label, 'Feature': 'Feature'},
                        color=plot_color_col,
                        color_continuous_scale=color_scale
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("⚠️ Feature influence for this specific model configuration could not be calculated.")
            else:
                st.warning("⚠️ Feature importance is not directly available for the selected model (e.g., standard SVM).")

# ==================== PAGE 6: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Make New Churn Predictions")
    
    if st.session_state.production_model is None:
        st.warning("⚠️ Please train models and select a Production Model on the 'Train Models' page first!")
        st.stop()
        
    model = st.session_state.production_model
    model_name = st.session_state.production_model_name
    schema = st.session_state.feature_schema
    
    st.info(f"✅ Ready to predict using the **{model_name}** model.")
    
    prediction_type = st.radio(
        "Choose Prediction Input Type",
        options=["Single Customer Input", "Bulk Upload (Test Set Sample)"],
        index=0,
        horizontal=True
    )
    
    st.markdown("---")
    
    # ------------------------------------------------------------------
    # Single Customer Input
    if prediction_type == "Single Customer Input":
        st.subheader("📝 Input Customer Details")
        
        input_data = {}
        
        # Split features into numeric and categorical for better organization
        num_cols = schema['numeric_features']
        cat_cols = schema['categorical_features']
        
        # Use two columns for input
        col_num, col_cat = st.columns(2)
        
        with col_num:
            st.markdown("#### Numeric Features")
            for col in num_cols:
                # Get min/max/mean from clean data if possible for sensible defaults
                df_clean = st.session_state.df_clean
                
                try:
                    min_val = df_clean[col].min()
                    max_val = df_clean[col].max()
                    mean_val = df_clean[col].mean()
                except:
                    min_val, max_val, mean_val = 0.0, 100.0, 50.0
                
                input_data[col] = st.number_input(
                    f"**{col}**",
                    min_value=min_val if pd.notna(min_val) else 0.0,
                    max_value=max_val if pd.notna(max_val) else 10000.0,
                    value=mean_val if pd.notna(mean_val) else 50.0,
                    step=(max_val - min_val) / 20 if (max_val - min_val) > 0 else 1.0,
                    format="%.2f",
                    key=f"input_num_{col}"
                )
                
        with col_cat:
            st.markdown("#### Categorical Features")
            for col in cat_cols:
                # Get unique values from clean data
                df_clean = st.session_state.df_clean
                
                try:
                    options = df_clean[col].dropna().unique().tolist()
                    if 'Unknown' not in options: options.append('Unknown') # Add "Unknown" for imputation
                    default_index = options.index(df_clean[col].mode()[0]) if len(df_clean[col].mode()) > 0 and df_clean[col].mode()[0] in options else 0
                except:
                    options = ["A", "B", "C", "Unknown"]
                    default_index = 0
                
                input_data[col] = st.selectbox(
                    f"**{col}**",
                    options=options,
                    index=default_index,
                    key=f"input_cat_{col}"
                )
                
        if st.button("Calculate Churn Probability", key="predict_single_button", use_container_width=True):
            try:
                # Create DataFrame from input
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                pred_proba = model.predict_proba(input_df)[0][1]
                prediction = model.predict(input_df)[0]
                
                st.subheader("Prediction Result")
                col_pred, col_proba = st.columns(2)
                
                with col_pred:
                    churn_status = "CHURN" if prediction == 1 else "NO CHURN"
                    icon = "🔥" if prediction == 1 else "✅"
                    st.markdown(f'<div class="metric-card"><h2>{icon} {churn_status}</h2></div>', unsafe_allow_html=True)
                
                with col_proba:
                    st.metric("Churn Probability", f"{pred_proba * 100:.2f}%")
                
                st.markdown("---")
                
                # Model Interpretation (Feature Influence)
                st.subheader("🧠 Model Interpretation (Why this prediction?)")
                
                # The get_feature_influence function is designed to handle this
                influence_df = get_feature_influence(model, input_df, schema)
                
                if not influence_df.empty:
                    influence_col = 'Prediction_Driver' if 'Prediction_Driver' in influence_df.columns else 'Influence_Score'
                    
                    st.write("Top 5 features influencing this specific prediction:")
                    
                    fig_influence = px.bar(
                        influence_df.sort_values(influence_col, ascending=True),
                        x=influence_col,
                        y='Feature',
                        orientation='h',
                        title="Top Feature Drivers",
                        labels={influence_col: "Impact Score", 'Feature': 'Feature'},
                        color=influence_col,
                        color_continuous_scale='RdBu_r' # Red/Blue diverging scale for impact
                    )
                    st.plotly_chart(fig_influence, use_container_width=True)
                else:
                    st.warning("Cannot provide detailed feature influence for this model type.")
                    
            except Exception as e:
                st.error("An error occurred during prediction.")
                st.exception(e)

    # ------------------------------------------------------------------
    # Bulk Upload (Test Set Sample)
    elif prediction_type == "Bulk Upload (Test Set Sample)":
        st.subheader("📤 Bulk Prediction (Test Set Sample)")
        
        if st.session_state.X_test is None:
            st.warning("⚠️ Please train models first to generate the test set sample.")
            st.stop()
            
        X_test = st.session_state.X_test
        y_test = st.session_state.models[model_name]['y_test']
        
        st.info(f"Using the **{len(X_test)}** row test set sample for bulk prediction.")
        
        if st.button("Run Bulk Prediction on Test Set", key="predict_bulk_button", use_container_width=True):
            try:
                with st.spinner("Calculating bulk predictions..."):
                    # Predict probability and class
                    y_proba_bulk = model.predict_proba(X_test)[:, 1]
                    y_pred_bulk = model.predict(X_test)
                    
                    # Create results DataFrame
                    results_df_bulk = X_test.copy()
                    results_df_bulk['Actual Churn'] = y_test
                    results_df_bulk['Predicted Churn'] = y_pred_bulk
                    results_df_bulk['Churn Probability'] = y_proba_bulk
                    
                    st.success("✅ Bulk prediction complete!")
                    
                    # Display results
                    st.subheader("Bulk Prediction Results")
                    st.dataframe(results_df_bulk.head(10), use_container_width=True)
                    
                    # Download button
                    csv = results_df_bulk.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Full Prediction CSV",
                        data=csv,
                        file_name='churn_predictions_bulk.csv',
                        mime='text/csv',
                        key='download_bulk_csv'
                    )
                    
                    st.markdown("---")
                    
                    # Display Accuracy on Test Set
                    st.subheader("Performance on Test Set")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy Score", f"{accuracy_score(y_test, y_pred_bulk):.4f}")
                    with col2:
                        st.metric("ROC AUC Score", f"{roc_auc_score(y_test, y_proba_bulk):.4f}")
                        
            except Exception as e:
                st.error("An error occurred during bulk prediction.")
                st.exception(e)
