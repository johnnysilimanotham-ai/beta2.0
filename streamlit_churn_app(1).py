"""
Customer Churn Prediction App
Save this as: streamlit_churn_app.py
Run with: streamlit run streamlit_churn_app.py
"""
import io
import json
import joblib
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
# === Compatibility fix for Streamlit >= 1.32 ===
def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "safe_rerun"):
        st.safe_rerun()

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
        font-size: 2.4rem;
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
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Ensure models dir exists
Path("models").mkdir(exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Normalize target to 0/1 format"""
    if target_col not in df.columns:
        return df

    s = df[target_col].copy()

    # If it's already numeric, coerce to 0/1 where possible
    if pd.api.types.is_numeric_dtype(s):
        df[target_col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
        # If more than 2 unique numeric values exist, attempt to binarize common patterns
        uniq = sorted(df[target_col].unique())
        if len(uniq) > 2:
            # If values look like 0/1 floats, clip
            df[target_col] = df[target_col].clip(0, 1).astype(int)
    # If it's object/string type
    elif s.dtype == "O" or pd.api.types.is_string_dtype(s):
        def to_bin(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return 0
            xs = str(x).strip().lower()
            if xs in {"yes", "true", "1", "1.0", "y", "t", "churn", "positive"}:
                return 1
            if xs in {"no", "false", "0", "0.0", "n", "f", "retain", "negative"}:
                return 0
            # fallback: if the string contains digits and it's "0"/"1"
            if xs.isdigit():
                return int(xs) if xs in {"0","1"} else 0
            return 0
        df[target_col] = s.map(to_bin).astype(int)
    else:
        try:
            df[target_col] = s.astype(int)
        except Exception:
            df[target_col] = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)

    return df

def clean_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Clean and prepare data"""
    df = df.copy()

    # Drop ID-like columns first
    id_cols = [c for c in df.columns if 'id' in c.lower() and df[c].nunique() > 0]
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

def build_onehot_encoder():
    # Compatibility wrapper for scikit-learn versions using sparse vs sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

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
        ("encoder", build_onehot_encoder())
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

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
        "SVM": SVC(kernel="rbf", probability=True, random_state=42, max_iter=2000)
    }

    results = []
    trained_models = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(models_def)
    for idx, (name, model) in enumerate(models_def.items()):
        status_text.text(f"Training {name}...")
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        # Predictions
        y_pred = pipe.predict(X_test)
        y_proba = None
        try:
            proba = pipe.predict_proba(X_test)
            y_proba = proba[:, 1]
        except Exception:
            # If model doesn't support predict_proba, try decision_function -> scale to (0,1)
            try:
                df_dec = pipe.decision_function(X_test)
                # min-max scale
                mins = df_dec.min()
                maxs = df_dec.max()
                if maxs - mins == 0:
                    y_proba = np.zeros_like(df_dec)
                else:
                    y_proba = (df_dec - mins) / (maxs - mins)
            except Exception:
                y_proba = None

        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
        }
        results.append(metrics)

        trained_models[name] = {
            "pipeline": pipe,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        progress_percent = int(((idx + 1) / total) * 100)
        progress_bar.progress(progress_percent)

    status_text.empty()
    progress_bar.empty()

    results_df = pd.DataFrame(results).set_index("model").sort_values("f1", ascending=False)

    return results_df, trained_models, schema, X_test

# ==================== MAIN APP ====================
st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox(
    "📍 Navigation",
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

# ==================== PAGE 2: DATA VISUALIZATION ====================
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
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            st.metric("Categorical Features", len(categorical_cols))
        with col4:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100) if len(df)>0 and len(df.columns)>0 else 0.0
            st.metric("Missing Data %", f"{missing_pct:.1f}%")

        # Missing data analysis
        st.subheader("⚠️ Missing Data Analysis")
        missing_data = pd.DataFrame({
            'Column': df.columns,
            'Missing': df.isnull().sum().values,
            'Percent': (df.isnull().sum().values / len(df) * 100).round(2) if len(df)>0 else 0
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
                        title=f"Top 15 values in {selected_cat}"
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.metric("Unique Values", unique_count)
                    if len(value_counts) > 0:
                        st.metric("Most Common", str(value_counts.index[0]))
                        st.metric("Frequency", int(value_counts.values[0]))
                    if unique_count > 50:
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
                cols = corr_matrix.columns
                for i in range(len(cols)):
                    for j in range(i+1, len(cols)):
                        val = corr_matrix.iloc[i, j]
                        if abs(val) > 0.7:
                            strong_corr.append({
                                'Feature 1': cols[i],
                                'Feature 2': cols[j],
                                'Correlation': val
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
            if numeric_cols and target_col in df.columns:
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
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.write("**Top Correlated Features:**")
                        for feat, corr in target_corr.head(5).items():
                            st.metric(feat, f"{corr:.3f}")

            # Box plots for numeric features
            if numeric_cols:
                selectable = [col for col in numeric_cols if col != st.session_state.target_col]
                if selectable:
                    selected_feature = st.selectbox(
                        "Select feature to compare with target",
                        selectable
                    )

                    if selected_feature:
                        fig = px.box(
                            df,
                            x=st.session_state.target_col,
                            y=selected_feature,
                            title=f"{selected_feature} by {st.session_state.target_col}",
                            color=st.session_state.target_col
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
            recommendations.append("✅ **Mixed/Categorical data** - Random Forest and tree-based models recommended")

        if not recommendations:
            recommendations.append("✅ Data looks good! Ready for model training.")

        for rec in recommendations:
            st.write(rec)

        st.info("💡 **Next Steps:** Review these insights, go back to Data Cleaning to make adjustments if needed, or proceed to Train Models when ready!")

# ==================== PAGE 3: DATA CLEANING ====================
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

        # Target column selection default heuristics
        cols_list = df.columns.tolist()
        default_index = 0
        lowered = [c.lower() for c in cols_list]
        if 'churn' in lowered:
            default_index = lowered.index('churn')
        elif st.session_state.target_col and st.session_state.target_col in cols_list:
            default_index = cols_list.index(st.session_state.target_col)
        else:
            default_index = min(len(cols_list) - 1, 0)

        target_col = st.selectbox(
            "Select Target Column (Churn)",
            options=cols_list,
            index=default_index
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
            'Missing %': (df.isnull().sum().values / len(df) * 100).round(2) if len(df)>0 else 0
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
                    st.experimental_rerun()
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
                            most_frequent = df[selected_col].mode().iloc[0]
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
                        st.safe_rerun()

        with st.expander("🎯 Clean Target Column", expanded=False):
            st.write(f"Current target: **{target_col}**")
            st.write(f"Unique values: {df[target_col].unique()}")
            st.write("Value counts:")
            st.dataframe(df[target_col].value_counts(), use_container_width=True)

            if st.button("Normalize Target to 0/1"):
                try:
                    df = normalize_target(df, target_col)
                    df = df.dropna(subset=[target_col])
                    st.session_state.df_clean = df
                    st.success("✅ Target normalized!")
                    st.safe_rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        with st.expander("🗑️ Remove Unwanted Columns", expanded=False):
            st.write("Remove columns that won't help prediction like IDs, names, dates, or high-cardinality features")

            potential_remove = []
            # Heuristics for potential columns to remove
            for col in df.columns:
                # very high cardinality possibly identifiers
                if df[col].nunique() > 0.9 * len(df):
                    potential_remove.append(col)
                # names / email-like heuristics
                if any(sub in col.lower() for sub in ['name', 'email', 'phone', 'address']):
                    potential_remove.append(col)

            potential_remove = sorted(set(potential_remove))
            selected_to_drop = st.multiselect("Select columns to remove", options=df.columns.tolist(), default=potential_remove[:3])
            if st.button("Drop Selected Columns"):
                if selected_to_drop:
                    df = df.drop(columns=selected_to_drop)
                    st.session_state.df_clean = df
                    st.success(f"✅ Dropped columns: {', '.join(selected_to_drop)}")
                    st.safe_rerun()
                else:
                    st.info("No columns selected")

        # Save cleaned dataframe to session (explicit Save button)
        if st.button("Save cleaned dataset"):
            st.session_state.df_clean = df
            st.success("✅ Cleaned dataset saved to session state")

# ==================== PAGE 4: TRAIN MODELS ====================
elif page == "🤖 Train Models":
    st.header("🤖 Train Models")

    if st.session_state.df_clean is None and st.session_state.df_raw is None:
        st.warning("⚠️ Please upload and clean a dataset first!")
    else:
        df = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw

        if st.session_state.target_col is None or st.session_state.target_col not in df.columns:
            st.warning("⚠️ Please set the target column in Data Cleaning first.")
        else:
            test_size = st.slider("Test set size (fraction)", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
            if st.button("Train models"):
                try:
                    with st.spinner("Training models — this may take a minute..."):
                        results_df, trained_models, schema, X_test = train_models(df, st.session_state.target_col, test_size=test_size)
                    st.session_state.results = results_df
                    st.session_state.trained_models = trained_models
                    st.session_state.feature_schema = schema

                    st.success("✅ Models trained!")
                    st.subheader("Model Results (sorted by F1)")
                    st.dataframe(results_df, use_container_width=True)

                    # Save each trained pipeline to disk and session for later use
                    for name, info in trained_models.items():
                        joblib_path = f"models/{name.replace(' ', '_')}.joblib"
                        joblib.dump(info['pipeline'], joblib_path)
                        st.info(f"Saved pipeline for {name} -> {joblib_path}")
                        st.session_state.models[name] = joblib_path

                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

# ==================== PAGE 5: MODEL COMPARISON ====================
elif page == "📈 Model Comparison":
    st.header("📈 Model Comparison & Evaluation")

    if st.session_state.results is None or st.session_state.trained_models is None:
        st.warning("⚠️ No models trained yet. Go to Train Models and run training.")
    else:
        results_df = st.session_state.results
        st.subheader("Summary Metrics")
        st.dataframe(results_df, use_container_width=True)

        model_to_show = st.selectbox("Select model to inspect", options=list(st.session_state.trained_models.keys()))
        info = st.session_state.trained_models[model_to_show]
        y_test = info['y_test']
        y_pred = info['y_pred']
        y_proba = info.get('y_proba', None)

        st.subheader(f"Confusion Matrix — {model_to_show}")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix: {model_to_show}")
        st.plotly_chart(fig, use_container_width=True)

        if y_proba is not None:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
            roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
            roc_fig.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', title=f"ROC Curve — {model_to_show}")
            st.plotly_chart(roc_fig, use_container_width=True)

        # Option to promote to production model
        if st.button(f"Promote {model_to_show} to production"):
            st.session_state.production_model = st.session_state.trained_models[model_to_show]['pipeline']
            joblib.dump(st.session_state.production_model, "models/production_model.joblib")
            st.success(f"✅ Promoted {model_to_show} to production and saved to models/production_model.joblib")

# ==================== PAGE 6: MAKE PREDICTIONS ====================
elif page == "🔮 Make Predictions":
    st.header("🔮 Make Predictions")

    # Decide which model to use
    model_choice = None
    if st.session_state.production_model is not None:
        st.write("Using **production model** from session.")
        model_choice = "production"
    elif st.session_state.models:
        model_choice = st.selectbox("Select trained model to use", options=list(st.session_state.models.keys()))
    else:
        st.warning("⚠️ No model found. Train a model first.")
        model_choice = None

    pipeline = None
    if model_choice == "production":
        pipeline = st.session_state.production_model
    elif model_choice is not None:
        path = st.session_state.models.get(model_choice)
        if path and os.path.exists(path):
            pipeline = joblib.load(path)
        else:
            st.warning("Selected model not found on disk; try retraining or choose a different model.")

    if pipeline is not None:
        st.subheader("Provide input for a single prediction")
        upload_single = st.file_uploader("Upload single-row CSV (first row as header)", type=['csv'], key="pred_upload")
        manual_df = None
        if upload_single is not None:
            try:
                manual_df = pd.read_csv(upload_single)
                if manual_df.shape[0] > 1:
                    st.warning("Only first row will be used for prediction")
                input_df = manual_df.iloc[[0]]
            except Exception as e:
                st.error(f"Could not read uploaded file: {e}")
                input_df = None
        else:
            # Build small form from schema if available
            schema = st.session_state.feature_schema
            if schema:
                numeric = schema.get('numeric_features', [])
                categorical = schema.get('categorical_features', [])
                input_data = {}
                st.write("Fill values (leave blank to use NaN):")
                with st.form("manual_input_form"):
                    for col in numeric:
                        val = st.text_input(f"{col} (numeric)", key=f"num_{col}")
                        input_data[col] = float(val) if val not in ("", None) else np.nan
                    for col in categorical:
                        val = st.text_input(f"{col} (categorical)", key=f"cat_{col}")
                        input_data[col] = val if val not in ("", None) and val != "" else np.nan
                    submitted = st.form_submit_button("Create input row")
                    if submitted:
                        input_df = pd.DataFrame([input_data])
                        st.write("Preview of input row:")
                        st.dataframe(input_df, use_container_width=True)
            else:
                st.info("No feature schema available. Upload a one-row CSV to predict.")

        if 'input_df' in locals() and input_df is not None:
            try:
                proba = pipeline.predict_proba(input_df)[:, 1] if hasattr(pipeline, "predict_proba") else None
                pred = pipeline.predict(input_df)

                st.subheader("Prediction Result")
                st.metric("Predicted Class", int(pred[0]))
                if proba is not None:
                    st.metric("Predicted Churn Probability", f"{proba[0]:.3f}")
                else:
                    st.info("Model does not provide probability (no predict_proba).")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.info("Tip: the feature names and types must match what was used for training.")

