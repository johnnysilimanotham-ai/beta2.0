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
# --- Cleaning Configuration ---
if 'imputation_strategies' not in st.session_state:
    st.session_state.imputation_strategies = {}
if 'columns_to_drop' not in st.session_state:
    st.session_state.columns_to_drop = []

# ==================== HELPER FUNCTIONS (UPDATED) ====================

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
    unique_classes = df_clean[target_col].unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target column must have at least 2 classes. Found only: {unique_classes}")
    
    return df_clean

# --- NEW HELPER FUNCTION FOR FEATURE INFLUENCE ---
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
        importances = model.coef_[0]
        influence_method = "Logistic Regression Coefficients"
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        influence_method = "Built-in Feature Importances (MDI)"

    if importances is None:
        st.warning("⚠️ Feature influence for this model type cannot be calculated.")
        return pd.DataFrame()
        
    st.info(f"💡 **Feature Influence Method:** {influence_method} used for interpretation.")
        
    # 3. Create a DataFrame for importance
    importance_df = pd.DataFrame({
        'Feature': feature_names, 
        'Influence_Score': importances
    }).sort_values(by='Influence_Score', key=lambda x: x.abs(), ascending=False)
    
    # 4. Calculate prediction-specific driver score for Logistic Regression (Coefficient * Feature_Value)
    if isinstance(model, LogisticRegression):
        # Preprocess the single row to get the numerical feature vector
        X_processed = preprocessor.transform(X_single_row)
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        
        # Calculate the driver score (Coefficient * Preprocessed Value)
        importance_df['Prediction_Driver'] = importance_df['Influence_Score'] * X_processed_df.iloc[0].loc[importance_df['Feature']].values
        
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

# ==================== MAIN APP (REDUCED FOR BREVITY - FOCUS ON PREDICTIONS) ====================

st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction Platform</h1>', unsafe_allow_html=True)

# Sidebar Navigation
page = st.sidebar.selectbox(
    "🔍 Navigation",
    ["📤 Upload Data", "📊 Data Visualization", "🧹 Data Cleaning", "🤖 Train Models", "📈 Model Comparison", "🔮 Make Predictions"]
)

# ... (PAGE 1: UPLOAD DATA - NO CHANGE)
# ... (PAGE 2: DATA VISUALIZATION - NO CHANGE)
# ... (PAGE 3: DATA CLEANING - NO CHANGE)
# ... (PAGE 4: TRAIN MODELS - NO CHANGE)
# ... (PAGE 5: MODEL COMPARISON - NO CHANGE)

# ==================== PAGE 6: MAKE PREDICTIONS (MODIFIED) ====================
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
            
            # --- User defines the actionable threshold ---
            st.subheader("⚙️ Business Threshold Setting")
            churn_threshold = st.slider(
                "Actionable Churn Probability Threshold (Probability % above which you intervene)",
                min_value=5, max_value=95, value=50, step=5
            ) / 100.0
            st.info(f"Customers with a probability > **{churn_threshold:.1%}** will be flagged as **High Risk/Churn**. ")

            st.markdown("---")
            
            # Feature Inputs
            st.subheader("Customer Inputs")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numeric Features**")
                for col in num_cols:
                    input_data[col] = st.number_input(f"{col}", value=0.0, help=f"Input type: Numeric")
            
            with col2:
                st.markdown("**Categorical Features**")
                for col in cat_cols:
                    if st.session_state.df_clean is not None and col in st.session_state.df_clean.columns:
                        options = st.session_state.df_clean[col].dropna().astype(str).unique().tolist()
                        if not options:
                            options = ["N/A"]
                        
                        default_index = 0
                        
                        input_data[col] = st.selectbox(f"{col}", options, index=default_index, help=f"Input type: Categorical. Options derived from training data.")
                    else:
                        input_data[col] = st.text_input(f"{col}", help=f"Input type: Categorical. Cannot determine options.")
            
            if st.button("🎯 Predict Churn", type="primary"):
                # Create DataFrame with correct column order
                final_input_data = {k: [v] for k, v in input_data.items()}
                X = pd.DataFrame(final_input_data, columns=num_cols + cat_cols)
                
                prediction = pipe.predict(X)[0]
                proba = pipe.predict_proba(X)[0, 1]
                
                # Dynamic prediction based on user threshold
                dynamic_prediction = 1 if proba >= churn_threshold else 0

                # --- NEW: Feature Influence Calculation ---
                try:
                    influence_df = get_feature_influence(pipe, X, schema)
                except Exception as e:
                    st.error(f"Could not calculate feature influence: {e}")
                    influence_df = pd.DataFrame()
                
                
                st.markdown("---")
                st.subheader("🔮 Prediction Result")
                
                # Display dynamic prediction
                if dynamic_prediction == 1:
                    st.error("⚠️ **High Risk of CHURN** (Intervention Recommended)")
                    st.markdown(f"### Churn Probability: {proba*100:.1f}% (Above {churn_threshold:.1%} Threshold)")
                else:
                    st.success("✅ **Low Risk** (Retention Likely)")
                    st.markdown(f"### Retention Probability: {(1-proba)*100:.1f}% (Below {churn_threshold:.1%} Threshold)")
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    title={'text': "Churn Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred" if proba > churn_threshold else "darkgreen"},
                        'steps': [
                            {'range': [0, churn_threshold * 100], 'color': "lightgreen"},
                            {'range': [churn_threshold * 100, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': churn_threshold * 100
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, width='stretch')
                
                # --- NEW: Why this Prediction? (Applicability Improvement) ---
                if not influence_df.empty:
                    st.subheader("💡 Key Drivers for This Prediction (Interpretability)")
                    
                    # Customize the description based on the deployed model type
                    if 'Prediction_Driver' in influence_df.columns:
                        st.markdown(f"The **{pipe.named_steps['model'].__class__.__name__}** model uses the following factors to calculate this risk:")
                        
                        # Split drivers into CHURN (positive) and RETENTION (negative)
                        churn_drivers = influence_df[influence_df['Prediction_Driver'] > 0]
                        retention_drivers = influence_df[influence_df['Prediction_Driver'] < 0]

                        col1_inf, col2_inf = st.columns(2)
                        
                        with col1_inf:
                            st.markdown("**Top Drivers for CHURN/Risk (Positive Contribution):**")
                            if not churn_drivers.empty:
                                for _, row in churn_drivers.head(3).iterrows():
                                    # Clean up feature name (e.g., from 'col__value' to 'Col = Value')
                                    feature_display = row['Feature'].split('__')
                                    if len(feature_display) > 1:
                                        display_text = f"**{feature_display[0]}** = **{feature_display[1]}**"
                                    else:
                                        display_text = f"**{feature_display[0]}**"
                                        
                                    st.write(f"1. ⬆️ **{display_text}** (Contribution: +{row['Prediction_Driver']:.3f})")
                            else:
                                st.write("*(No significant positive drivers found)*")

                        with col2_inf:
                            st.markdown("**Top Drivers for RETENTION (Negative Contribution):**")
                            if not retention_drivers.empty:
                                for _, row in retention_drivers.sort_values(by='Prediction_Driver', ascending=True).head(3).iterrows():
                                    feature_display = row['Feature'].split('__')
                                    if len(feature_display) > 1:
                                        display_text = f"**{feature_display[0]}** = **{feature_display[1]}**"
                                    else:
                                        display_text = f"**{feature_display[0]}**"

                                    st.write(f"1. ⬇️ **{display_text}** (Contribution: {row['Prediction_Driver']:.3f})")
                            else:
                                st.write("*(No significant negative drivers found)*")
                    
                    else:
                        st.markdown("Top 5 *Globally* Important Features (Feature influence per prediction is not available for this model type):")
                        for _, row in influence_df.iterrows():
                            st.write(f"- **{row['Feature']}** (Importance: {row['Influence_Score']:.3f})")

        
        else:  # Batch Prediction
            st.subheader("📁 Upload CSV File for Batch Predictions")
            st.info("💡 Upload a CSV file with customer data. The file should NOT include the target column.")
            
            # --- User defines the actionable threshold for batch prediction ---
            st.subheader("⚙️ Business Threshold Setting")
            batch_churn_threshold = st.slider(
                "Actionable Churn Probability Threshold",
                min_value=5, max_value=95, value=50, step=5, key='batch_thresh'
            ) / 100.0
            
            batch_file = st.file_uploader(
                "Upload CSV for batch predictions",
                type=['csv'],
                key="batch_upload"
            )
            
            if batch_file is not None:
                batch_df = pd.read_csv(batch_file)
                
                st.success(f"✅ File loaded: {len(batch_df)} customers")
                st.dataframe(batch_df.head(), width='stretch')
                
                if st.button("🎯 Predict All", type="primary"):
                    try:
                        with st.spinner("Making predictions..."):
                            # Ensure columns match expected features
                            num_cols = schema['numeric_features']
                            cat_cols = schema['categorical_features']
                            expected_cols = num_cols + cat_cols
                            
                            # Check for missing columns
                            missing_cols = [col for col in expected_cols if col not in batch_df.columns]
                            if missing_cols:
                                raise ValueError(f"Batch data is missing required features: {', '.join(missing_cols)}")
                            
                            # Reorder columns to match training
                            X_batch = batch_df[expected_cols]
                            
                            # Make predictions
                            predictions = pipe.predict(X_batch) # Standard prediction (uses model's default 0.5 threshold)
                            probabilities = pipe.predict_proba(X_batch)[:, 1]
                            
                            # Dynamic prediction based on user threshold
                            dynamic_predictions = (probabilities >= batch_churn_threshold).astype(int)
                            
                            # Add predictions to dataframe
                            result_df = batch_df.copy()
                            result_df['Churn_Prediction'] = dynamic_predictions
                            result_df['Churn_Probability'] = probabilities
                            
                            # Dynamic Risk Level based on user threshold
                            def get_risk_level(prob):
                                if prob >= batch_churn_threshold:
                                    return 'High Risk'
                                elif prob >= (batch_churn_threshold * 0.75): # Slightly below threshold is Medium
                                    return 'Medium Risk'
                                else:
                                    return 'Low Risk'

                            result_df['Risk_Level'] = result_df['Churn_Probability'].apply(get_risk_level)
                        
                        st.success("✅ Predictions completed!")
                        
                        # Summary metrics based on user-defined threshold
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", len(result_df))
                        with col2:
                            st.metric(f"Customers to Intervene (Risk > {batch_churn_threshold:.1%})", (dynamic_predictions == 1).sum())
                        with col3:
                            intervention_rate = (dynamic_predictions == 1).mean() * 100
                            st.metric("Intervention Rate", f"{intervention_rate:.1f}%")
                        
                        # Risk distribution
                        st.subheader("📊 Risk Distribution (Based on User Threshold)")
                        risk_counts = result_df['Risk_Level'].value_counts()
                        fig = px.pie(
                            values=risk_counts.values,
                            names=risk_counts.index,
                            title="Customer Risk Distribution",
                            color_discrete_sequence=['#764ba2', '#FFA500', '#667eea']
                        )
                        st.plotly_chart(fig, width='stretch')
                        
                        # Results table
                        st.subheader("📋 Prediction Results")
                        st.dataframe(
                            result_df.sort_values('Churn_Probability', ascending=False),
                            width='stretch'
                        )
                        
                        # Download results
                        csv = result_df.to_csv(index=False).encode('utf-8')
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
# ... (NO CHANGE)
