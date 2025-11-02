# ===============================================================
#  🧠 Streamlit Churn Prediction Platform (2025-ready)
#  Fully compatible with Streamlit v1.32+ (no deprecated APIs)
# ===============================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ===============================================================
# 🧩 Compatibility Helpers (for old/new Streamlit)
# ===============================================================

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

def safe_data_editor(data, **kwargs):
    if hasattr(st, "data_editor"):
        return st.data_editor(data, **kwargs)
    elif hasattr(st, "experimental_data_editor"):
        return st.experimental_data_editor(data, **kwargs)
    else:
        st.error("Your Streamlit version is too old for editable tables.")
        return data

def safe_columns(spec):
    if hasattr(st, "columns"):
        return st.columns(spec)
    elif hasattr(st, "beta_columns"):
        return st.beta_columns(spec)
    else:
        return [st]


# ===============================================================
# 📁 Page Setup
# ===============================================================
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("💡 Customer Churn Prediction App")

# ===============================================================
# 🔸 Sidebar Navigation
# ===============================================================
pages = ["📤 Upload Data", "🧹 Clean Data", "⚙️ Train Model", "📈 Compare Models", "🔮 Predict"]
choice = st.sidebar.radio("Navigation", pages)

# ===============================================================
# 📤 PAGE 1: Upload Data
# ===============================================================
if choice == "📤 Upload Data":
    st.header("Upload your dataset")
    file = st.file_uploader("Upload a CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.session_state.df = df
        st.write("✅ File uploaded successfully!")
        st.dataframe(df.head())
    else:
        st.info("Please upload a CSV file to begin.")


# ===============================================================
# 🧹 PAGE 2: Clean Data
# ===============================================================
elif choice == "🧹 Clean Data":
    if "df" not in st.session_state:
        st.warning("Please upload data first.")
        st.stop()

    df = st.session_state.df.copy()
    st.header("Data Cleaning Tools")

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    # Remove columns
    with st.expander("🧩 Remove Columns"):
        cols = st.multiselect("Select columns to remove", df.columns)
        if st.button("Remove Selected Columns"):
            before = df.shape[1]
            df = df.drop(columns=cols)
            st.session_state.df_clean = df
            st.success(f"Removed {before - df.shape[1]} columns.")
            safe_rerun()

    # Handle missing values
    with st.expander("🧼 Handle Missing Values"):
        method = st.selectbox("Choose a method", ["None", "Fill with Mean", "Fill with Median", "Drop Rows with NA"])
        if method == "Fill with Mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif method == "Fill with Median":
            df = df.fillna(df.median(numeric_only=True))
        elif method == "Drop Rows with NA":
            df = df.dropna()
        st.session_state.df_clean = df
        st.success("✅ Missing values handled.")

    # Choose target column
    with st.expander("🎯 Set Target Column"):
        target_col = st.selectbox("Select target column", df.columns)
        st.session_state.target_col = target_col
        st.info(f"Target column set to: **{target_col}**")

    st.subheader("Final Preview")
    st.dataframe(df.head())


# ===============================================================
# ⚙️ PAGE 3: Train Model
# ===============================================================
elif choice == "⚙️ Train Model":
    if "df_clean" not in st.session_state or "target_col" not in st.session_state:
        st.warning("Please clean data and select target first.")
        st.stop()

    df = st.session_state.df_clean
    target_col = st.session_state.target_col

    st.header("Model Training")

    # Encode categorical data
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = label_enc.fit_transform(df[col].astype(str))

    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = {"Accuracy": acc, "F1 Score": f1}
        st.write(f"### {name}")
        st.write(f"- Accuracy: {acc:.4f}")
        st.write(f"- F1 Score: {f1:.4f}")
        st.text(classification_report(y_test, preds))
        st.divider()

    st.session_state.model_results = results
    st.session_state.trained_models = models
    st.success("✅ Models trained successfully!")


# ===============================================================
# 📈 PAGE 4: Compare Models
# ===============================================================
elif choice == "📈 Compare Models":
    if "model_results" not in st.session_state:
        st.warning("Please train models first.")
        st.stop()

    st.header("Model Comparison Dashboard")
    results = pd.DataFrame(st.session_state.model_results).T
    st.bar_chart(results)
    st.dataframe(results.style.highlight_max(axis=0))
    st.success("✅ Comparison complete!")


# ===============================================================
# 🔮 PAGE 5: Predict
# ===============================================================
elif choice == "🔮 Predict":
    if "trained_models" not in st.session_state:
        st.warning("Please train a model first.")
        st.stop()

    models = st.session_state.trained_models
    df = st.session_state.df_clean
    target_col = st.session_state.target_col

    st.header("Make New Predictions")

    model_name = st.selectbox("Select model", list(models.keys()))
    model = models[model_name]

    input_data = {}
    for col in df.drop(columns=[target_col]).columns:
        val = st.text_input(f"Enter value for {col}")
        input_data[col] = val

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        for col in input_df.columns:
            if input_df[col].dtype == object:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
        prediction = model.predict(input_df)
        st.success(f"🎯 Predicted outcome: **{prediction[0]}**")

