import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    layout="wide"
)

st.title("📊 Telco Churn Prediction Dashboard")
st.write("Upload CSV file, adjust threshold, and analyze churn risk")

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

# ---------------- LOAD PIPELINE ----------------
@st.cache_resource
def load_pipeline():
    return joblib.load(PIPELINE_PATH)

pipeline = load_pipeline()

# ---------------- THRESHOLD SLIDER ----------------
threshold = st.slider(
    "🎚️ Select Churn Threshold",
    min_value=0.40,
    max_value=0.60,
    value=0.40,
    step=0.01
)

st.write(f"**Current Threshold:** `{threshold}`")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload Telco CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # ---------------- PREPROCESS ----------------
    # Clean data: fix TotalCharges (handle blank spaces and nulls)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # Drop customerID if present
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    # ---------------- PREDICTION ----------------
    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["Churn_Probability"] = probs
    df["Churn_Predicted"] = preds

    # ---------------- KPIs ----------------
    total_customers = len(df)
    churn_count = int(df["Churn_Predicted"].sum())
    churn_percent = (churn_count / total_customers) * 100

    high_risk_count = int((df["Churn_Probability"] > 0.7).sum())
    high_risk_percent = (high_risk_count / total_customers) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("👥 Total Customers", total_customers)
    col2.metric("❌ Predicted Churn", churn_count)
    col3.metric("📉 Churn %", f"{churn_percent:.2f}%")
    col4.metric(
        "🚨 High Risk Customers",
        high_risk_count,
        f"{high_risk_percent:.2f}%"
    )

    # ---------------- CHURN DISTRIBUTION ----------------
    st.subheader("📈 Churn Distribution")

    fig, ax = plt.subplots()
    df["Churn_Predicted"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xticklabels(["No Churn", "Churn"], rotation=0)
    ax.set_ylabel("Customer Count")
    st.pyplot(fig)

    # ---------------- HIGH RISK TABLE ----------------
    st.subheader("🚨 High-Risk Customers (Probability > 0.7)")
    st.dataframe(df[df["Churn_Probability"] > 0.7])

    # ---------------- DOWNLOAD ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

    st.subheader("🧠 Top Churn Drivers (Model Insights)")

    # Extract feature importance from pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2])
    
    all_features = list(num_features) + list(cat_features)
    coefficients = model.coef_[0]
    
    feature_importance = pd.DataFrame({
        "Feature": all_features,
        "Coefficient": coefficients,
        "Abs_Coefficient": np.abs(coefficients)
    }).sort_values("Abs_Coefficient", ascending=False)
    
    top_features = feature_importance.head(10)
    
    # Display as interactive bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in top_features['Coefficient']]
    ax2.barh(range(len(top_features)), top_features['Coefficient'], color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['Feature'])
    ax2.set_xlabel("Coefficient Value")
    ax2.set_title("Top 10 Features Driving Churn")
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Show data table
    st.dataframe(top_features[["Feature", "Coefficient"]].reset_index(drop=True))


else:
    st.info("👆 Please upload a CSV file to start")
