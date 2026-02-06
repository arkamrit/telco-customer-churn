import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Telco Churn Prediction Dashboard",
    layout="wide"
)

st.title("📊 Telco Churn Prediction Dashboard")
st.write("Upload CSV file, adjust threshold, and analyze churn risk")

# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/logistic_churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

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
    X = df.drop("Churn", axis=1, errors="ignore")
    X = pd.get_dummies(X)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]
    X_scaled = scaler.transform(X)

    # ---------------- PREDICTION ----------------
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["Churn_Predicted"] = preds
    df["Churn_Probability"] = probs

    # ---------------- KPIs ----------------
    total_customers = len(df)
    churn_count = df["Churn_Predicted"].sum()
    churn_percent = (churn_count / total_customers) * 100

    high_risk_count = (df["Churn_Probability"] > 0.7).sum()
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
    high_risk_df = df[df["Churn_Probability"] > 0.7]
    st.dataframe(high_risk_df)

    # ---------------- DOWNLOAD ----------------
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv,
        file_name="churn_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Please upload a CSV file to start")
