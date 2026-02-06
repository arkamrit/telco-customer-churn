import os
import pandas as pd
import joblib

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "logistic_churn_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

INPUT_CSV = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "predictions.csv")

# ---------------- LOAD ARTIFACTS ----------------
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_PATH)

# ---------------- PREPROCESS ----------------
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Drop target if exists
    X = df.drop("Churn", axis=1, errors="ignore")

    # One-hot encode (same as training)
    X = pd.get_dummies(X)

    # Add missing columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    # Remove extra columns + keep order
    X = X[feature_names]

    # Scale
    X_scaled = scaler.transform(X)

    return X_scaled, df

# ---------------- PREDICT ----------------
def predict_churn(csv_path, threshold=0.4):
    X_scaled, original_df = load_and_preprocess(csv_path)

    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    original_df["Churn_Predicted"] = predictions
    original_df["Churn_Probability"] = probabilities

    return original_df, predictions

# ---------------- MAIN ----------------
if __name__ == "__main__":
    result_df, _ = predict_churn(INPUT_CSV)

    result_df.to_csv(OUTPUT_CSV, index=False)

    print("✅ Prediction completed")
    print(result_df[["Churn_Predicted"]].head())
    print("📁 Saved to:", OUTPUT_CSV)
