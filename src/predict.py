import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PIPELINE_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.pkl")

pipeline = joblib.load(PIPELINE_PATH)

def predict_churn(csv_path, threshold=0.4):
    df = pd.read_csv(csv_path)
    
    # Clean data: fix TotalCharges (handle blank spaces and nulls)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # Drop customerID if present
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    probs = pipeline.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    df["Churn_Probability"] = probs
    df["Churn_Predicted"] = preds

    return df

if __name__ == "__main__":
    input_csv = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
    output_csv = os.path.join(BASE_DIR, "data", "predictions.csv")

    result_df = predict_churn(input_csv)
    result_df.to_csv(output_csv, index=False)

    print("✅ Predictions saved")
