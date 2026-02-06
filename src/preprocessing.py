import pandas as pd
import numpy as np

def preprocess_data(df):
    df = df.copy()

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop ID column
    df = df.drop(columns=["customerID"])

    # Binary encoding
    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df
