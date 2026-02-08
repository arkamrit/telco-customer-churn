import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/raw/telco_churn.csv")

# Fix TotalCharges
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------- FEATURES ----------------
num_features = ["tenure", "MonthlyCharges", "TotalCharges"]

cat_features = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# ---------------- PREPROCESSOR ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ---------------- PIPELINE ----------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=3000,
            class_weight={0: 1, 1: 2}
        ))
    ]
)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- TRAIN ----------------
pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
probs = pipeline.predict_proba(X_test)[:, 1]
preds = (probs >= 0.4).astype(int)

print("Pipeline Logistic Regression Results")
print(classification_report(y_test, preds))

# ---------------- SAVE ----------------
joblib.dump(pipeline, "models/churn_pipeline.pkl")

print("✅ Pipeline saved to models/churn_pipeline.pkl")
