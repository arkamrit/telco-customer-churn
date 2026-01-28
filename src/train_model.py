import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from preprocessing import preprocess_data

# Load & preprocess
df = pd.read_csv("data/raw/telco_churn.csv")
df = preprocess_data(df)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== SCALING (IMPORTANT for Logistic Regression) =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with class weight
lr = LogisticRegression(
    max_iter=3000,
    class_weight={0: 1, 1: 2}
)
lr.fit(X_train_scaled, y_train)

lr_probs = lr.predict_proba(X_test_scaled)[:, 1]
lr_pred = (lr_probs >= 0.4).astype(int)

print("Logistic Regression (Threshold 0.4 + Class Weight) Results")
print(classification_report(y_test, lr_pred))
print("Accuracy:", accuracy_score(y_test, lr_pred))

# Random Forest (NO scaling needed)
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight={0: 1, 1: 2}
)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_probs >= 0.4).astype(int)

print("Random Forest (Threshold 0.4 + Class Weight) Results")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

# ===== SAVE MODELS =====
joblib.dump(lr, "models/logistic_churn_model.pkl")
joblib.dump(rf, "models/random_forest_churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
