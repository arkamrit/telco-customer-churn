📊 Telco Customer Churn Prediction

Machine Learning | Python | scikit-learn | Streamlit

Predicts high-risk telecom customers using a production-ready ML pipeline and an interactive Streamlit app.

🔎 At a Glance 

Problem: Identify customers likely to churn

Solution: Supervised ML model with tuned decision threshold

Impact: High recall for churn class → fewer missed at-risk customers

Delivery: Train → Predict → Web App → Download results

Tech: Python, pandas, scikit-learn, Streamlit

🎯 Business Objective

Predict customer churn (Yes / No)

Optimize for recall on churn customers

Retention teams prefer false positives over missed churners

Enable non-technical users via a web UI

🧠 ML Highlights

Model: Logistic Regression (class-weighted)

Preprocessing:

One-hot encoding (categorical variables)

Feature scaling (StandardScaler)

Decision Threshold: 0.4 (tuned for recall)

Evaluation:

Confusion matrix

Precision, Recall, F1-score

Inference-safe pipeline (feature alignment guaranteed)

🖥️ Demo (Streamlit App)

Upload customer CSV

Adjust churn probability threshold

View churn probability + prediction

Download results as CSV

streamlit run src/app.py

📁 Project Structure
telco-customer-churn/
├── src/
│   ├── train_model.py      # Model training
│   ├── predict.py          # Batch prediction
│   ├── preprocessing.py   # Feature engineering
│   └── app.py              # Streamlit UI
│
├── models/                 # Saved model & artifacts
├── data/                   # Raw data & predictions
└── notebooks/              # EDA & experiments

🚀 Quick Start
pip install -r requirements.txt
streamlit run src/app.py


Batch prediction:

python src/predict.py --input data/raw/telco_churn.csv

🛠 Tech Stack

Python 3.10

pandas, numpy

scikit-learn

joblib

Streamlit

VS Code

💡 What This Project Demonstrates

✔ End-to-end ML workflow
✔ Feature-safe inference
✔ Business-driven metric optimization
✔ Model deployment with Streamlit
✔ Clean, production-style project structure

🔮 Next Improvements

CI pipeline (GitHub Actions)

Model monitoring

FastAPI backend

Cloud deployment (Streamlit Cloud / AWS)