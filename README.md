# 📊 Telco Customer Churn Prediction

This project focuses on predicting **customer churn** for a telecom company using machine learning.  
The goal is to identify customers who are likely to leave the service, so that proactive retention strategies can be applied.

---

## 🚀 Project Objective

- Predict whether a customer will **churn (leave)** or **not churn**
- Focus on **high recall for churn customers (Class 1)**  
  > Missing a churn customer is more costly than flagging a non-churn customer

---

## 📁 Project Structure

telco-customer-churn/
│
├── data/
│   ├── raw/                    # Original, unmodified data
│   │   └── telco_churn.csv
│   └── processed/              # Cleaned / processed data (future use)
│
├── models/                     # Saved trained models
│   ├── logistic_churn_model.pkl
│   ├── random_forest_churn_model.pkl
│   └── scaler.pkl
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_eda.ipynb             # Exploratory Data Analysis
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_loading.py         # Data loading utilities
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   └── train_model.py          # Model training & evaluation
│
│
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── .gitignore                  # Files/folders to ignore in git


---

## 📊 Dataset Description

- **Dataset**: Telco Customer Churn (Kaggle)
- **Rows**: 7,043 customers
- **Target Variable**: `Churn`
  - `1` → Customer will churn
  - `0` → Customer will not churn

### Feature Types
- **Demographics**: gender, senior citizen, partner, dependents
- **Services**: phone service, internet service, streaming, tech support, etc.
- **Account info**: tenure, contract type, payment method
- **Charges**: monthly charges, total charges

---

## 🧠 Models Used

### 1️⃣ Logistic Regression
- Feature scaling using `StandardScaler`
- Class weight tuning: `{0: 1, 1: 2}`
- Custom threshold: **0.4** (instead of default 0.5)
- Optimized for **high recall on churn class**

### 2️⃣ Random Forest
- Ensemble-based model
- Class weight handling for imbalance
- Threshold set to **0.4**

---

## 📈 Model Evaluation Metrics

- **Recall (Class 1)** → Primary metric
- Precision
- Accuracy
- Confusion Matrix

### Why Recall?
In churn prediction:
- **False Negative** = customer leaves unnoticed (high business loss)
- **False Positive** = retention offer to loyal customer (acceptable cost)

---

## ⚙️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
