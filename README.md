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

� Model Performance

### Logistic Regression (Deployed)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.74 |
| **Precision** | 0.51 |
| **Recall** | 0.83 |
| **F1-Score** | 0.63 |
| **AUC-ROC** | 0.86 |

**Why Recall?** Missing a churner (false negative) = lost customer revenue. False positives = extra retention effort (acceptable cost).

### Random Forest (Comparison)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.79 |
| **Precision** | 0.60 |
| **Recall** | 0.65 |
| **F1-Score** | 0.62 |
| **AUC-ROC** | 0.84 |

**Decision:** LR chosen for interpretability + production stability.

🔧 API Documentation

### `train_model.py`
Trains Logistic Regression with class weighting and preprocessing pipeline.

```bash
python src/train_model.py
```

**Output:**
- `models/churn_pipeline.pkl` – Preprocessor + model

### `predict.py`
Batch prediction on new customer data.

```bash
python src/predict.py --input data/raw/telco_churn.csv --output data/predictions/results.csv
```

**Input:** CSV with same schema as training data
**Output:** CSV with `Churn_Probability` and `Churn_Predicted` columns

### `app.py` (Streamlit)
Interactive web UI for single / batch predictions.

```bash
streamlit run src/app.py
```

**Features:**
- Threshold slider (0.4 - 0.6)
- CSV upload & preview
- KPI cards (total, churn %, high-risk)
- Top drivers visualization
- Results download

🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: models/churn_pipeline.pkl` | Run `python src/train_model.py` first |
| `ValueError: could not convert string to float` | TotalCharges has blanks; app auto-fills with median |
| `StreamlitAPIException` on app start | Clear cache: `streamlit cache clear` |
| Model predictions seem off | Check threshold slider (default=0.4); lower = more sensitive |
| CSV upload fails | Ensure columns match training schema (check notebooks) |

🔮 Next Improvements

✨ **Short Term**
- Add unit tests (`pytest`)
- Implement input validation & error handling
- Add model versioning & metadata tracking

🚀 **Medium Term**
- CI/CD pipeline (GitHub Actions)
- Model monitoring & drift detection
- Performance metrics dashboard

🌐 **Long Term**
- FastAPI backend for production
- Cloud deployment (Streamlit Cloud / AWS / GCP)
- A/B testing framework

