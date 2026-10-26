# Telecom Customer Churn Prediction System (Flask + XGBoost + SHAP)

A production-ready machine learning web application that predicts telecom customer churn with explainability (SHAP) and personalized retention strategies.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-black)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-green)
![SHAP](https://img.shields.io/badge/SHAP-0.54.1-orange)

## Live Demo (Flask App)
**Render Deployment:** https://telecom-churn-flask.onrender.com

## Other Demos / Older Versions (Optional)
- Streamlit App (older): https://telecom-churn-prediction-xpa5emx28vo8c23kegohcs.streamlit.app/
- Frontend UI demo (older): https://telecom-churn-prediction-main.vercel.app/

---

## Project Overview

This system helps telecom companies:
- **Predict** which customers will churn
- **Explain** WHY they will churn (SHAP top drivers)
- **Prioritize** customers by risk level (GREEN / YELLOW / RED)
- **Suggest** personalized retention strategies (offers + urgency)

---

## Key Results (Model)
| Metric | Score |
|--------|-------|
| Accuracy | 94.01% |
| AUC-ROC | 93.81% |
| Precision | 73.54% |
| Recall | 64.38% |
| F1-Score | 68.66% |

> Metrics are from model evaluation during training/validation.

---

## Risk Segmentation

| Category | Customers | Actual Churn Rate |
|----------|-----------|-------------------|
| 🔴 RED (Urgent) | 7,396 | 93.9% |
| 🟡 YELLOW (Monitor) | 3,279 | 61.8% |
| 🟢 GREEN (Safe) | 89,324 | 1.4% |

---

## What’s Unique in This Web App

1) **SHAP Explainability**
- Shows the top features pushing churn **up** or **down**
- Helps business teams understand *why* a customer is risky

2) **Three-tier Risk System**
- 🔴 RED = urgent retention
- 🟡 YELLOW = monitor and intervene
- 🟢 GREEN = normal engagement

3) **Personalized Retention Strategy Engine**
- Generates urgency + reasons + recommended offers based on usage patterns

4) **Telecom-specific signals**
- ARPU trends (Month 1/2/3)
- Recharge behavior (amount/number trends)
- Calling usage (OG/IC MOU patterns)
- Roaming / 2G/3G indicators (advanced inputs)

---
## Tech Stack (This Flask Version)

| Technology | Purpose |
|----------|---------|
| Python 3.12 | Runtime |
| Flask | Backend Web Framework |
| Gunicorn | Production server |
| XGBoost | ML model |
| SHAP + Numba | Explainability |
| Scikit-learn | Scaling / pipeline pieces |
| Pandas / NumPy | Data processing |
| Bootstrap | UI components |
| Render | Deployment |

---

telecom_churn_flask/
│
├── app.py # Flask backend (predict + SHAP + recommendations)
├── requirements.txt # Dependencies
├── runtime.txt # Pins Python version for Render
│
├── models/
│ ├── final_xgboost_model.pkl
│ ├── scaler.pkl
│ ├── shap_explainer.pkl
│ ├── feature_names.json
│ └── meta.json
│
├── templates/
│ └── index.html # Glassmorphism UI (advanced mode)
│
└── static/
├── css/
│ └── style.css # Glass UI + gauge styles
└── js/
└── app.js # Form -> /predict + gauge + SHAP render + engineered preview

## Run Locally

```bash
python -m venv .venv
# activate venv (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
---

## Important Note: Feature Pipeline (Accuracy Fix)

The trained model expects a **153-feature** input vector.

To ensure stable, accurate predictions even when users fill only the core form inputs:
1. The backend initializes all missing model features using **training means** from the saved `StandardScaler` (not zeros)
2. Overwrites user-provided values
3. Computes engineered features (trends/ratios/std/segments) to match the training notebook

This prevents incorrect predictions caused by zero-filling missing fields.

---

## Project Structure (This Repo)
