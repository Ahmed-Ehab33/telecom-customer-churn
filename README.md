# telecom-customer-churn

# 📡 Telecom Customer Churn Prediction

A machine learning project to predict which telecom customers are likely to churn,
built as a university project and later improved after discovering critical data
leakage issues in the original pipeline.

---

## 🚨 What I Fixed

The original code had a serious inconsistency:
models were **trained on unscaled data** but **tested on scaled data**,
which made all results unreliable.

**The correct order is:** `Scaling → SMOTE → Training`

---

## 🔧 Project Pipeline

| Step | Details |
|------|---------|
| Data Cleaning | Fixed hidden blank values in `TotalCharges`, converted to numeric |
| Encoding | One-Hot Encoding for categorical features |
| Scaling | StandardScaler on `tenure`, `monthlycharges`, `totalcharges` |
| Imbalance | SMOTE applied on training set only |
| Modeling | Naive Bayes, Decision Tree, ANN, XGBoost |
| Tuning | GridSearchCV on XGBoost + custom threshold = 0.35 |
| Deployment | Streamlit app (local) |

---

## 📊 Results (XGBoost — Best Model)

| Metric | Score |
|--------|-------|
| Accuracy | 0.70 |
| Recall (Churn) | **0.85** |
| F1-Score | 0.60 |

> In churn problems, **Recall matters more than Accuracy.**
> Missing a customer who's about to leave costs far more than
> a false alarm call to someone who wasn't leaving.

Out of **374 customers who churned**, the model caught **319** ✅
Only **55 were missed.**

---

## ❌ ANN — Why It Failed

The ANN showed `Recall = 1.0` with `Precision = 0.27`,
meaning it predicted **everyone** as churning.
That's not a model — that's a failure.
This was a direct result of the scaling inconsistency.

---

## 🗂️ Files

| File | Description |
|------|-------------|
| `Telecom_Cust_Churn.ipynb` | Full analysis, EDA, model training & comparison |
| `Churn_appML.py` | Streamlit deployment app |

---

## 🛠️ Tech Stack

`Python` `Pandas` `Scikit-learn` `XGBoost` `TensorFlow` `SMOTE` `Streamlit`
