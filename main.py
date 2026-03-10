from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pickle
import pandas as pd

model = joblib.load("xgb_churn_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = pickle.load(open("model_columns.pkl", "rb"))

app = FastAPI()

class Customer(BaseModel):
    seniorcitizen: int
    tenure: float
    monthlycharges: float
    totalcharges: float
    gender_Male: int
    partner_Yes: int
    dependents_Yes: int
    multiplelines_No_phone_service: int
    multiplelines_Yes: int
    contract_One_year: int
    contract_Two_year: int
    paperlessbilling_Yes: int
    paymentmethod_Credit_card_automatic: int
    paymentmethod_Electronic_check: int
    paymentmethod_Mailed_check: int

@app.post("/predict")
def predict(customer: Customer):
    
    # بناء الـ data بنفس أسماء الـ columns الحقيقية
    raw = {
        "seniorcitizen": customer.seniorcitizen,
        "tenure": customer.tenure,
        "monthlycharges": customer.monthlycharges,
        "totalcharges": customer.totalcharges,
        "gender_Male": customer.gender_Male,
        "partner_Yes": customer.partner_Yes,
        "dependents_Yes": customer.dependents_Yes,
        "multiplelines_No phone service": customer.multiplelines_No_phone_service,
        "multiplelines_Yes": customer.multiplelines_Yes,
        "contract_One year": customer.contract_One_year,
        "contract_Two year": customer.contract_Two_year,
        "paperlessbilling_Yes": customer.paperlessbilling_Yes,
        "paymentmethod_Credit card (automatic)": customer.paymentmethod_Credit_card_automatic,
        "paymentmethod_Electronic check": customer.paymentmethod_Electronic_check,
        "paymentmethod_Mailed check": customer.paymentmethod_Mailed_check,
    }

    data = pd.DataFrame([raw])
    data = data[columns]

    scale_cols = ['tenure', 'monthlycharges', 'totalcharges']
    data[scale_cols] = scaler.transform(data[scale_cols])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        return {"result": "⚠️ العميل هيمشي", "probability": f"{probability:.0%}"}
    else:
        return {"result": "✅ العميل مش هيمشي", "probability": f"{probability:.0%}"}