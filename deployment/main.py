from datetime import date
from fastapi import FastAPI
import json
import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import numpy as np
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

gb_model = mlflow.sklearn.load_model("/mlflow/mlruns/1/models/m-f38632264a0544bebfd9d74338d6aaf2/artifacts")
rf_model = mlflow.sklearn.load_model("/mlflow/mlruns/1/models/m-4deec2c6562945849bd2a9c795944c22/artifacts")

models_dir = os.getenv("MODELS_DIR", "models")

with open(f"{models_dir}/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

with open(f"{models_dir}/incident_type_cols.json", "r") as f:
    incident_type_cols = json.load(f)

with open(f"{models_dir}/pa_cat_cols.json", "r") as f:
    pa_cat_cols = json.load(f)

app = FastAPI(title='FEMA Recovery Cost Predictor')

class DisasterRequest(BaseModel):
    states_affected: int
    counties_affected: int
    ih_declared: int
    ia_declared: int
    pa_declared: int
    hm_declared: int
    tribal_request: int
    fy_declared: int
    incident_begin:   date
    incident_end:     date
    declaration_date: date
    totalAmountIaApproved: float
    project_count: int
    unique_applicants: int
    unique_damage_categories: int
    pa_cat_A: int
    pa_cat_B: int
    pa_cat_C: int
    pa_cat_D: int
    pa_cat_E: int
    pa_cat_F: int
    pa_cat_G: int
    pa_cat_I: int
    pa_cat_Z: int
    federal_share_ratio: float
    incident_type: str

def process_request(data: dict):
    data['incident_duration_days'] = (data['incident_end'] - data['incident_begin']).days
    data['declaration_lag_days']   = (data['declaration_date'] - data['incident_begin']).days
    data['declaration_month']      = data['declaration_date'].month

    # Check for unknown type BEFORE the loop
    known_types = [col.replace("incident_type_", "") for col in incident_type_cols]
    if data['incident_type'] not in known_types:
        data['incident_type'] = 'Other'

    # Now encode
    for col in incident_type_cols:
        type_name = col.replace('incident_type_', '')
        data[col] = 1 if type_name == data['incident_type'] else 0

    return data

@app.get('/')

def root():
    return {"message": "FEMA Recovery Cost Predictor API is running"}

@app.post('/predict')
def predict(request: DisasterRequest):
    data = request.model_dump()
    data = process_request(data)
    data = pd.DataFrame([data])
    data = data[feature_columns]

    gb_pred = np.expm1(gb_model.predict(data)[0])
    rf_pred = np.expm1(rf_model.predict(data)[0])

    return {
        "gradient_boosting_prediction": f"${gb_pred:,.2f}",
        "random_forest_prediction": f"${rf_pred:,.2f}"
    }
    
    
