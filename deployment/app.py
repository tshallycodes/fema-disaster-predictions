import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import date
from dotenv import load_dotenv
import requests
import os

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FEMA Recovery Cost Predictor", layout="wide")
st.title("FEMA Recovery Cost Predictor")

with st.form("fema_form"):
    states_affected = st.number_input("States Affected", min_value=1, max_value=50, value=1)
    counties_affected = st.number_input("Counties Affected", min_value=1, max_value=3000, value=1)
    ih_declared = st.checkbox("IH Declared (Individual and Households Program)")
    ia_declared = st.checkbox("IA Declared (Public Assistance - Individuals and Households)")
    pa_declared = st.checkbox("PA Declared (Public Assistance - Public Infrastructure)")
    hm_declared = st.checkbox("HM Declared (Hazard Mitigation Grant Program)")
    tribal_request = st.checkbox("Tribal Request (Tribal Lands)")
    fy_declared = st.number_input("Fiscal Year Declared", min_value=1990, max_value=date.today().year, value=date.today().year)
    incident_begin = st.date_input("Incident Begin", value=date.today())
    incident_end = st.date_input("Incident End", value=date.today())
    declaration_date = st.date_input("Declaration Date", value=date.today())
    total_amount_ia_approved = st.number_input("Total Amount IA Approved", min_value=0.0, max_value=1e12, value=0.0)
    project_count = st.number_input("Project Count", min_value=1, max_value=1000000, value=1)
    unique_applicants = st.number_input("Unique Applicants", min_value=0, max_value=1000000, value=0)
    unique_damage_categories = st.number_input("Unique Damage Categories", min_value=0, max_value=100, value=0)
    pa_cat_A = st.number_input("Count of Category A Projects (Debris Removal)", min_value=0, max_value=1000000, value=0)
    pa_cat_B = st.number_input("Count of Category B Projects (Emergency Protective Measures)", min_value=0, max_value=1000000, value=0)
    pa_cat_C = st.number_input("Count of Category C Projects (Roads and Bridges)", min_value=0, max_value=1000000, value=0)
    pa_cat_D = st.number_input("Count of Category D Projects (Water Control Facilities)", min_value=0, max_value=1000000, value=0)
    pa_cat_E = st.number_input("Count of Category E Projects (Buildings and Equipment)", min_value=0, max_value=1000000, value=0)
    pa_cat_F = st.number_input("Count of Category F Projects (Public Utilities)", min_value=0, max_value=1000000, value=0)
    pa_cat_G = st.number_input("Count of Category G Projects (Parks, Recreational, and Other Facilities)", min_value=0, max_value=1000000, value=0)
    pa_cat_I = st.number_input("Count of Category I Projects (Other)", min_value=0, max_value=1000000, value=0)
    pa_cat_Z = st.number_input("Count of Category Z Projects (Other)", min_value=0, max_value=1000000, value=0)
    federal_share_ratio = st.number_input("Federal Share Ratio", min_value=0.0, max_value=1.0, value=0.75)
    incident_type = st.selectbox("Incident Type", options=["Biological", "Coastal Storm", "Earthquake", "Fire", "Flood", "Hurricane", "Other", "Severe Ice Storm", "Severe Storm", "Snowstorm", "Tornado", "Tropical Storm", "Typhoon", "Winter Storm"])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Prediction
if submitted:
    payload = {
        "states_affected": int(states_affected),
        "counties_affected": int(counties_affected),
        "ih_declared": int(ih_declared),
        "ia_declared": int(ia_declared),
        "pa_declared": int(pa_declared),
        "hm_declared": int(hm_declared),
        "tribal_request": int(tribal_request),
        "fy_declared": int(fy_declared),
        "incident_begin": incident_begin.isoformat(),
        "incident_end": incident_end.isoformat(),
        "declaration_date": declaration_date.isoformat(),
        "totalAmountIaApproved": total_amount_ia_approved,
        "project_count": int(project_count),
        "unique_applicants": int(unique_applicants),
        "unique_damage_categories": int(unique_damage_categories),
        "pa_cat_A": int(pa_cat_A),
        "pa_cat_B": int(pa_cat_B),
        "pa_cat_C": int(pa_cat_C),
        "pa_cat_D": int(pa_cat_D),
        "pa_cat_E": int(pa_cat_E),
        "pa_cat_F": int(pa_cat_F),
        "pa_cat_G": int(pa_cat_G),
        "pa_cat_I": int(pa_cat_I),
        "pa_cat_Z": int(pa_cat_Z),
        "federal_share_ratio": federal_share_ratio,
        "incident_type": incident_type
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    result = response.json()
    st.success(f"Gradient Boosting: {result['gradient_boosting_prediction']}")
    st.success(f"Random Forest: {result['random_forest_prediction']}")
