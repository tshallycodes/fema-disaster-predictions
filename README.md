# FEMA Disaster Recovery Cost Predictor

A machine learning system that predicts federal disaster recovery costs at the earliest possible stage of a disaster declaration. Built for **TerraNova** as part of a Disaster Recovery Cost Forecasting Framework.

---

## Problem Statement

When a disaster is declared, federal agencies and state governments need to rapidly estimate recovery costs for budget planning and resource allocation. Currently this process is manual, slow, and inconsistent.

This project predicts total federal recovery expenditure — covering Public Assistance, Hazard Mitigation, and Individual Assistance obligations — using information available at the time of declaration combined with early Public Assistance activity data.

---

## Dataset

Data is sourced from the **FEMA OpenFEMA API** across three endpoints:

| Dataset | Endpoint | Description |
|---|---|---|
| `declarations.csv` | `/api/open/v2/DisasterDeclarationsSummaries` | One row per county per disaster — declaration metadata, programme flags, incident dates |
| `disaster_summaries.csv` | `/api/open/v1/FemaWebDisasterSummaries` | One row per disaster — total federal obligations (target variable source) |
| `public_assistance.csv` | `/api/open/v2/PublicAssistanceFundedProjectsDetails` | One row per PA project — damage categories, applicants, project counts |

Filtered to disaster numbers 1239–4899. Final dataset: **1,765 disasters** after joining on shared disaster numbers.

---

## Target Variable

```
total_recovery_cost = totalObligatedAmountPa
                    + totalObligatedAmountHmgp
                    + totalAmountIhpApproved
```

Log-transformed at training time (`np.log1p`) to handle right skew. Predictions are reverse-transformed (`np.expm1`) before being returned to the user.

---

## Pipeline

```
declarations.csv          ┐
disaster_summaries.csv    ├──► data.py ──► master.csv
public_assistance.csv     ┘

master.csv ──► feature_engineering.py ──► ft_eng.csv

ft_eng.csv ──► model.ipynb ──► trained models + MLflow tracking

trained models ──► FastAPI (main.py) ──► Streamlit (app.py)
                        ↑
                   Docker Compose
```

---

## Features

### Declaration-time features (known at Day 1)
| Feature | Description |
|---|---|
| `states_affected` | Number of states included in the declaration |
| `counties_affected` | Number of designated counties |
| `incident_type` | Type of disaster (Hurricane, Flood, Fire, etc.) |
| `incident_duration_days` | Days between incident start and end |
| `declaration_lag_days` | Days between incident start and federal declaration |
| `declaration_month` | Month of declaration — captures seasonality |
| `fy_declared` | Federal fiscal year declared |
| `ih/ia/pa/hm_declared` | Programme flags — which assistance types were activated |
| `tribal_request` | Whether a tribal government requested the declaration |

### Public Assistance activity features (early recovery)
| Feature | Description |
|---|---|
| `project_count` | Number of PA projects filed |
| `unique_applicants` | Number of distinct organisations that applied |
| `unique_damage_categories` | Number of damage category types present |
| `federal_share_ratio` | Federal share as a proportion of total obligations |
| `pa_cat_A` through `pa_cat_Z` | Project counts per damage category |

---

## Model Performance

Three models trained and evaluated using 5-fold cross-validation:

| Model | R² Score | Median Absolute Error |
|---|---|---|
| Ridge Regression | 0.50 | $3,155,743 |
| Random Forest | 0.83 | $3,085,040 |
| **Gradient Boosting** | **0.85** | **$2,545,705** |

**Gradient Boosting is the primary model** for predictions. Typical disaster in the dataset costs ~$8.9M, so the model predicts within ~30% on a typical disaster.

### Key predictors (SHAP analysis)
- `hm_declared` — Hazard Mitigation activation is the strongest signal for high-cost disasters
- `counties_affected` — wider geographic spread means higher cost
- `incident_duration_days` — longer disasters cost more
- `total_federal_obligated` — early PA obligations signal total recovery scale

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python, Pandas, NumPy | Data processing and feature engineering |
| Scikit-learn | Model training and evaluation |
| MLflow | Experiment tracking and model logging |
| SHAP | Feature importance and explainability |
| FastAPI | REST API for serving predictions |
| Streamlit | Interactive user interface |
| Docker + Docker Compose | Containerised deployment |
| python-dotenv | Environment variable management |

---

## Project Structure

```
fema-predictions/
├── data/
│   └── raw/csv/                  # Raw FEMA CSV files
├── src/
│   ├── data.py                   # Load, filter, aggregate → master.csv
│   ├── feature_engineering.py    # Clean, encode, transform → ft_eng.csv
│   ├── model.ipynb               # Train models, log to MLflow
│   ├── mlflow.db                 # MLflow tracking database
│   └── mlruns/                   # MLflow artifacts
├── models/
│   ├── fema_gb_model.pkl         # Gradient Boosting model
│   ├── fema_rf_model.pkl         # Random Forest model
│   ├── fema_ridge_model.pkl      # Ridge Regression model
│   ├── feature_columns.json      # Ordered feature list for inference
│   ├── incident_type_cols.json   # One-hot encoded incident type columns
│   └── pa_cat_cols.json          # PA damage category columns
├── deployment/
│   ├── main.py                   # FastAPI prediction API
│   └── app.py                    # Streamlit user interface
├── docker/
│   ├── Dockerfile.api            # API container build instructions
│   └── Dockerfile.streamlit      # Streamlit container build instructions
├── tests/                        # Pytest test suite
├── docker-compose.yml            # Multi-container orchestration
├── requirements.txt              # Python dependencies
└── .env                          # Environment variables (not committed)
```

---

## Setup and Usage

### Prerequisites
- Python 3.11+
- Docker Desktop

### 1. Clone the repository
```bash
git clone https://github.com/tshallycodes/fema-predictions.git
cd fema-predictions
```

### 2. Create your `.env` file
```
MLFLOW_TRACKING_URI=sqlite:///src/mlflow.db
MODELS_DIR=/app/models
API_URL=http://api:8000
```

### 3. Install dependencies (for local development)
```bash
pip install -r requirements.txt
```

### 4. Run the data pipeline
```bash
python src/data.py
python src/feature_engineering.py
```

### 5. Train the models
Open and run `src/model.ipynb`. Models are saved to `models/` and logged to MLflow.

### 6. Run with Docker
```bash
docker-compose up --build
```

Then open:
- **Streamlit UI**: http://localhost:8501
- **FastAPI docs**: http://localhost:8000/docs
- **MLflow UI**: run `mlflow ui` locally pointing to `src/mlflow.db`

### 7. Run locally without Docker
```bash
# Terminal 1 — API
uvicorn deployment.main:app --reload

# Terminal 2 — Streamlit
streamlit run deployment/app.py
```

---

## API Reference

### `POST /predict`

Accepts disaster declaration details and returns predicted recovery cost.

**Request body:**
```json
{
  "incident_type": "Hurricane",
  "states_affected": 2,
  "counties_affected": 45,
  "incident_begin": "2024-08-20",
  "incident_end": "2024-09-05",
  "declaration_date": "2024-08-22",
  "ih_declared": 1,
  "ia_declared": 1,
  "pa_declared": 1,
  "hm_declared": 1,
  "tribal_request": 0,
  "fy_declared": 2024,
  "totalAmountIaApproved": 5000000,
  "project_count": 150,
  "unique_applicants": 20,
  "unique_damage_categories": 4,
  "pa_cat_A": 30,
  "pa_cat_B": 50,
  "pa_cat_C": 20,
  "pa_cat_D": 10,
  "pa_cat_E": 15,
  "pa_cat_F": 10,
  "pa_cat_G": 5,
  "pa_cat_I": 0,
  "pa_cat_Z": 10,
  "federal_share_ratio": 0.75
}
```

**Response:**
```json
{
  "gradient_boosting_prediction": "$11,664,590.49",
  "random_forest_prediction": "$8,554,603.93"
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Limitations

- PA activity features (`project_count`, `pa_cat_*` etc.) are only available after recovery begins, limiting true Day 1 prediction capability. A fully early-prediction variant using declaration-time features only achieves R²=0.63.
- Model is trained on US federal disasters from 1990s onwards — performance may degrade for disaster types with few historical examples.
- Dollar predictions should be treated as directional estimates, not precise budget figures.

---

## Author

**Chukwuebuka** — University of Bradford, AI Student  
GitHub: [@tshallycodes](https://github.com/tshallycodes)
