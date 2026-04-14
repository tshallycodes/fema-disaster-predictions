"""
Shared fixtures and setup for all tests.

The sys.modules patch at the top must happen before deployment.main is
imported anywhere, because main.py calls mlflow.sklearn.load_model() at
module level. conftest.py is loaded by pytest before any test file, so
this is the right place to intercept it.
"""
import os
import sys
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Point to the local models/ folder before load_dotenv() runs inside main.py.
# load_dotenv() skips variables that are already set, so this wins over .env.
os.environ["MODELS_DIR"] = "models"

# ── Mock mlflow before main.py is ever imported ────────────────────────────
_mock_mlflow = MagicMock()
_mock_model = MagicMock()
# expm1(10.0) ≈ $22,026.47 — gives tests a predictable dollar amount to assert on
_mock_model.predict.return_value = np.array([10.0])
_mock_mlflow.sklearn.load_model.return_value = _mock_model

sys.modules["mlflow"] = _mock_mlflow
sys.modules["mlflow.sklearn"] = _mock_mlflow.sklearn

# Safe to import now — model loading calls hit our mock instead of the filesystem
from deployment.main import app, process_request  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """A test HTTP client that talks to the FastAPI app in-process (no server needed)."""
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """
    A complete, valid JSON-serialisable request body.
    Used by integration and validation tests (dates as ISO strings because
    the request goes through JSON → Pydantic parsing).
    """
    return {
        "states_affected": 3,
        "counties_affected": 10,
        "ih_declared": 1,
        "ia_declared": 1,
        "pa_declared": 1,
        "hm_declared": 0,
        "tribal_request": 0,
        "fy_declared": 2020,
        "incident_begin": "2020-08-20",
        "incident_end": "2020-08-30",
        "declaration_date": "2020-09-01",
        "totalAmountIaApproved": 5_000_000.0,
        "project_count": 150,
        "unique_applicants": 100,
        "unique_damage_categories": 5,
        "pa_cat_A": 20,
        "pa_cat_B": 30,
        "pa_cat_C": 15,
        "pa_cat_D": 5,
        "pa_cat_E": 10,
        "pa_cat_F": 8,
        "pa_cat_G": 3,
        "pa_cat_I": 2,
        "pa_cat_Z": 1,
        "federal_share_ratio": 0.75,
        "incident_type": "Hurricane",
    }


@pytest.fixture # pytest.fixture is a decorator that is used to define a fixture. It is used to share data between tests.
# A fixture is a function that is called by pytest to provide data to a test function.
# The data is provided to the test function as an argument.
# The fixture is called once for each test function that uses it.
# The fixture is called in the order that it is defined.
def process_data():
    """
    A data dict ready to pass directly to process_request().
    Dates are Python date objects, not strings — that's what Pydantic
    produces after parsing a real request, and what process_request expects.
    """
    return {
        "states_affected": 3,
        "counties_affected": 10,
        "ih_declared": 1,
        "ia_declared": 1,
        "pa_declared": 1,
        "hm_declared": 0,
        "tribal_request": 0,
        "fy_declared": 2020,
        "incident_begin": date(2020, 8, 20),
        "incident_end": date(2020, 8, 30),
        "declaration_date": date(2020, 9, 1),
        "totalAmountIaApproved": 5_000_000.0,
        "project_count": 150,
        "unique_applicants": 100,
        "unique_damage_categories": 5,
        "pa_cat_A": 20, "pa_cat_B": 30, "pa_cat_C": 15, "pa_cat_D": 5,
        "pa_cat_E": 10, "pa_cat_F": 8, "pa_cat_G": 3, "pa_cat_I": 2,
        "pa_cat_Z": 1,
        "federal_share_ratio": 0.75,
        "incident_type": "Hurricane",
    }
