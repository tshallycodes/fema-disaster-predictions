"""
Integration tests for the FastAPI endpoints.

Requests go through the full HTTP stack (routing, Pydantic parsing,
response serialisation) but the ML models are mocked, so these tests
run without Docker or MLflow.
"""
import pytest


def test_root_returns_200(client):
    """
    The GET / health-check route should always return 200 with a message.
    If this fails, the app didn't start at all — a useful baseline check.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_valid_input_returns_200(client, valid_payload):
    """
    A complete, correctly-typed payload should return HTTP 200.
    This is the happy path — confirms the full pipeline runs end-to-end.
    """
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_response_contains_both_predictions(client, valid_payload):
    """
    The response body must have exactly the two keys the Streamlit app reads.
    If a key is renamed in main.py but not app.py, this test catches it.
    """
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert "gradient_boosting_prediction" in body
    assert "random_forest_prediction" in body


def test_predict_response_values_are_dollar_strings(client, valid_payload):
    """
    Predictions are formatted as dollar strings (e.g. '$22,026.47').
    The mock model returns np.array([10.0]), and np.expm1(10.0) ≈ 22026.47,
    so we can assert the exact format.

    This test would catch a regression where someone accidentally returns
    a raw float instead of the formatted string.
    """
    response = client.post("/predict", json=valid_payload)
    body = response.json()
    assert body["gradient_boosting_prediction"].startswith("$")
    assert body["random_forest_prediction"].startswith("$")


def test_predict_missing_required_field_returns_422(client, valid_payload):
    """
    FastAPI uses Pydantic to validate the request body. If a required field
    is missing, Pydantic raises a validation error and FastAPI returns 422
    Unprocessable Entity automatically — no custom error handling needed.

    422 means 'I understood the request but the data is invalid', which is
    more precise than 400 Bad Request.
    """
    del valid_payload["incident_type"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_wrong_type_for_int_field_returns_422(client, valid_payload):
    """
    Sending a string where an int is expected (states_affected: "three")
    should be rejected by Pydantic with a 422. This confirms type coercion
    is NOT silently happening — a string that can't be cast to int is an error.
    """
    valid_payload["states_affected"] = "three"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_extra_fields_are_ignored(client, valid_payload):
    """
    Pydantic's default behaviour is to silently ignore fields that aren't
    in the model. This test confirms an extra field doesn't cause a crash
    or a 422 — it's just dropped.
    """
    valid_payload["unknown_field"] = "should be ignored"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_predict_invalid_date_format_returns_422(client, valid_payload):
    """
    Dates must be ISO format (YYYY-MM-DD). A non-date string should be
    rejected by Pydantic before process_request() ever runs.
    """
    valid_payload["incident_begin"] = "not-a-date"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
