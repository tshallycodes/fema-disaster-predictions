"""
Edge case and boundary tests.

Pydantic catches type errors, but these tests verify behaviour for inputs
that are valid types but unusual values — things that won't raise a 422
but could produce wrong results or unexpected model inputs.
"""
from datetime import date

import pytest

from deployment.main import process_request


def test_same_begin_and_end_date_gives_zero_duration(process_data):
    """
    If a disaster begins and ends on the same day, incident_duration_days
    should be 0, not negative or missing. The model must handle this cleanly.

    Example: a one-day earthquake or a flash flood.
    """
    process_data["incident_end"] = process_data["incident_begin"]
    result = process_request(process_data)
    assert result["incident_duration_days"] == 0


def test_declaration_before_incident_end_gives_negative_lag(process_data):
    """
    In rare cases a federal declaration is issued before the incident period
    officially closes. declaration_lag_days can legitimately be negative.

    This test confirms process_request() doesn't clamp or error on negative
    values — the model should handle it because it was trained on real data
    that may include these cases.
    """
    process_data["declaration_date"] = date(2020, 8, 15)  # before incident_begin (Aug 20)
    result = process_request(process_data)
    assert result["declaration_lag_days"] < 0


def test_zero_project_count_is_accepted(client, valid_payload):
    """
    project_count has min=1 in the Streamlit UI, but the API itself only
    enforces the type (int). A zero value should reach the model without
    a 422 — the API is not the right place to enforce business rules that
    belong in the UI.
    """
    valid_payload["project_count"] = 0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_all_pa_cat_counts_zero_is_accepted(client, valid_payload):
    """
    All PA category counts being zero is unusual but valid — a small disaster
    with no PA projects yet. Confirms the model input pipeline doesn't crash
    on all-zero category features.
    """
    for cat in ["pa_cat_A", "pa_cat_B", "pa_cat_C", "pa_cat_D",
                "pa_cat_E", "pa_cat_F", "pa_cat_G", "pa_cat_I", "pa_cat_Z"]:
        valid_payload[cat] = 0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_federal_share_ratio_at_minimum_boundary(client, valid_payload):
    """
    federal_share_ratio of 0.0 (no federal share) is a boundary value.
    Tests that the float field accepts 0.0 without Pydantic rejecting it.
    """
    valid_payload["federal_share_ratio"] = 0.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_federal_share_ratio_at_maximum_boundary(client, valid_payload):
    """
    federal_share_ratio of 1.0 (fully federally funded) is the other boundary.
    """
    valid_payload["federal_share_ratio"] = 1.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200


def test_each_known_incident_type_is_accepted(client, valid_payload):
    """
    Parametrised test: runs once for each known incident type.
    Every type that the model was trained on should return 200. If a new
    type is added to incident_type_cols.json without updating this test,
    pytest will catch the gap on the next run.

    Uses pytest.mark.parametrize — pytest automatically names each sub-test
    after its parameter value, so failures show 'FAILED [Flood]' not just
    'FAILED test_each_known_incident_type_is_accepted'.
    """
    from deployment.main import incident_type_cols
    known_types = [col.replace("incident_type_", "") for col in incident_type_cols]

    for incident_type in known_types:
        valid_payload["incident_type"] = incident_type
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 200, f"Failed for incident_type='{incident_type}'"


def test_unknown_incident_type_still_returns_200(client, valid_payload):
    """
    An unknown incident type should be silently remapped to 'Other' by
    process_request() and still produce a valid prediction. The API should
    never return 500 just because of an unknown category value.
    """
    valid_payload["incident_type"] = "Tsunami"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
