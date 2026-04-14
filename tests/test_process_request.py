"""
Unit tests for process_request() in deployment/main.py.

These tests call the function directly with a plain dict — no HTTP, no models.
They verify the feature engineering logic in isolation.
"""
from datetime import date

from deployment.main import process_request 


def test_incident_duration_days(process_data):
    """
    process_request() should compute the number of days between incident_begin
    and incident_end. Here that's Aug 20 → Aug 30 = 10 days.

    This derived feature replaces the raw dates in the model input — the model
    was trained on durations, not calendar dates.
    """
    result = process_request(process_data)
    assert result["incident_duration_days"] == 10


def test_declaration_lag_days(process_data):
    """
    Declaration lag is how many days after the incident started the disaster
    was officially declared. Here Aug 20 → Sep 1 = 12 days.

    A longer lag might signal a slower-developing or more complex disaster.
    """
    result = process_request(process_data)
    assert result["declaration_lag_days"] == 12


def test_declaration_month(process_data):
    """
    The month of declaration (1–12) is extracted as a numeric feature.
    Sep 1 → month 9.

    Month captures seasonality: hurricane season, wildfire season, etc. all
    cluster in certain months and correlate with cost.
    """
    result = process_request(process_data)
    assert result["declaration_month"] == 9


def test_known_incident_type_sets_correct_column_to_one(process_data):
    """
    One-hot encoding: for a known incident type (Hurricane), exactly the
    matching column (incident_type_Hurricane) should be 1, and all others 0.

    This is the standard way to feed a categorical variable to a tree model
    without implying any ordering between categories.
    """
    result = process_request(process_data)
    assert result["incident_type_Hurricane"] == 1


def test_known_incident_type_sets_all_other_columns_to_zero(process_data):
    """
    The flip side of the above: every other incident_type_* column must be 0.
    If two columns were 1, the model would see a contradictory input.
    """
    result = process_request(process_data)
    other_cols = [
        k for k in result
        if k.startswith("incident_type_") and k != "incident_type_Hurricane"
    ]
    assert all(result[col] == 0 for col in other_cols)


def test_unknown_incident_type_maps_to_other(process_data):
    """
    If the API receives an incident type the model was never trained on
    (e.g. "Volcano"), process_request() should silently remap it to "Other"
    before encoding. This prevents a KeyError and keeps the model input valid.

    The check happens BEFORE the encoding loop — that ordering matters,
    which is why this test exists.
    """
    process_data["incident_type"] = "Volcano"
    result = process_request(process_data)
    assert result["incident_type_Other"] == 1


def test_unknown_type_does_not_set_original_column(process_data):
    """
    Paired with the above: "Volcano" should NOT produce an
    incident_type_Volcano column (which would be ignored by the model
    anyway, but it confirms the remapping worked cleanly).
    """
    process_data["incident_type"] = "Volcano"
    result = process_request(process_data)
    assert "incident_type_Volcano" not in result
