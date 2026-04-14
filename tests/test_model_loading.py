"""
Tests for model artefact loading.

The ML models themselves are mocked in conftest.py (they live inside Docker),
but the JSON metadata files are real files on disk and must be readable in
any environment where tests run. These tests guard against accidentally
deleting or corrupting those files.
"""
import json
import os

import numpy as np
import pytest

from deployment.main import feature_columns, gb_model, incident_type_cols, pa_cat_cols, rf_model


def test_feature_columns_is_a_non_empty_list():
    """
    feature_columns.json drives the final column selection before prediction.
    If it's empty or not a list, every prediction will silently produce
    a zero-column DataFrame and the model will crash or return garbage.
    """
    assert isinstance(feature_columns, list)
    assert len(feature_columns) > 0


def test_incident_type_cols_is_a_non_empty_list():
    """
    incident_type_cols.json is what process_request() loops over to build
    the one-hot encoding. An empty list would silently skip all encoding.
    """
    assert isinstance(incident_type_cols, list)
    assert len(incident_type_cols) > 0


def test_pa_cat_cols_is_a_non_empty_list():
    """
    pa_cat_cols.json defines which PA category columns exist. Loaded at
    startup — if corrupted, process_request() will produce wrong features.
    """
    assert isinstance(pa_cat_cols, list)
    assert len(pa_cat_cols) > 0


def test_incident_type_cols_all_have_prefix():
    """
    Every entry in incident_type_cols should start with 'incident_type_'.
    The encoding loop in process_request() strips this prefix to get the
    type name — if any entry is malformed the encoding silently breaks.
    """
    assert all(col.startswith("incident_type_") for col in incident_type_cols)


def test_feature_columns_includes_derived_date_features():
    """
    The three date-derived features must be in feature_columns, because
    process_request() computes them and the model was trained on them.
    If they're missing, the column-selection step will raise a KeyError.
    """
    assert "incident_duration_days" in feature_columns
    assert "declaration_lag_days" in feature_columns
    assert "declaration_month" in feature_columns


def test_feature_columns_does_not_include_raw_dates():
    """
    Raw date columns (incident_begin, incident_end, declaration_date) should
    NOT be in feature_columns — they're used only to derive the numeric
    features and must be dropped before the model sees the DataFrame.
    """
    assert "incident_begin" not in feature_columns
    assert "incident_end" not in feature_columns
    assert "declaration_date" not in feature_columns


def test_mock_models_have_predict_method():
    """
    In the test environment, gb_model and rf_model are mocks. This test
    confirms they expose a .predict() method — the same interface the real
    sklearn models have. If conftest.py is misconfigured, this will fail
    before any other test does something confusing.
    """
    assert hasattr(gb_model, "predict")
    assert hasattr(rf_model, "predict")


def test_mock_model_predict_returns_array():
    """
    The predict function does np.expm1(model.predict(data)[0]), so the mock
    must return something indexable. Confirms the mock is wired correctly.
    """
    import pandas as pd
    dummy = pd.DataFrame([{"x": 1}])
    result = gb_model.predict(dummy)
    assert hasattr(result, "__getitem__")
