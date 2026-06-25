"""
tests/test_app_logic.py
========================
Unit tests for dashboard prediction logic —
scaler transform + model prediction on sample input.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from train import train_all_models
from preprocess import fit_scaler


SENSOR_NAMES = [
    "T2_fan_inlet_temp", "T24_lpc_outlet_temp", "T30_hpc_outlet_temp",
    "T50_lpt_outlet_temp", "P2_fan_inlet_pres", "P15_bypass_duct_pres",
    "P30_hpc_outlet_pres", "Nf_fan_speed", "Nc_core_speed",
    "epr_engine_pres_ratio", "Ps30_hpc_static_pres", "phi_ratio_fuel_flow",
    "NRf_corrected_fan_speed", "NRc_corrected_core_speed", "BPR_bypass_ratio",
]


@pytest.fixture
def trained_artifacts(tmp_path):
    """Train a quick RF model and scaler for dashboard tests."""
    np.random.seed(7)
    n = 300
    X = pd.DataFrame(
        np.random.randn(n, len(SENSOR_NAMES)),
        columns=SENSOR_NAMES,
    )
    y = pd.Series(np.random.randint(0, 2, n), name="fault")

    scaler = fit_scaler(X)
    X_sc = pd.DataFrame(scaler.transform(X), columns=SENSOR_NAMES)
    joblib.dump(scaler, tmp_path / "scaler.pkl")

    # Train only RF for speed
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(X_sc, y)
    joblib.dump(model, tmp_path / "champion_model.pkl")

    return tmp_path, scaler, model


def test_scaler_transform_single_row(trained_artifacts):
    """Scaler should transform a single input row without error."""
    tmp_path, scaler, _ = trained_artifacts
    raw_input = np.array([[518, 642, 1590, 1400, 14.6, 21.6, 550,
                            2388, 9046, 1.3, 47.5, 521, 2388, 8138, 8.4]])
    scaled = scaler.transform(raw_input)
    assert scaled.shape == (1, len(SENSOR_NAMES))
    assert not np.any(np.isnan(scaled))


def test_model_predict_single_row(trained_artifacts):
    """Model should return a single binary prediction."""
    tmp_path, scaler, model = trained_artifacts
    raw_input = np.array([[518, 642, 1590, 1400, 14.6, 21.6, 550,
                            2388, 9046, 1.3, 47.5, 521, 2388, 8138, 8.4]])
    scaled = scaler.transform(raw_input)
    pred = model.predict(scaled)
    assert pred.shape == (1,)
    assert pred[0] in {0, 1}


def test_model_predict_proba_single_row(trained_artifacts):
    """predict_proba should return shape (1, 2) with sum ≈ 1."""
    tmp_path, scaler, model = trained_artifacts
    raw_input = np.array([[518, 642, 1590, 1400, 14.6, 21.6, 550,
                            2388, 9046, 1.3, 47.5, 521, 2388, 8138, 8.4]])
    scaled = scaler.transform(raw_input)
    proba = model.predict_proba(scaled)
    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-5


def test_model_predict_multiple_rows(trained_artifacts):
    """Model should handle batches correctly."""
    tmp_path, scaler, model = trained_artifacts
    np.random.seed(99)
    raw_batch = np.random.randn(10, len(SENSOR_NAMES))
    scaled = scaler.transform(raw_batch)
    preds = model.predict(scaled)
    assert preds.shape == (10,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_saved_model_loads_and_predicts(trained_artifacts):
    """Saved champion_model.pkl and scaler.pkl should reload cleanly."""
    tmp_path, _, _ = trained_artifacts
    model  = joblib.load(tmp_path / "champion_model.pkl")
    scaler = joblib.load(tmp_path / "scaler.pkl")

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert hasattr(scaler, "transform")

    raw = np.ones((1, len(SENSOR_NAMES)))
    scaled = scaler.transform(raw)
    pred = model.predict(scaled)
    assert pred[0] in {0, 1}
