"""
tests/test_train.py
===================
Unit tests for src/train.py — model loading and prediction shape.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from train import get_model_registry, train_all_models
from preprocess import split_data, fit_scaler, apply_scaler


@pytest.fixture
def tiny_dataset():
    """Minimal dataset for fast training tests."""
    np.random.seed(1)
    n = 200
    X = pd.DataFrame({
        "f1": np.random.normal(0, 1, n),
        "f2": np.random.normal(0, 1, n),
        "f3": np.random.normal(0, 1, n),
    })
    y = pd.Series(np.random.randint(0, 2, n), name="fault")
    return X, y


def test_model_registry_has_six_models():
    registry = get_model_registry()
    assert len(registry) == 6


def test_model_registry_keys():
    registry = get_model_registry()
    expected = {"logistic_regression", "decision_tree", "random_forest",
                "svm", "knn", "xgboost"}
    assert set(registry.keys()) == expected


def test_each_model_has_estimator_and_grid():
    registry = get_model_registry()
    for name, cfg in registry.items():
        assert "estimator" in cfg, f"{name} missing estimator"
        assert "param_grid" in cfg, f"{name} missing param_grid"
        assert len(cfg["param_grid"]) > 0, f"{name} param_grid is empty"


def test_train_all_models_returns_dataframe(tiny_dataset, tmp_path):
    X, y = tiny_dataset
    results = train_all_models(
        X_train=X, y_train=y,
        models_dir=str(tmp_path),
        cv_folds=2,   # fast
    )
    assert isinstance(results, pd.DataFrame)
    assert "model" in results.columns
    assert "best_cv_f1" in results.columns
    assert len(results) == 6


def test_train_all_models_saves_pkl_files(tiny_dataset, tmp_path):
    X, y = tiny_dataset
    train_all_models(X_train=X, y_train=y, models_dir=str(tmp_path), cv_folds=2)
    expected_files = [
        "logistic_regression.pkl", "decision_tree.pkl", "random_forest.pkl",
        "svm.pkl", "knn.pkl", "xgboost.pkl",
    ]
    for fname in expected_files:
        assert (tmp_path / fname).exists(), f"Missing: {fname}"


def test_loaded_model_predict_shape(tiny_dataset, tmp_path):
    X, y = tiny_dataset
    train_all_models(X_train=X, y_train=y, models_dir=str(tmp_path), cv_folds=2)

    # Test each saved model
    for name in ["logistic_regression", "decision_tree", "random_forest",
                 "svm", "knn", "xgboost"]:
        model = joblib.load(tmp_path / f"{name}.pkl")
        preds = model.predict(X)
        assert preds.shape == (len(X),), f"{name}: wrong prediction shape"
        assert set(np.unique(preds)).issubset({0, 1}), f"{name}: unexpected label values"


def test_loaded_model_predict_proba_shape(tiny_dataset, tmp_path):
    X, y = tiny_dataset
    train_all_models(X_train=X, y_train=y, models_dir=str(tmp_path), cv_folds=2)

    for name in ["logistic_regression", "random_forest", "xgboost"]:
        model = joblib.load(tmp_path / f"{name}.pkl")
        assert hasattr(model, "predict_proba"), f"{name} should have predict_proba"
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2), f"{name}: wrong proba shape"
        # Probabilities should sum to ~1
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5), f"{name}: probabilities don't sum to 1"
