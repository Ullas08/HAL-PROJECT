"""
tests/test_preprocess.py
=========================
Unit tests for src/preprocess.py
Run with: pytest tests/ -v
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocess import (
    impute_missing,
    cap_outliers,
    split_data,
    fit_scaler,
    apply_scaler,
    apply_smote,
    select_features,
    run_preprocessing_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Small balanced dataframe for fast tests."""
    np.random.seed(42)
    n = 400
    df = pd.DataFrame({
        "sensor1": np.random.normal(500, 5, n),
        "sensor2": np.random.normal(640, 3, n),
        "sensor3": np.random.normal(1590, 10, n),
        "sensor4": np.random.normal(47, 1, n),
        "fault":   np.random.randint(0, 2, n),
    })
    return df


@pytest.fixture
def imbalanced_df():
    """Heavily imbalanced dataframe (5% fault) to trigger SMOTE."""
    np.random.seed(0)
    n = 500
    fault = np.zeros(n, dtype=int)
    fault[:25] = 1   # 5% fault
    df = pd.DataFrame({
        "sensor1": np.random.normal(500, 5, n),
        "sensor2": np.random.normal(640, 3, n),
        "fault": fault,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Imputation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_impute_no_missing(sample_df):
    """Frame with no NaNs passes through unchanged."""
    out = impute_missing(sample_df.copy())
    assert out.isnull().sum().sum() == 0
    assert out.shape == sample_df.shape


def test_impute_fills_nans(sample_df):
    df = sample_df.copy()
    df.loc[:10, "sensor1"] = np.nan
    out = impute_missing(df)
    assert out["sensor1"].isnull().sum() == 0


def test_impute_drops_high_missing_column(sample_df):
    df = sample_df.copy()
    df["bad_col"] = np.nan          # 100% missing → should be dropped
    out = impute_missing(df, threshold=0.40)
    assert "bad_col" not in out.columns


def test_impute_keeps_low_missing_column(sample_df):
    df = sample_df.copy()
    df.loc[:5, "sensor2"] = np.nan  # ~1.5% missing → should be kept
    out = impute_missing(df, threshold=0.40)
    assert "sensor2" in out.columns
    assert out["sensor2"].isnull().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Outlier capping tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cap_outliers_reduces_range(sample_df):
    df = sample_df.copy()
    df.loc[0, "sensor1"] = 99999.0   # extreme outlier
    df.loc[1, "sensor1"] = -99999.0
    out = cap_outliers(df, target_col="fault")
    assert out["sensor1"].max() < 99999.0
    assert out["sensor1"].min() > -99999.0


def test_cap_outliers_preserves_target(sample_df):
    original_target = sample_df["fault"].copy()
    out = cap_outliers(sample_df.copy(), target_col="fault")
    pd.testing.assert_series_equal(out["fault"], original_target)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Train-test split tests
# ─────────────────────────────────────────────────────────────────────────────

def test_split_shapes(sample_df):
    X_tr, X_te, y_tr, y_te = split_data(sample_df, target_col="fault", test_size=0.2)
    n = len(sample_df)
    assert len(X_tr) + len(X_te) == n
    assert len(y_tr) == len(X_tr)
    assert len(y_te) == len(X_te)


def test_split_stratified(sample_df):
    """Fault ratio in train and test should be within 5 percentage points."""
    X_tr, X_te, y_tr, y_te = split_data(sample_df, target_col="fault", test_size=0.2)
    ratio_tr = y_tr.mean()
    ratio_te = y_te.mean()
    assert abs(ratio_tr - ratio_te) < 0.05


def test_split_no_target_in_X(sample_df):
    X_tr, X_te, _, _ = split_data(sample_df, target_col="fault")
    assert "fault" not in X_tr.columns
    assert "fault" not in X_te.columns


# ─────────────────────────────────────────────────────────────────────────────
# 4. Scaler tests
# ─────────────────────────────────────────────────────────────────────────────

def test_scaler_mean_near_zero(sample_df):
    X_tr, X_te, _, _ = split_data(sample_df)
    scaler = fit_scaler(X_tr)
    X_tr_sc, X_te_sc = apply_scaler(X_tr, X_te, scaler)
    # Training columns should be ~zero mean after scaling
    assert all(abs(X_tr_sc.mean()) < 0.1)


def test_scaler_std_near_one(sample_df):
    X_tr, X_te, _, _ = split_data(sample_df)
    scaler = fit_scaler(X_tr)
    X_tr_sc, _ = apply_scaler(X_tr, X_te, scaler)
    assert all(abs(X_tr_sc.std() - 1.0) < 0.1)


def test_scaler_preserves_shape(sample_df):
    X_tr, X_te, _, _ = split_data(sample_df)
    scaler = fit_scaler(X_tr)
    X_tr_sc, X_te_sc = apply_scaler(X_tr, X_te, scaler)
    assert X_tr_sc.shape == X_tr.shape
    assert X_te_sc.shape == X_te.shape


# ─────────────────────────────────────────────────────────────────────────────
# 5. SMOTE tests
# ─────────────────────────────────────────────────────────────────────────────

def test_smote_applied_when_imbalanced(imbalanced_df):
    X_tr, _, y_tr, _ = split_data(imbalanced_df, target_col="fault", test_size=0.2)
    scaler = fit_scaler(X_tr)
    X_tr_sc, _ = apply_scaler(X_tr, X_tr, scaler)
    X_res, y_res = apply_smote(X_tr_sc, y_tr, minority_threshold=0.30)
    # After SMOTE fault ratio should increase
    assert y_res.mean() > y_tr.mean()


def test_smote_skipped_when_balanced(sample_df):
    X_tr, _, y_tr, _ = split_data(sample_df, target_col="fault", test_size=0.2)
    scaler = fit_scaler(X_tr)
    X_tr_sc, _ = apply_scaler(X_tr, X_tr, scaler)
    original_len = len(X_tr_sc)
    X_res, y_res = apply_smote(X_tr_sc, y_tr, minority_threshold=0.10)
    # Should not grow when already balanced
    assert len(X_res) == original_len


# ─────────────────────────────────────────────────────────────────────────────
# 6. Feature selection test
# ─────────────────────────────────────────────────────────────────────────────

def test_select_features_returns_subset(sample_df):
    X_tr, _, y_tr, _ = split_data(sample_df, target_col="fault", test_size=0.2)
    scaler = fit_scaler(X_tr)
    X_tr_sc, _ = apply_scaler(X_tr, X_tr, scaler)
    selected = select_features(X_tr_sc, y_tr, top_n=4)
    assert len(selected) <= 4
    assert all(f in X_tr_sc.columns for f in selected)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

def test_full_pipeline_output_shape(sample_df, tmp_path):
    out = run_preprocessing_pipeline(
        df=sample_df,
        target_col="fault",
        top_n_features=4,
        save_dir=str(tmp_path / "processed"),
        model_dir=str(tmp_path / "models"),
    )
    assert "X_train" in out
    assert "X_test" in out
    assert "y_train" in out
    assert "y_test" in out
    assert "scaler" in out
    assert len(out["feature_names"]) > 0
    # Check shapes are consistent
    assert len(out["X_train"]) == len(out["y_train"])
    assert len(out["X_test"]) == len(out["y_test"])


def test_full_pipeline_saves_files(sample_df, tmp_path):
    proc = tmp_path / "processed"
    mdls = tmp_path / "models"
    run_preprocessing_pipeline(
        df=sample_df,
        target_col="fault",
        top_n_features=4,
        save_dir=str(proc),
        model_dir=str(mdls),
    )
    assert (proc / "train_clean.csv").exists()
    assert (proc / "test_clean.csv").exists()
    assert (mdls / "scaler.pkl").exists()
