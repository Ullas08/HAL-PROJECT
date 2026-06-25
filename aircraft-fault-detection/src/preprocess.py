"""
src/preprocess.py
=================
Reusable preprocessing pipeline functions for the Aircraft Fault Detection project.
All steps follow the specification in the Implementation Plan (Phase 2).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_cmapss(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load NASA C-MAPSS FD001 train file and attach a binary fault label.
    RUL <= 30 cycles  => FAULT (1),  else NORMAL (0).
    """
    col_names = (
        ["unit", "cycle"]
        + [f"op{i}" for i in range(1, 4)]
        + [f"sensor{i}" for i in range(1, 22)]
    )
    path = os.path.join(raw_dir, "train_FD001.txt")
    df = pd.read_csv(path, sep=r"\s+", header=None, names=col_names)

    # Compute Remaining Useful Life per unit
    max_cycle = df.groupby("unit")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="unit")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df["fault"] = (df["RUL"] <= 30).astype(int)

    # Drop non-feature columns
    df.drop(columns=["unit", "max_cycle", "RUL"], inplace=True)
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Generic CSV loader for any tabular dataset."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# 2. Missing Value Imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame, threshold: float = 0.40) -> pd.DataFrame:
    """
    - Drop columns with > threshold missing ratio.
    - Impute remaining numeric columns with column median.
    """
    missing_ratio = df.isnull().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    if drop_cols:
        print(f"[impute] Dropping columns (>{threshold*100:.0f}% missing): {drop_cols}")
    df = df.drop(columns=drop_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


# ---------------------------------------------------------------------------
# 3. Outlier Capping (IQR)
# ---------------------------------------------------------------------------

def cap_outliers(df: pd.DataFrame, target_col: str = "fault") -> pd.DataFrame:
    """
    Cap feature values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    The target column is excluded from capping.
    """
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_col]
    for col in feature_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ---------------------------------------------------------------------------
# 4. Train / Test Split (stratified)
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_col: str = "fault",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Return X_train, X_test, y_train, y_test with stratified split."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return train_test_split(X, y, test_size=test_size,
                            stratify=y, random_state=random_state)


# ---------------------------------------------------------------------------
# 5. Feature Scaling
# ---------------------------------------------------------------------------

def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler: StandardScaler,
):
    """Transform both splits using the pre-fitted scaler."""
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# 6. Class Imbalance — SMOTE
# ---------------------------------------------------------------------------

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    minority_threshold: float = 0.30,
    random_state: int = 42,
):
    """
    Apply SMOTE only when fault class ratio < minority_threshold.
    Never apply to the test set.
    """
    fault_ratio = y_train.mean()
    if fault_ratio < minority_threshold:
        print(f"[SMOTE] Fault ratio={fault_ratio:.3f} < {minority_threshold}. Applying SMOTE...")
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"[SMOTE] After: {pd.Series(y_res).value_counts().to_dict()}")
        return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res)
    else:
        print(f"[SMOTE] Fault ratio={fault_ratio:.3f} >= {minority_threshold}. Skipping SMOTE.")
        return X_train, y_train


# ---------------------------------------------------------------------------
# 7. Feature Selection
# ---------------------------------------------------------------------------

def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 15,
    corr_threshold: float = 0.95,
    random_state: int = 42,
) -> list:
    """
    1. Train a Random Forest and keep top_n features by importance.
    2. From those, drop one column from each highly-correlated pair (|r| > corr_threshold).
    Returns list of selected feature names.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(top_n).index.tolist()

    # Drop correlated features
    corr_matrix = X_train[top_features].corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [col for col in upper_tri.columns
               if any(upper_tri[col] > corr_threshold)]
    selected = [f for f in top_features if f not in to_drop]
    print(f"[feature_select] Top-{top_n} -> after corr filter: {len(selected)} features")
    print(f"[feature_select] Selected: {selected}")
    return selected


# ---------------------------------------------------------------------------
# 8. Full Pipeline (convenience wrapper)
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str = "fault",
    top_n_features: int = 15,
    save_dir: str = "data/processed",
    model_dir: str = "models",
) -> dict:
    """
    Run the complete preprocessing pipeline end-to-end.
    Returns a dict with X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    print("=== PREPROCESSING PIPELINE ===")
    print(f"Input shape: {df.shape}")

    df = impute_missing(df)
    df = cap_outliers(df, target_col=target_col)

    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fault ratio: {y_train.mean():.3f}")

    scaler = fit_scaler(X_train)
    X_train_sc, X_test_sc = apply_scaler(X_train, X_test, scaler)

    X_train_res, y_train_res = apply_smote(X_train_sc, y_train)

    selected_features = select_features(X_train_res, y_train_res, top_n=top_n_features)

    X_train_final = X_train_res[selected_features]
    X_test_final = X_test_sc[selected_features]

    # Save processed datasets
    os.makedirs(save_dir, exist_ok=True)
    train_out = pd.concat([X_train_final.reset_index(drop=True),
                            y_train_res.reset_index(drop=True).rename(target_col)], axis=1)
    test_out = pd.concat([X_test_final.reset_index(drop=True),
                           y_test.reset_index(drop=True).rename(target_col)], axis=1)
    train_out.to_csv(os.path.join(save_dir, "train_clean.csv"), index=False)
    test_out.to_csv(os.path.join(save_dir, "test_clean.csv"), index=False)
    print(f"[save] Saved processed datasets to {save_dir}/")

    # Save scaler
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    print(f"[save] Scaler saved to {model_dir}/scaler.pkl")

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_train": y_train_res,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": selected_features,
    }
