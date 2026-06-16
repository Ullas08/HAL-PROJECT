"""
src/train.py
============
Training script — trains all 6 ML algorithms with GridSearchCV and serialises
the best models to the models/ directory.

Usage (command line):
    python src/train.py --train data/processed/train_clean.csv \
                        --target fault \
                        --models_dir models
"""

import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, classification_report


# ---------------------------------------------------------------------------
# Model registry — algorithm + hyper-parameter grid
# ---------------------------------------------------------------------------

def get_model_registry(random_state: int = 42) -> dict:
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(solver="lbfgs", max_iter=1000,
                                            random_state=random_state),
            "param_grid": {"C": [0.01, 0.1, 1, 10]},
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, None],
            },
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=random_state),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
                "gamma": ["scale", "auto"],
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier(n_jobs=-1),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11],
                "metric": ["euclidean", "manhattan"],
            },
        },
        "xgboost": {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        },
    }


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models_dir: str = "models",
    cv_folds: int = 5,
    random_state: int = 42,
    scoring: str = "f1",
) -> pd.DataFrame:
    """
    Run GridSearchCV for all 6 algorithms.
    Saves each best estimator as <model_name>.pkl.
    Returns a DataFrame summarising CV results.
    """
    os.makedirs(models_dir, exist_ok=True)
    registry = get_model_registry(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results = []

    for name, config in registry.items():
        print(f"\n{'='*60}")
        print(f"Training: {name.upper()}")
        print(f"{'='*60}")

        gs = GridSearchCV(
            estimator=config["estimator"],
            param_grid=config["param_grid"],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        gs.fit(X_train, y_train)

        best_score = gs.best_score_
        best_params = gs.best_params_
        best_model = gs.best_estimator_

        print(f"  Best CV {scoring.upper()}: {best_score:.4f}")
        print(f"  Best params: {best_params}")

        # Serialise
        pkl_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(best_model, pkl_path)
        print(f"  Saved -> {pkl_path}")

        results.append({
            "model": name,
            f"best_cv_{scoring}": round(best_score, 4),
            "best_params": json.dumps(best_params),
        })

    results_df = pd.DataFrame(results).sort_values(f"best_cv_{scoring}", ascending=False)
    return results_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train aircraft fault detection models")
    parser.add_argument("--train", default="data/processed/train_clean.csv")
    parser.add_argument("--target", default="fault")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", default="f1")
    args = parser.parse_args()

    print(f"Loading training data from {args.train} ...")
    df = pd.read_csv(args.train)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    results = train_all_models(
        X_train=X,
        y_train=y,
        models_dir=args.models_dir,
        cv_folds=args.cv,
        scoring=args.scoring,
    )

    print("\n=== FINAL COMPARISON TABLE ===")
    print(results.to_string(index=False))

    results.to_csv(os.path.join(args.models_dir, "cv_results.csv"), index=False)
    print(f"\nResults saved -> {args.models_dir}/cv_results.csv")


if __name__ == "__main__":
    main()
