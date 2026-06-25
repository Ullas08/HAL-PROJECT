"""
src/run_pipeline.py
===================
Master pipeline script — runs all phases sequentially:
  1. Generate / load dataset
  2. Preprocess
  3. Train all 6 models
  4. Evaluate & generate charts
  5. Select champion model

Usage:
    python src/run_pipeline.py [--dataset synthetic|cmapss] [--data_path path/to/file.csv]
"""

import argparse
import os
import sys
import pandas as pd

# Make src importable when called from project root
sys.path.insert(0, os.path.dirname(__file__))

from generate_dataset import generate_synthetic_dataset
from preprocess import load_cmapss, load_csv, run_preprocessing_pipeline
from train import train_all_models
from evaluate import (
    evaluate_all_models,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_pr_curves,
    plot_feature_importance,
    select_champion,
)


def main():
    parser = argparse.ArgumentParser(description="Run the full aircraft fault detection pipeline")
    parser.add_argument(
        "--dataset", choices=["synthetic", "cmapss", "csv"], default="synthetic",
        help="Dataset to use: synthetic (default), cmapss (NASA FD001), or csv (custom path)"
    )
    parser.add_argument("--data_path", default=None, help="Path to CSV file (for --dataset csv)")
    parser.add_argument("--target", default="fault", help="Target column name")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.reports_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)

    print("\n" + "="*70)
    print("  HAL AIRCRAFT FAULT DETECTION — FULL PIPELINE")
    print("="*70)

    # ── PHASE 1: Load Dataset ──────────────────────────────────────────────
    print("\n[PHASE 1] Loading dataset ...")
    if args.dataset == "synthetic":
        df = generate_synthetic_dataset(save_dir="data/raw")
        target_col = "fault"
    elif args.dataset == "cmapss":
        df = load_cmapss(raw_dir="data/raw")
        target_col = "fault"
    elif args.dataset == "csv":
        if not args.data_path:
            print("ERROR: --data_path required when using --dataset csv")
            sys.exit(1)
        df = load_csv(args.data_path)
        target_col = args.target
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Fault class distribution:\n{df[target_col].value_counts().to_string()}")

    # ── PHASE 2: Preprocess ────────────────────────────────────────────────
    print("\n[PHASE 2] Preprocessing ...")
    pipeline_output = run_preprocessing_pipeline(
        df=df,
        target_col=target_col,
        top_n_features=15,
        save_dir=args.processed_dir,
        model_dir=args.models_dir,
    )
    X_train = pipeline_output["X_train"]
    X_test  = pipeline_output["X_test"]
    y_train = pipeline_output["y_train"]
    y_test  = pipeline_output["y_test"]
    feature_names = pipeline_output["feature_names"]

    # ── PHASE 3: Train ─────────────────────────────────────────────────────
    print("\n[PHASE 3] Training all 6 models ...")
    cv_results = train_all_models(
        X_train=X_train,
        y_train=y_train,
        models_dir=args.models_dir,
        cv_folds=args.cv,
    )
    print("\n=== Cross-Validation Results ===")
    print(cv_results.to_string(index=False))

    # ── PHASE 4: Evaluate ──────────────────────────────────────────────────
    print("\n[PHASE 4] Evaluating on test set ...")
    metrics_df, loaded_models = evaluate_all_models(
        models_dir=args.models_dir,
        X_test=X_test,
        y_test=y_test,
    )
    print("\n=== Test-Set Metrics ===")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(args.reports_dir, "metrics_table.csv"), index=False)

    # ── PHASE 4: Visualise ─────────────────────────────────────────────────
    print("\n[PHASE 4] Generating charts ...")
    plot_roc_curves(
        loaded_models, X_test, y_test,
        save_path=os.path.join(args.reports_dir, "roc_curves.png")
    )
    plot_confusion_matrices(loaded_models, X_test, y_test, save_dir=args.reports_dir)
    plot_pr_curves(
        loaded_models, X_test, y_test,
        save_path=os.path.join(args.reports_dir, "pr_curve.png")
    )

    # Feature importance for best tree-based model
    tree_names = ["random_forest", "xgboost", "decision_tree"]
    tree_candidates = metrics_df[metrics_df["model"].isin(tree_names)]
    if not tree_candidates.empty:
        best_tree = tree_candidates.iloc[0]["model"]
        plot_feature_importance(
            loaded_models[best_tree],
            feature_names,
            model_name=best_tree,
            save_path=os.path.join(args.reports_dir, "feature_importance.png"),
        )

    # ── Select Champion ────────────────────────────────────────────────────
    print("\n[PHASE 4] Selecting champion model ...")
    champion = select_champion(metrics_df, loaded_models, models_dir=args.models_dir)

    print("\n" + "="*70)
    print(f"  PIPELINE COMPLETE  |  Champion: {champion.upper()}")
    print("="*70)
    print("\nNext step -> launch the dashboard:")
    print("  streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
