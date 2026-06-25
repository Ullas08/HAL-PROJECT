"""
src/evaluate.py
===============
Model evaluation and visualisation.
Computes all metrics, plots ROC curves, confusion matrices,
precision-recall curves, and feature importance charts.
Selects the champion model (F1 >= 0.90 target).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
    f1_score,
)

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

PALETTE = ["#4A90D9", "#E85D75", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
plt.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor": "#1A1D27",
    "axes.edgecolor": "#3A3F5C",
    "text.color": "#E2E8F0",
    "axes.labelcolor": "#E2E8F0",
    "xtick.color": "#A0AEC0",
    "ytick.color": "#A0AEC0",
    "grid.color": "#2D3748",
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})


# ---------------------------------------------------------------------------
# 1. Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
    """Return dict of all metrics for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics = {
        "model": model_name,
        "accuracy": round(report["accuracy"], 4),
        "precision": round(report["1"]["precision"], 4),
        "recall": round(report["1"]["recall"], 4),
        "f1": round(report["1"]["f1-score"], 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
    }
    return metrics


def evaluate_all_models(
    models_dir: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: list = None,
) -> pd.DataFrame:
    """Evaluate all serialised models and return comparison DataFrame."""
    if model_names is None:
        model_names = [
            "logistic_regression", "decision_tree", "random_forest",
            "svm", "knn", "xgboost"
        ]

    all_metrics = []
    loaded_models = {}

    for name in model_names:
        pkl_path = os.path.join(models_dir, f"{name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"[warn] {pkl_path} not found, skipping.")
            continue
        model = joblib.load(pkl_path)
        loaded_models[name] = model
        m = evaluate_model(model, X_test, y_test, name)
        all_metrics.append(m)
        print(f"[eval] {name:<25} F1={m['f1']:.4f}  AUC={m['roc_auc']:.4f}")

    df = pd.DataFrame(all_metrics).sort_values("f1", ascending=False).reset_index(drop=True)
    return df, loaded_models


# ---------------------------------------------------------------------------
# 2. ROC Curves (all models on one chart)
# ---------------------------------------------------------------------------

def plot_roc_curves(loaded_models: dict, X_test, y_test, save_path: str = "reports/roc_curves.png"):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#1A1D27")
    fig.patch.set_facecolor("#0F1117")

    for (name, model), color in zip(loaded_models.items(), PALETTE):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold", color="#E2E8F0")
    ax.legend(loc="lower right", fontsize=9, facecolor="#1A1D27", edgecolor="#3A3F5C")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] ROC curves -> {save_path}")


# ---------------------------------------------------------------------------
# 3. Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    loaded_models: dict, X_test, y_test,
    save_dir: str = "reports"
):
    os.makedirs(save_dir, exist_ok=True)
    n = len(loaded_models)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.patch.set_facecolor("#0F1117")
    axes = axes.flatten() if n > 1 else [axes]

    for ax, (name, model) in zip(axes, loaded_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap="Blues",
            linewidths=0.5, linecolor="#0F1117",
            annot_kws={"size": 13, "weight": "bold"},
            xticklabels=["NORMAL", "FAULT"],
            yticklabels=["NORMAL", "FAULT"],
        )
        ax.set_title(name.replace("_", " ").title(), color="#E2E8F0", fontsize=11)
        ax.set_xlabel("Predicted", color="#A0AEC0")
        ax.set_ylabel("Actual", color="#A0AEC0")
        ax.set_facecolor("#1A1D27")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Confusion Matrices — All Models", fontsize=15,
                 fontweight="bold", color="#E2E8F0", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Confusion matrices -> {path}")


# ---------------------------------------------------------------------------
# 4. Precision-Recall Curves (top 2 models)
# ---------------------------------------------------------------------------

def plot_pr_curves(
    loaded_models: dict, X_test, y_test,
    top_n: int = 2,
    save_path: str = "reports/pr_curve.png"
):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    for (name, model), color in zip(list(loaded_models.items())[:top_n], PALETTE):
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})",
                    color=color, lw=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves (Top Models)", fontsize=14,
                 fontweight="bold", color="#E2E8F0")
    ax.legend(fontsize=10, facecolor="#1A1D27", edgecolor="#3A3F5C")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] PR curves -> {save_path}")


# ---------------------------------------------------------------------------
# 5. Feature Importance (best tree-based model)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    model, feature_names: list,
    model_name: str = "model",
    save_path: str = "reports/feature_importance.png"
):
    if not hasattr(model, "feature_importances_"):
        print(f"[warn] {model_name} has no feature_importances_ attribute. Skipping.")
        return

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(5, len(importances) * 0.35)))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D27")

    bars = ax.barh(importances.index, importances.values,
                   color=PALETTE[0], edgecolor="#0F1117", height=0.65)

    # Gradient colouring
    for bar, val in zip(bars, importances.values):
        alpha = 0.4 + 0.6 * (val / importances.max())
        bar.set_alpha(alpha)

    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Feature Importance — {model_name.replace('_', ' ').title()}",
                 fontsize=14, fontweight="bold", color="#E2E8F0")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[plot] Feature importance -> {save_path}")


# ---------------------------------------------------------------------------
# 6. Select champion model
# ---------------------------------------------------------------------------

def select_champion(
    metrics_df: pd.DataFrame,
    loaded_models: dict,
    models_dir: str = "models",
    f1_target: float = 0.90,
) -> str:
    """
    Pick the model with the highest F1 on the test set.
    Warn if it doesn't reach the F1 target.
    Saves it as models/champion_model.pkl.
    """
    best_row = metrics_df.iloc[0]
    champion_name = best_row["model"]
    champion_f1 = best_row["f1"]

    if champion_f1 < f1_target:
        print(f"[warn] Champion F1={champion_f1:.4f} is below target {f1_target}. "
              f"Consider further tuning.")
    else:
        print(f"[champion] {champion_name} meets target! F1={champion_f1:.4f} >= {f1_target}")

    champion = loaded_models[champion_name]
    save_path = os.path.join(models_dir, "champion_model.pkl")
    joblib.dump(champion, save_path)
    print(f"[save] Champion model saved -> {save_path}")

    # Save champion metadata
    meta = {"name": champion_name, "f1": champion_f1,
            "roc_auc": best_row["roc_auc"]}
    with open(os.path.join(models_dir, "champion_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return champion_name


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate aircraft fault detection models")
    parser.add_argument("--test", default="data/processed/test_clean.csv")
    parser.add_argument("--target", default="fault")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--reports_dir", default="reports")
    args = parser.parse_args()

    df = pd.read_csv(args.test)
    X_test = df.drop(columns=[args.target])
    y_test = df[args.target]

    metrics_df, loaded = evaluate_all_models(args.models_dir, X_test, y_test)
    print("\n=== METRICS TABLE ===")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(args.reports_dir, "metrics_table.csv"), index=False)

    plot_roc_curves(loaded, X_test, y_test, os.path.join(args.reports_dir, "roc_curves.png"))
    plot_confusion_matrices(loaded, X_test, y_test, args.reports_dir)
    plot_pr_curves(loaded, X_test, y_test, save_path=os.path.join(args.reports_dir, "pr_curve.png"))

    # Feature importance for best tree model
    tree_models = {k: v for k, v in loaded.items()
                   if k in ["random_forest", "xgboost", "decision_tree"]}
    if tree_models:
        best_tree_name = metrics_df[metrics_df["model"].isin(tree_models)].iloc[0]["model"]
        plot_feature_importance(
            tree_models[best_tree_name],
            list(X_test.columns),
            model_name=best_tree_name,
            save_path=os.path.join(args.reports_dir, "feature_importance.png"),
        )

    champion = select_champion(metrics_df, loaded, args.models_dir)
    print(f"\nChampion model: {champion}")
