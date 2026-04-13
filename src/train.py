"""
train.py — LoanLens training pipeline

Usage:
    python src/train.py --data data/cs-training.csv --output models/

Downloads dataset from Kaggle if not present. Trains XGBoost with
cross-validation, handles class imbalance, saves model + SHAP explainer.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE

from preprocess import load_data, preprocess


def compute_class_weight(y: pd.Series) -> float:
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    ratio = neg / pos
    print(f"Class ratio (neg/pos): {ratio:.1f}x — using as scale_pos_weight")
    return ratio


def train_model(X_train, y_train, scale_pos_weight: float):
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    print(f"Best iteration: {model.best_iteration}")
    return model


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)

    print(f"\n{'='*40}")
    print(f"  AUC-ROC:              {auc:.4f}")
    print(f"  Average Precision:    {ap:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, preds, target_names=["No Default", "Default"]))

    return auc, ap


def generate_shap(model, X_train_sample, feature_cols, output_dir):
    print("\nGenerating SHAP values...")
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed (500 rows)
    sample = X_train_sample.sample(min(500, len(X_train_sample)), random_state=42)
    shap_values = explainer(sample)

    # --- Beeswarm plot ---
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("Global Feature Impact (SHAP Beeswarm)", fontsize=14, pad=12)
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved beeswarm plot → {beeswarm_path}")

    # --- Bar summary ---
    fig, ax = plt.subplots(figsize=(9, 6))
    shap.plots.bar(shap_values, max_display=12, show=False)
    plt.title("Mean |SHAP| Feature Importance", fontsize=14, pad=12)
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "shap_bar.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar plot → {bar_path}")

    # Save explainer
    explainer_path = os.path.join(output_dir, "shap_explainer.pkl")
    joblib.dump(explainer, explainer_path)
    print(f"Saved SHAP explainer → {explainer_path}")

    return explainer


def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load & preprocess ---
    df = load_data(data_path)
    X, y, scaler, feature_cols = preprocess(
        df, fit_scaler=True, save_dir=output_dir
    )

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Default rate — train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")

    # --- Class imbalance: SMOTE on training set only ---
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — train size: {len(X_resampled)}, balance: {y_resampled.mean():.2%}")

    scale_pos_weight = compute_class_weight(y_train)

    # --- Train ---
    print("\nTraining XGBoost...")
    model = train_model(X_resampled, y_resampled, scale_pos_weight)

    # --- Evaluate ---
    evaluate(model, X_test, y_test)

    # --- Save model ---
    model_path = os.path.join(output_dir, "model.pkl")
    joblib.dump(model, model_path)
    print(f"\nSaved model → {model_path}")

    # --- SHAP ---
    generate_shap(model, X_train, feature_cols, output_dir)

    print("\nTraining complete. All artifacts saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LoanLens credit risk model")
    parser.add_argument("--data", default="data/cs-training.csv", help="Path to training CSV")
    parser.add_argument("--output", default="models/", help="Directory to save artifacts")
    args = parser.parse_args()
    main(args.data, args.output)
