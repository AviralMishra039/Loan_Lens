import os
import joblib
import numpy as np
import pandas as pd
import shap


def load_artifacts(model_dir: str) -> dict:
    """Load all model artifacts from disk into a dict."""
    required = ["model.pkl", "scaler.pkl", "feature_cols.pkl", "shap_explainer.pkl"]
    for fname in required:
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact: {path}\n"
                "Run: python src/train.py --data data/cs-training.csv --output models/"
            )

    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_cols.pkl"))
    explainer = joblib.load(os.path.join(model_dir, "shap_explainer.pkl"))

    # Pre-compute global feature importance from XGBoost gain
    importance_dict = model.get_booster().get_score(importance_type="gain")
    features = list(importance_dict.keys())
    importances = [importance_dict[f] for f in features]

    # Sort descending
    sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_pairs) if sorted_pairs else ([], [])

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "explainer": explainer,
        "global_importance": {
            "features": list(features),
            "importances": [round(float(i), 2) for i in importances],
        },
    }
