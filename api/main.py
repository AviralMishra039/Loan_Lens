"""
LoanLens FastAPI — Credit Risk Scoring API

Endpoints:
  POST /predict         → risk score + SHAP explanation
  GET  /health          → service health check
  GET  /feature-importance → global SHAP bar data (for dashboard)
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import ApplicantInput, PredictionResponse, FeatureImportanceResponse
from model_loader import load_artifacts

app = FastAPI(
    title="LoanLens API",
    description="Credit risk scoring with XGBoost + SHAP explainability",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts at startup
MODEL_DIR = os.getenv("MODEL_DIR", "../models")
artifacts = {}


@app.on_event("startup")
def startup():
    global artifacts
    artifacts = load_artifacts(MODEL_DIR)
    print("✓ Model, scaler, SHAP explainer loaded")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in artifacts,
        "version": "1.0.0",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(applicant: ApplicantInput):
    try:
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        explainer = artifacts["explainer"]
        feature_cols = artifacts["feature_cols"]

        # Build raw DataFrame from input
        raw = pd.DataFrame([applicant.dict()])

        # Rename fields to match training column names
        raw = raw.rename(columns={
            "revolving_utilization": "RevolvingUtilizationOfUnsecuredLines",
            "age": "age",
            "past_due_30_59": "NumberOfTime30-59DaysPastDueNotWorse",
            "debt_ratio": "DebtRatio",
            "monthly_income": "MonthlyIncome",
            "open_credit_lines": "NumberOfOpenCreditLinesAndLoans",
            "times_90_days_late": "NumberOfTimes90DaysLate",
            "real_estate_loans": "NumberRealEstateLoansOrLines",
            "past_due_60_89": "NumberOfTime60-89DaysPastDueNotWorse",
            "dependents": "NumberOfDependents",
        })

        # Engineer features (same logic as training)
        raw["total_late_payments"] = (
            raw["NumberOfTime30-59DaysPastDueNotWorse"]
            + raw["NumberOfTimes90DaysLate"]
            + raw["NumberOfTime60-89DaysPastDueNotWorse"]
        )
        raw["monthly_debt"] = raw["DebtRatio"] * raw["MonthlyIncome"]
        raw["income_per_dependent"] = raw["MonthlyIncome"] / (raw["NumberOfDependents"] + 1)
        raw["log_income"] = np.log1p(raw["MonthlyIncome"])
        raw["income_was_missing"] = 0

        X = raw[feature_cols]
        X_scaled = scaler.transform(X)
        X_df = pd.DataFrame(X_scaled, columns=feature_cols)

        # Prediction
        prob = float(model.predict_proba(X_df)[0][1])
        risk_band = _risk_band(prob)

        # SHAP values for this prediction
        shap_vals = explainer(X_df)
        values = shap_vals.values[0]
        base_value = float(shap_vals.base_values[0])

        top_factors = sorted(
            [
                {
                    "feature": feat,
                    "impact": round(float(val), 4),
                    "direction": "increases risk" if val > 0 else "decreases risk",
                }
                for feat, val in zip(feature_cols, values)
            ],
            key=lambda x: abs(x["impact"]),
            reverse=True,
        )[:6]

        return PredictionResponse(
            probability_of_default=round(prob, 4),
            risk_band=risk_band,
            base_value=round(base_value, 4),
            top_factors=top_factors,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance", response_model=FeatureImportanceResponse)
def feature_importance():
    """Return global mean |SHAP| values for dashboard bar chart."""
    if "global_importance" not in artifacts:
        raise HTTPException(status_code=503, detail="Global importance not computed yet")
    return FeatureImportanceResponse(
        features=artifacts["global_importance"]["features"],
        importances=artifacts["global_importance"]["importances"],
    )


def _risk_band(prob: float) -> str:
    if prob < 0.10:
        return "Very Low"
    elif prob < 0.20:
        return "Low"
    elif prob < 0.35:
        return "Medium"
    elif prob < 0.55:
        return "High"
    else:
        return "Very High"
