from pydantic import BaseModel, Field
from typing import List


class ApplicantInput(BaseModel):
    revolving_utilization: float = Field(
        ..., ge=0, le=10, description="Revolving credit utilization (0–1 typical)"
    )
    age: int = Field(..., ge=18, le=110, description="Applicant age in years")
    past_due_30_59: int = Field(..., ge=0, description="Times 30–59 days past due")
    debt_ratio: float = Field(..., ge=0, description="Monthly debt / monthly income")
    monthly_income: float = Field(..., ge=0, description="Monthly income in USD")
    open_credit_lines: int = Field(..., ge=0, description="Number of open credit lines")
    times_90_days_late: int = Field(..., ge=0, description="Times 90+ days past due")
    real_estate_loans: int = Field(..., ge=0, description="Number of real estate loans")
    past_due_60_89: int = Field(..., ge=0, description="Times 60–89 days past due")
    dependents: int = Field(..., ge=0, description="Number of dependents")

    model_config = {
        "json_schema_extra": {
            "example": {
                "revolving_utilization": 0.35,
                "age": 45,
                "past_due_30_59": 0,
                "debt_ratio": 0.25,
                "monthly_income": 5000,
                "open_credit_lines": 8,
                "times_90_days_late": 0,
                "real_estate_loans": 1,
                "past_due_60_89": 0,
                "dependents": 1,
            }
        }
    }


class FactorDetail(BaseModel):
    feature: str
    impact: float
    direction: str


class PredictionResponse(BaseModel):
    probability_of_default: float
    risk_band: str
    base_value: float
    top_factors: List[FactorDetail]


class FeatureImportanceResponse(BaseModel):
    features: List[str]
    importances: List[float]
