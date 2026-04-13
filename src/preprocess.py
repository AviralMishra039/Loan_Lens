import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

FEATURE_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

TARGET_COL = "SeriousDlqin2yrs"


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fill missing values with median (robust to outliers)
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Flag that income was missing (informative missingness)
    df["income_was_missing"] = df["MonthlyIncome"].isna().astype(int)

    # Clip extreme outliers (top 1%)
    df["RevolvingUtilizationOfUnsecuredLines"] = df[
        "RevolvingUtilizationOfUnsecuredLines"
    ].clip(upper=1.0)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=df["DebtRatio"].quantile(0.99))

    # New engineered features
    df["total_late_payments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
    )

    df["monthly_debt"] = df["DebtRatio"] * df["MonthlyIncome"]

    df["income_per_dependent"] = df["MonthlyIncome"] / (
        df["NumberOfDependents"] + 1
    )

    df["log_income"] = np.log1p(df["MonthlyIncome"])

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the final list of feature columns used in training."""
    base = FEATURE_COLS.copy()
    extra = [
        "total_late_payments",
        "monthly_debt",
        "income_per_dependent",
        "log_income",
        "income_was_missing",
    ]
    return [c for c in base + extra if c in df.columns]


def preprocess(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = False,
    save_dir: str = None,
):
    df = engineer_features(df)
    feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if save_dir:
            joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(feature_cols, os.path.join(save_dir, "feature_cols.pkl"))
            print(f"Saved scaler and feature list to {save_dir}")
    elif scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
    return X_scaled, y, scaler, feature_cols
