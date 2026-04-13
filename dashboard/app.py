"""
LoanLens Dashboard — Streamlit UI

Connects to the FastAPI backend and shows:
  1. Live applicant scoring form
  2. SHAP waterfall chart for the prediction
  3. Global feature importance bar chart
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LoanLens — Credit Risk Scorer",
    page_icon="🔍",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔍 LoanLens")
st.caption("XGBoost credit risk scoring with SHAP explainability")
st.divider()

# ── Sidebar: Applicant form ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Applicant Details")
    revolving_utilization = st.slider("Credit utilization (%)", 0.0, 1.0, 0.35, 0.01)
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    monthly_income = st.number_input("Monthly income ($)", min_value=0, value=5000, step=100)
    debt_ratio = st.slider("Debt ratio", 0.0, 3.0, 0.25, 0.01)
    dependents = st.number_input("Number of dependents", min_value=0, max_value=20, value=1)
    open_credit_lines = st.number_input("Open credit lines", min_value=0, max_value=50, value=8)
    real_estate_loans = st.number_input("Real estate loans", min_value=0, max_value=20, value=1)

    st.subheader("Payment history")
    past_due_30_59 = st.number_input("Times 30–59 days late", 0, 20, 0)
    past_due_60_89 = st.number_input("Times 60–89 days late", 0, 20, 0)
    times_90_days_late = st.number_input("Times 90+ days late", 0, 20, 0)

    score_btn = st.button("🔎 Score Applicant", use_container_width=True, type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

if score_btn:
    payload = {
        "revolving_utilization": revolving_utilization,
        "age": int(age),
        "past_due_30_59": int(past_due_30_59),
        "debt_ratio": debt_ratio,
        "monthly_income": float(monthly_income),
        "open_credit_lines": int(open_credit_lines),
        "times_90_days_late": int(times_90_days_late),
        "real_estate_loans": int(real_estate_loans),
        "past_due_60_89": int(past_due_60_89),
        "dependents": int(dependents),
    }

    with st.spinner("Scoring..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    prob = result["probability_of_default"]
    band = result["risk_band"]
    factors = result["top_factors"]

    # ── Score card ────────────────────────────────────────────────────────────
    with col1:
        band_colors = {
            "Very Low": "🟢", "Low": "🟢",
            "Medium": "🟡",
            "High": "🔴", "Very High": "🔴",
        }
        st.metric("Default probability", f"{prob:.1%}")
        st.metric("Risk band", f"{band_colors.get(band, '')} {band}")
        st.progress(min(prob, 1.0))

        st.subheader("Top risk factors")
        for f in factors:
            direction_icon = "↑" if f["direction"] == "increases risk" else "↓"
            color = "red" if f["direction"] == "increases risk" else "green"
            st.markdown(
                f"**{f['feature']}** &nbsp; "
                f":{color}[{direction_icon} {abs(f['impact']):.4f}]"
            )

    # ── SHAP waterfall ────────────────────────────────────────────────────────
    with col2:
        st.subheader("SHAP explanation — why this score?")

        feat_names = [f["feature"] for f in factors]
        impacts = [f["impact"] for f in factors]
        colors = ["#D85A30" if v > 0 else "#1D9E75" for v in impacts]

        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["relative"] * len(impacts) + ["total"],
            y=feat_names + ["Final score"],
            x=impacts + [sum(impacts)],
            connector={"line": {"color": "rgba(100,100,100,0.3)"}},
            decreasing={"marker": {"color": "#1D9E75"}},
            increasing={"marker": {"color": "#D85A30"}},
            totals={"marker": {"color": "#7F77DD"}},
        ))
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="SHAP value (impact on risk score)",
            font=dict(size=13),
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    with col1:
        st.info("Fill in the applicant details in the sidebar and click **Score Applicant**.")
    with col2:
        # ── Global feature importance (always visible) ─────────────────────
        st.subheader("Global feature importance")
        try:
            fi_resp = requests.get(f"{API_URL}/feature-importance", timeout=5)
            fi_resp.raise_for_status()
            fi = fi_resp.json()

            fig = px.bar(
                x=fi["importances"][:10],
                y=fi["features"][:10],
                orientation="h",
                labels={"x": "Mean gain", "y": "Feature"},
                color=fi["importances"][:10],
                color_continuous_scale=["#1D9E75", "#EF9F27", "#D85A30"],
            )
            fig.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=30, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.warning("Start the API server to see feature importance.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("LoanLens v1.0 · XGBoost + SHAP · Built with FastAPI & Streamlit")
