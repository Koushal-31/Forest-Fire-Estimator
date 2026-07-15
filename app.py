"""
app.py — Forest Fire Risk and Burned Area Estimator (Streamlit App)
----------------------------------------------------------------------
Loads the models trained and saved by the notebook (Section 12) and provides
an interactive UI: sidebar inputs, a Predict button, burned-area prediction,
fire-risk prediction, recommendations, and a feature-importance chart.

Run with:
    streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(page_title="Forest Fire Risk Estimator", page_icon="🔥", layout="wide")


# ---------------------------------------------------------------------------
# Load model artifacts (cached so they only load once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    reg_model = joblib.load(os.path.join(MODEL_DIR, "regression_model.joblib"))
    clf_model = joblib.load(os.path.join(MODEL_DIR, "classification_model.joblib"))
    le_month = joblib.load(os.path.join(MODEL_DIR, "label_encoder_month.joblib"))
    le_day = joblib.load(os.path.join(MODEL_DIR, "label_encoder_day.joblib"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    return reg_model, clf_model, le_month, le_day, feature_cols


# ---------------------------------------------------------------------------
# Recommendation engine (same rules as the notebook)
# ---------------------------------------------------------------------------
def recommend(risk):
    risk = str(risk).strip().title()
    if risk == "Low":
        return ["Normal monitoring", "No immediate danger", "Continue routine observations"]
    elif risk == "Medium":
        return ["Increase monitoring frequency", "Alert forest staff", "Prepare emergency response teams"]
    elif risk == "High":
        return ["Avoid open burning", "Restrict forest access", "Increase patrols",
                "Alert relevant authorities", "Deploy firefighting resources"]
    return ["Unknown risk category — unable to generate recommendation."]


RISK_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
st.title("🔥 Forest Fire Risk and Burned Area Estimator")
st.caption("Enter weather and fire-index readings in the sidebar, then click Predict.")

try:
    reg_model, clf_model, le_month, le_day, feature_cols = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found in `models/`. Please run the notebook "
        "(`Forest_Fire_Complete_Notebook.ipynb`) first — Section 12 saves the "
        "required `.joblib` files into that folder."
    )
    st.stop()

# ---- Sidebar inputs ----
st.sidebar.header("Input Conditions")

X_coord = st.sidebar.slider("X coordinate (1-9)", 1, 9, 5)
Y_coord = st.sidebar.slider("Y coordinate (1-9)", 1, 9, 5)
month = st.sidebar.selectbox("Month", list(le_month.classes_),
                              index=list(le_month.classes_).index("aug") if "aug" in le_month.classes_ else 0)
day = st.sidebar.selectbox("Day of week", list(le_day.classes_),
                            index=list(le_day.classes_).index("sat") if "sat" in le_day.classes_ else 0)

st.sidebar.markdown("**Fire Weather Index components**")
FFMC = st.sidebar.slider("FFMC (Fine Fuel Moisture Code)", 20.0, 96.5, 90.0)
DMC = st.sidebar.slider("DMC (Duff Moisture Code)", 1.0, 300.0, 100.0)
DC = st.sidebar.slider("DC (Drought Code)", 5.0, 860.0, 400.0)
ISI = st.sidebar.slider("ISI (Initial Spread Index)", 0.0, 40.0, 9.0)

st.sidebar.markdown("**Weather readings**")
temp = st.sidebar.slider("Temperature (°C)", -5.0, 45.0, 25.0)
RH = st.sidebar.slider("Relative Humidity (%)", 0.0, 100.0, 40.0)
wind = st.sidebar.slider("Wind speed (km/h)", 0.0, 30.0, 6.0)
rain = st.sidebar.slider("Rain (mm/m²)", 0.0, 10.0, 0.0)

predict_clicked = st.sidebar.button("🔍 Predict", type="primary", use_container_width=True)

# ---- Main panel ----
col1, col2 = st.columns([1.1, 1])

if predict_clicked:
    try:
        month_enc = le_month.transform([month])[0]
        day_enc = le_day.transform([day])[0]
    except ValueError as e:
        st.error(f"Encoding error: {e}")
        st.stop()

    input_row = pd.DataFrame([{
        "X": X_coord, "Y": Y_coord, "month_encoded": month_enc, "day_encoded": day_enc,
        "FFMC": FFMC, "DMC": DMC, "DC": DC, "ISI": ISI,
        "temp": temp, "RH": RH, "wind": wind, "rain": rain,
    }])[feature_cols]

    predicted_area = max(reg_model.predict(input_row)[0], 0)
    predicted_risk = clf_model.predict(input_row)[0]
    recommendations = recommend(predicted_risk)
    color = RISK_COLORS.get(predicted_risk, "#95a5a6")

    with col1:
        st.subheader("Prediction Result")
        st.markdown(
            f"<div style='padding:18px;border-radius:10px;background-color:{color}22;"
            f"border:2px solid {color};'>"
            f"<h2 style='color:{color};margin:0;'>Risk Level: {predicted_risk}</h2>"
            f"<p style='font-size:18px;margin:8px 0 0 0;'>Predicted burned area: "
            f"<b>{predicted_area:.2f} hectares</b></p></div>",
            unsafe_allow_html=True,
        )

        st.markdown("### Recommendations")
        for action in recommendations:
            st.markdown(f"- {action}")

        st.caption(
            "⚠️ Decision-support estimate only — confirm with official fire/weather "
            "authorities before taking action."
        )

    with col2:
        st.subheader("Feature Importance")
        if hasattr(clf_model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": clf_model.feature_importances_,
            }).sort_values("Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2C5F2D")
            ax.set_xlabel("Importance")
            ax.set_title("What drives this model's predictions")
            st.pyplot(fig)
        else:
            st.info(
                f"The selected classifier ({type(clf_model).__name__}) does not expose "
                "feature importances directly."
            )
else:
    with col1:
        st.info("Set your conditions in the sidebar and click **Predict** to see results.")
    with col2:
        if hasattr(clf_model, "feature_importances_"):
            st.subheader("Feature Importance (default view)")
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": clf_model.feature_importances_,
            }).sort_values("Importance", ascending=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(importance_df["Feature"], importance_df["Importance"], color="#2C5F2D")
            ax.set_xlabel("Importance")
            st.pyplot(fig)

st.markdown("---")
st.caption(
    "Forest Fire Risk and Burned Area Estimator — decision-support prototype. "
    "Not a substitute for official fire-management judgment."
)
