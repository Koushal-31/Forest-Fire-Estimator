# Presentation: Forest Fire Risk and Burned Area Estimator
*(10-slide content — paste into PowerPoint/Google Slides, one section per slide)*

---

## Slide 1 — Title
**Forest Fire Risk and Burned Area Estimator**
Predicting fire risk and burned area from weather & fuel-moisture data
Team: [Add Names Here]

---

## Slide 2 — The Problem & Real-World Impact
- Forest fires spread fast and cause severe ecological and economic damage.
- Early risk detection lets rangers and fire departments act before a fire starts or spreads.
- Goal: turn raw weather readings into an actionable risk alert, not just a number.

---

## Slide 3 — Dataset & Reference Material
- Reference: UCI/Kaggle Forest Fires Dataset — Montesinho Park, Portugal.
- Columns: X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area.
- 600 rows, no missing values, no duplicates (after cleaning check).
- If the real file wasn't available, a synthetic dataset with the same schema was used —
  clearly disclosed, drop-in replaceable.

---

## Slide 4 — System Workflow
Load & inspect → Clean → Encode (month/day) → EDA → Feature engineering (Risk label) →
Train regression + classification models → Evaluate → Feature importance →
Recommendation engine → Streamlit app

---

## Slide 5 — AI / ML Innovation
**Regression (burned area):** Linear Regression, Decision Tree, Random Forest, Gradient
Boosting — compared on MAE / RMSE / R², best model auto-selected.

**Classification (risk level):** Decision Tree, Random Forest, Logistic Regression —
compared on Accuracy / Precision / Recall / F1, best model auto-selected.

Why two models: a raw hectare number isn't actionable — bucketing into Low/Medium/High and
explaining the top drivers turns a prediction into a decision-support alert.

---

## Slide 6 — Prototype: Streamlit Demo App
- Sidebar inputs for all weather/fire-index readings.
- One-click "Predict" button.
- Shows: predicted burned area, risk level, recommended actions, feature-importance chart.
- [Insert screenshot of the running app here]

---

## Slide 7 — Results & Sample Output
- Insert your actual metrics table here (see notebook Sections 7 & 8 output).
- Top predictive features: fire-weather indices (DC, DMC, ISI, FFMC) and temp/RH.
- Regression performance is modest by design — burned area is a genuinely noisy target
  (matches published research on this dataset).

---

## Slide 8 — Limitations & Responsible Use
- Synthetic training data unless the real dataset is supplied.
- Weather-only signal — doesn't capture fuel load, terrain, or ignition sources.
- "High" risk is a rare class — hardest to predict reliably.
- Not a replacement for professional fire-management judgment.

---

## Slide 9 — Future Improvements
- Retrain on the real UCI/Kaggle dataset.
- Add satellite/remote-sensing features (NDVI, thermal hotspots).
- Try XGBoost/LightGBM with tuning; address class imbalance (e.g. SMOTE).
- Map-based UI, multi-day weather-forecast integration.

---

## Slide 10 — Conclusion
A complete, explainable pipeline — data → models → risk bucket → recommendation — built for
real decision-makers, not just a notebook.
Thank you.
