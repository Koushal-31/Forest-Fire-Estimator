# 🔥 Forest Fire Risk and Burned Area Estimator

## Project Title
Forest Fire Risk and Burned Area Estimator — Complete Notebook + Streamlit App

## Problem Statement
Forest fires spread quickly and cause severe ecological and economic damage. This project
predicts **burned area** (regression) and a **fire-risk category** (Low / Medium / High,
classification) from weather and fire-weather-index readings, and turns that prediction into
a plain-English recommendation for real users — forest rangers, disaster-management officers,
and fire departments.

## Dataset / Reference Source
- **Reference:** UCI/Kaggle Forest Fires Dataset — columns: `X, Y, month, day, FFMC, DMC, DC,
  ISI, temp, RH, wind, rain, area`.
- **This repo:** `data/forestfires.csv`. If the real Kaggle file wasn't available when this was
  built, a synthetic dataset with the identical schema was used instead — this is stated
  explicitly wherever it matters. Drop in the real file (same name/columns) and everything
  re-runs without code changes.

## Tools Used
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Joblib, Streamlit, Jupyter.

## Project Workflow
```
Load & inspect data → Clean (duplicates check) → Encode month/day (LabelEncoder) →
EDA (histograms, scatter plots, heatmap, pairplot, boxplots) → Feature engineering (Risk label) →
Regression models (4) → Classification models (3) → Feature importance →
Recommendation engine → predict_fire() function → Save models → Streamlit app
```

## AI / ML Component
| Task | Models trained | Target |
|---|---|---|
| Regression | Linear Regression, Decision Tree, Random Forest, Gradient Boosting | `area` (hectares) |
| Classification | Decision Tree, Random Forest, Logistic Regression | `Risk` (Low/Medium/High) |

The best model of each type (lowest RMSE for regression, highest F1 for classification) is
selected automatically and saved for use by the app. Feature importance (from the Random
Forest / Decision Tree) explains which weather conditions matter most, and a `recommend()`
function turns the predicted risk into a concrete action checklist.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the notebook top-to-bottom (trains + saves all models)
jupyter notebook notebooks/Forest_Fire_Complete_Notebook.ipynb
# — or, to execute it non-interactively from the command line:
jupyter nbconvert --to notebook --execute --inplace notebooks/Forest_Fire_Complete_Notebook.ipynb

# 3. Launch the Streamlit app (requires step 2 to have run first)
cd app
streamlit run app.py
```

## Folder Structure
See `FOLDER_STRUCTURE.md` for the full annotated layout.

## Demo Screenshots
Add screenshots of the running Streamlit app here before submission.

## Results and Insights
Exact numbers depend on the run (see the notebook's printed comparison tables), but in
general: fire-weather indices (DC, DMC, ISI, FFMC) and direct weather readings (temp, RH) are
the strongest predictors; regression performance is modest by design (burned area is
inherently noisy — this matches published research on the original dataset); the "High" risk
class is rare and hardest to predict reliably.

## Limitations
- Uses a synthetic stand-in dataset unless the real Kaggle CSV is supplied.
- Burned-area predictions are order-of-magnitude estimates, not precise forecasts.
- "High" risk is a minority class — treat positive "High" predictions as a strong signal to
  double-check, and don't over-trust the absence of a "High" prediction either.
- Decision-support only — not a replacement for official fire-management authorities.

## Future Improvements
- Retrain on the real dataset once available.
- Add satellite/remote-sensing features (NDVI, active-fire hotspots).
- Try XGBoost/LightGBM with hyperparameter tuning.
- Address class imbalance with resampling (e.g. SMOTE) for the "High" risk class.
- Add a map-based UI and multi-day forecast integration.

## Team Member Names
_Add your name(s) here._

---
### Responsible-use note
Do not use this tool as the sole basis for evacuation or emergency-response decisions —
always confirm with official fire and disaster-management authorities.
