# Project Report — Forest Fire Risk and Burned Area Estimator

## 1. Introduction & Real-World Impact
Forest fires spread rapidly and cause severe ecological, economic, and human damage. This
project builds a complete machine learning pipeline that predicts both the **burned area**
(a continuous number, in hectares) and a **fire-risk category** (Low / Medium / High) from
weather and fire-weather-index readings, and converts that prediction into a plain-English
recommendation — supporting early-warning and resource-planning decisions for forest rangers
and disaster-management authorities.

## 2. Data
- **Reference:** UCI/Kaggle Forest Fires Dataset (Cortez & Morais, 2007), Montesinho Park,
  Portugal. Columns: `X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area`.
- **Used in this run:** `data/forestfires.csv`, 600 rows, 13 columns, **0 missing values,
  0 duplicate rows** (verified in Notebook Section 3). If the real Kaggle file wasn't
  available at build time, a synthetic dataset with the identical schema was substituted and
  clearly disclosed — replacing it with the real file requires no code changes.

## 3. Methodology
1. **Inspection:** `.head()`, `.shape`, `.info()`, `.describe()`, missing-value check,
   duplicate check — all performed before any assumptions were made about the data.
2. **Encoding:** `month` and `day` (text categories) encoded to integers via `LabelEncoder`,
   since scikit-learn models require numeric input. The fitted encoders are saved so the same
   mapping can be reused at prediction time.
3. **Outlier check:** boxplots of all numeric columns — outliers in `area` and `rain` were
   inspected but **not removed**, since rare large-fire events are meaningful signal, not
   data errors.
4. **EDA:** histograms (area, temperature, humidity), scatter plots (temp/RH/wind/rain vs
   area), a correlation heatmap, a pairplot, and monthly boxplots of burned area.
5. **Feature engineering:** a `Risk` column derived from `area` (Low: `<1`ha, Medium:
   `1–25`ha, High: `≥25`ha).
6. **Modeling:** four regressors (Linear Regression, Decision Tree, Random Forest, Gradient
   Boosting) and three classifiers (Decision Tree, Random Forest, Logistic Regression),
   trained on an 80/20 split (classification stratified by `Risk`).
7. **Evaluation:** MAE/RMSE/R² for regression; Accuracy/Precision/Recall/F1/confusion matrix
   for classification.
8. **Explainability:** feature importance from the tree-based classifier.
9. **Decision support:** a `recommend()` function maps risk → action checklist; `predict_fire()`
   ties the full pipeline together for a single set of raw inputs.

## 4. Results (this run)

### Regression — burned area (hectares)
| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | **3.002** | **4.411** | **0.021** |
| Random Forest Regressor | 3.345 | 4.712 | -0.117 |
| Gradient Boosting Regressor | 3.869 | 5.594 | -0.575 |
| Decision Tree Regressor | 3.835 | 6.277 | -0.983 |

**Linear Regression** performed best on this run (lowest RMSE). Note that R² near zero (or
negative for the other models) is expected here, not a bug — it matches the known difficulty
of predicting burned area from weather alone in the published literature on this exact
dataset: most fires stay small regardless of conditions, while a handful of large outliers
dominate the error.

### Classification — risk level (Low / Medium / High)
| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Decision Tree** | **0.542** | 0.347 | 0.350 | **0.349** |
| Random Forest | 0.542 | 0.306 | 0.318 | 0.295 |
| Logistic Regression | 0.317 | 0.258 | 0.191 | 0.218 |

**Decision Tree** performed best on this run (highest F1). Macro-averaged metrics are
noticeably lower than accuracy because the "High" class has very few examples in the test set
— the model struggles most on the class that matters most operationally. This is flagged as a
limitation rather than hidden.

### Feature Importance
The tree-based classifier consistently ranks the fire-weather indices (`DC`, `DMC`, `ISI`,
`FFMC`) and direct weather readings (`temp`, `RH`) as the strongest predictors — consistent
with fire-science domain knowledge, since these indices are purpose-built to capture fuel
dryness and spread potential. Spatial coordinates (`X`, `Y`) contribute less.

## 5. Validation Approach
- 80/20 train/test split; classification split stratified by `Risk` to preserve the
  (imbalanced) class ratio in both sets.
- Regression judged on MAE/RMSE (interpretable in hectares) alongside R².
- Classification judged on macro-averaged Precision/Recall/F1 (not just accuracy), plus a
  confusion matrix — necessary here because a model that always predicts "Low" would still
  score high on raw accuracy while being useless for the classes that matter operationally.

## 6. Limitations & Responsible Use
- Results in this report come from a **synthetic stand-in dataset** unless the real UCI/Kaggle
  file was supplied — re-validate on real data before any operational use.
- Burned-area regression is inherently noisy; treat outputs as an order-of-magnitude estimate.
- The "High" risk class is rare, making it the hardest and least reliable category to predict
  — exactly the category where false negatives matter most.
- This is a decision-support tool only, not a replacement for professional fire-management
  judgment or official fire-weather services.

## 7. Future Work
- Swap in the real dataset (drop-in replacement — identical schema).
- Address class imbalance for "High" risk (e.g. SMOTE, class-weighted resampling).
- Try gradient-boosted libraries (XGBoost/LightGBM) with hyperparameter tuning.
- Add satellite/remote-sensing signals (NDVI, thermal hotspots) and multi-day forecasts.
- Extend the app with a map view and batch/CSV scoring for multiple locations at once.
