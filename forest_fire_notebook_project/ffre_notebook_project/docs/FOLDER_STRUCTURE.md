# Folder Structure

```text
forest_fire_risk_and_burned_area_estimator/
│
├── data/
│   └── forestfires.csv                  # Dataset (X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain, area)
│
├── notebooks/
│   └── Forest_Fire_Complete_Notebook.ipynb   # Full 14-section pipeline: EDA → models → predictions
│
├── app/
│   ├── app.py                           # Streamlit demo app
│   └── models/                          # Created by running the notebook (Section 12)
│       ├── regression_model.joblib
│       ├── classification_model.joblib
│       ├── label_encoder_month.joblib
│       ├── label_encoder_day.joblib
│       └── feature_columns.joblib
│
├── docs/
│   ├── presentation.md                  # 10-slide presentation content
│   └── project_report.md                # Detailed written report
│
├── requirements.txt                     # Python dependencies
├── .gitignore
└── README.md                            # Start here
```

## Notes
- `app/models/` is created automatically the first time you run the notebook end-to-end — it
  isn't pre-populated in version control (see `.gitignore`), since model files are
  regenerable artifacts, not source content.
- `data/forestfires.csv` should be replaced with the real UCI/Kaggle dataset if/when available;
  no other files need to change since the column schema is identical.
- Run order matters: **notebook first** (creates the model files), **then** the Streamlit app
  (loads those model files). Running the app before the notebook will show a clear error
  message telling you to run the notebook first.
