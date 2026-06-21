# Heart Disease Predictor

A compact Streamlit app that predicts the likelihood of heart disease from clinical inputs using a pruned Decision Tree model.

Live demo: https://heart-disease-predictor-bychandu.streamlit.app/

## Quick start

Prerequisites:
- Python 3.8+
- pip

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
streamlit run app.py
# If your Streamlit script has a different name, replace `app.py` with that filename.
```

Or use the hosted app at the link above.

## How to use

1. Open the app (local or hosted).
2. Enter patient data in the input sections (age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar, resting ECG, max heart rate, exercise-induced angina, ST depression, slope of ST segment, number of major vessels, thalassemia).
3. Click the Predict button.
4. The app returns a risk label (High Risk / Low Risk) and a confidence score (0–100%).

## Notes

- Model: Decision Tree classifier (cost-complexity pruned).
- Preprocessing: median imputation for missing values and one-hot encoding for categorical features.
- Model is serialized with joblib for consistent predictions.

That's it — minimal, focused, and ready to use.