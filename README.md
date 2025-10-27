# Diabetes Detector — ML + ACT-R (Teaching Demo)

This project reproduces the **Diabetes Detection** idea from your slides: a side‑by‑side comparison between a **Random Forest** model and an **ACT‑R–inspired** memory recall predictor.

## What you get
- Streamlit app (`app.py`)
- Random Forest classifier with feature importance chart
- ACT‑R–style recall model (`actr_memory.py`): finds similar past cases and predicts diabetic if ≥ 60% of neighbors had diabetes
- Bundled demo CSV (`demo_diabetes.csv`) so it runs out of the box
- Upload your own CSV with Pima‑like columns if you have one

## Quick start

```bash
# 1) create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) install dependencies
pip install -r requirements.txt

# 3) run the app
streamlit run app.py
```

Open the local URL Streamlit prints in your terminal.

## Data
If you **don't upload** a CSV, the app uses `demo_diabetes.csv` (synthetic demo).  
If you **do upload** a CSV, it should include these columns:

```
Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
BMI, DiabetesPedigreeFunction, Age, Outcome
```

`Outcome` must be 0/1 (non‑diabetic / diabetic).

## How predictions work

- **Random Forest (ML):** trained on the dataset (75/25 split), predicts a probability of diabetes and shows **feature importances**.

- **ACT‑R–style recall:** computes similarity between your inputs and all past cases, selects top‑k neighbors (default 25).  
  If at least **60%** of those neighbors are diabetic → predicts **diabetic**.

These two perspectives help with **transparency** and **trust**: model‑based importance vs. human‑like case recall.

## Notes
- This is **not medical advice** and intended for learning only.
- Accuracy depends on data quality and size. The bundled demo is synthetic for illustration.

## Troubleshooting
- If Streamlit doesn't open in the browser, copy‑paste the URL from the terminal.
- If install errors occur, try upgrading pip:
  ```
  python -m pip install --upgrade pip
  ```

Enjoy!
