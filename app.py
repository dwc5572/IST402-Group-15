
import os
import io
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

from actr_memory import recall_predict

st.set_page_config(page_title="Diabetes Detector (ML + ACT-R)", page_icon="🩺")

st.title("🩺 Diabetes Detector — ML + ACT-R (Demo)")
st.write("Side‑by‑side predictions: a Random Forest model (ML) and an ACT‑R–inspired memory recall model.")

with st.expander("ℹ️ How this works"):
    st.markdown("""
- **Random Forest**: a standard machine learning model trained on tabular health data.
- **ACT‑R–style recall**: finds past patients most similar to you; if **≥ 60%** of those matches had diabetes, it predicts **diabetic**.
- You can upload a CSV (same columns as Pima dataset) or use the bundled demo data.
    """)

# Data loading
uploaded = st.file_uploader("Upload a CSV (optional). Expected columns like Pima: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    source = "user-uploaded CSV"
else:
    demo_path = os.path.join(os.path.dirname(__file__), "demo_diabetes.csv")
    df = pd.read_csv(demo_path)
    source = "bundled demo dataset"

st.caption(f"Data source: {source}. Rows: {len(df)}")

# Columns and features
label_col = "Outcome"
all_expected = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age",label_col]
missing_cols = [c for c in all_expected if c not in df.columns]
if missing_cols:
    st.error(f"Your data is missing columns: {missing_cols}")
    st.stop()

features = [c for c in df.columns if c != label_col]

# Train / test split and RF training
X = df[features].copy()
y = df[label_col].astype(int).values

imp = SimpleImputer(strategy="median")
X_imp = imp.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.25, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance (hold‑out test)")
st.write(f"Random Forest accuracy: **{acc:.2%}** on a 25% test split.")
st.caption("Accuracy will vary with your uploaded data. On small demos, accuracy is illustrative only.")

# User input form
st.subheader("Enter Your Health Info")
with st.form("user_inputs"):
    c1, c2, c3, c4 = st.columns(4)
    Pregnancies = c1.number_input("Pregnancies", min_value=0, max_value=20, value=2, step=1)
    Glucose = c2.number_input("Glucose", min_value=40, max_value=250, value=120, step=1)
    BloodPressure = c3.number_input("BloodPressure", min_value=30, max_value=200, value=72, step=1)
    SkinThickness = c4.number_input("SkinThickness", min_value=0, max_value=99, value=23, step=1)

    c5, c6, c7, c8 = st.columns(4)
    Insulin = c5.number_input("Insulin", min_value=0, max_value=900, value=94, step=1)
    BMI = c6.number_input("BMI", min_value=10.0, max_value=80.0, value=30.5, step=0.1)
    DiabetesPedigreeFunction = c7.number_input("DiabetesPedigreeFunction", min_value=0.05, max_value=3.0, value=0.5, step=0.01)
    Age = c8.number_input("Age", min_value=15, max_value=100, value=33, step=1)

    submitted = st.form_submit_button("Run Prediction")

user_dict = {
    "Pregnancies": Pregnancies,
    "Glucose": Glucose,
    "BloodPressure": BloodPressure,
    "SkinThickness": SkinThickness,
    "Insulin": Insulin,
    "BMI": BMI,
    "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
    "Age": Age,
}

if submitted:
    # ML prediction
    user_vec = np.array([[user_dict[f] for f in features]])
    user_vec_imp = imp.transform(user_vec)
    proba = rf.predict_proba(user_vec_imp)[0,1]
    ml_label = int(proba >= 0.5)

    # ACT-R style recall
    k_matches, diabetic_ratio, actr_label = recall_predict(df, user_dict, features, label_col=label_col, k=25, match_threshold=0.60)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ML (Random Forest)")
        st.write(f"Prediction: **{'Diabetic' if ml_label==1 else 'Not diabetic'}**")
        st.write(f"Probability of diabetes: **{proba:.1%}**")

    with c2:
        st.markdown("### ACT‑R–style Recall")
        st.write(f"Prediction: **{'Diabetic' if actr_label==1 else 'Not diabetic'}**")
        st.write(f"Nearest similar cases considered (k): **{k_matches}**")
        st.write(f"Share of those cases with diabetes: **{diabetic_ratio:.1%}**")

    st.divider()
    st.markdown("### Feature Importance (Random Forest)")
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    fig = plt.figure()
    plt.bar([features[i] for i in order], importances[order])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Importance")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Feature influence helps explain ML predictions. The ACT‑R view explains via similar past cases.")

st.divider()
st.markdown(
    "Need help? See the README below for setup. "
    "This app is a teaching demo, not medical advice."
)
