import streamlit as st
import pickle
import numpy as np
import os

# Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load Model
@st.cache_resource
def load_all():
    if os.path.exists('heart_model.pkl'):
        return pickle.load(open('heart_model.pkl', 'rb'))
    return None

data = load_all()

st.title("❤️ Heart Disease Prediction System")
st.markdown("---")

if data is None:
    st.error("Error: heart_model.pkl file nahi mili!")
else:
    model = data['model']
    mapping = data['mapping']

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 100, 45)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        # ASY=0, ATA=1, NAP=2, TA=3 (Alphabetical order of categories)
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: ["ASY", "ATA", "NAP", "TA"][x])
        trestbps = st.number_input("Resting BP", 50, 200, 120)
        chol = st.number_input("Cholesterol", 0, 600, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120", options=[0, 1])
        restecg = st.selectbox("Resting ECG", options=[0, 1, 2], format_func=lambda x: ["LVH", "Normal", "ST"][x])
        thalach = st.number_input("Max Heart Rate", 50, 210, 150)
        exang = st.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 0.0)
        slope = st.selectbox("ST Slope", options=[0, 1, 2], format_func=lambda x: ["Down", "Flat", "Up"][x])

    if st.button("Predict Result"):
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
        prediction = model.predict(features)
        prob = model.predict_proba(features)[0][1]

        if prediction[0] == 1:
            st.error(f"🚨 High Risk Detected! (Confidence: {prob:.2%})")
        else:
            st.success(f"✅ No Significant Risk (Confidence: {1-prob:.2%})")

        # Graphs Display
        st.image('confusion_matrix.png', caption="Model Performance")
        st.image('feature_importance.png', caption="Why this result?")