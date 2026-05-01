import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# --- MODEL LOADING ---
# Hum check kar rahe hain ke model file maujood hai ya nahi
model_path = 'heart_model.pkl'

@st.cache_resource # Is se model baar baar load nahi hoga, app fast chalegi
def load_model():
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    return None

model = load_model()

if model is None:
    st.error("⚠️ Error: 'heart_model.pkl' nahi mili! Pehle training script chala kar model file banayein aur GitHub par upload karein.")

# --- CUSTOM STYLING (CSS) ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("❤️ Heart Disease Prediction System")
st.write("Patient ka clinical data enter karein taake risk level ka andaza lagaya ja sakay.")
st.markdown("---")

# --- INPUT FORM ---
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (Umr)", min_value=1, max_value=120, value=50)
        
        sex_display = st.selectbox("Gender (Jins)", options=["Male", "Female"])
        sex = 1 if sex_display == "Male" else 0
        
        cp_options = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }
        cp_display = st.selectbox("Chest Pain Type", options=list(cp_options.keys()))
        cp = cp_options[cp_display]
        
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", value=200)
        
        fbs_display = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"])
        fbs = 1 if fbs_display == "True" else 0

    with col2:
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], help="0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy")
        thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
        
        exang_display = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
        exang = 1 if exang_display == "Yes" else 0
        
        oldpeak = st.number_input("ST Depression (Oldpeak)", value=1.0, format="%.1f", step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
        
        thal_options = {"Normal": 1, "Fixed Defect": 2, "Reversable Defect": 3}
        thal_display = st.selectbox("Thalassemia", options=list(thal_options.keys()))
        thal = thal_options[thal_display]

st.markdown("---")

# --- PREDICTION LOGIC ---
if st.button("Analyze Results"):
    if model is not None:
        # CRITICAL: Ye order heart_2.csv ke columns ke mutabiq hai
        # Order: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        # Display Results
        st.subheader("Analysis Summary:")
        
        if prediction[0] == 1:
            if probability > 0.8:
                st.error(f"### 🚨 Result: High risk of heart disease")
            else:
                st.warning(f"### ⚠️ Result: Sign of heart disease")
        else:
            if probability < 0.2:
                st.success(f"### ✅ Result: No sign of heart disease")
            else:
                st.info(f"### ℹ️ Result: Low risk of heart disease")
        
        # Confidence Meter
        st.write(f"**Confidence Level:** {probability:.2%}")
        st.progress(probability)
    else:
        st.error("Model file load nahi ho saki. Please check your GitHub repository.")

st.markdown("---")
st.caption("Note: Ye AI model sirf educational purpose ke liye hai. Medical advice ke liye doctor se ruju karein.")