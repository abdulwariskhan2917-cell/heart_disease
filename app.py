import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Model load karein
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: 'heart_model.pkl' file not found. Please run the training script first.")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's clinical data below to assess the risk level.")

# Form for user input
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Gender", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        cp = st.selectbox("Chest Pain Type", options=[(0, "Typical Angina"), (1, "Atypical Angina"), (2, "Non-anginal Pain"), (3, "Asymptomatic")], format_func=lambda x: x[1])[0]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", value=120)
        chol = st.number_input("Serum Cholestoral (mg/dl)", value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(1, "True"), (0, "False")], format_func=lambda x: x[1])[0]

    with col2:
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2])
        thalach = st.number_input("Maximum Heart Rate Achieved", value=150)
        exang = st.selectbox("Exercise Induced Angina", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]
        oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.25, value=1.0, format="%.1f")
        slope = st.selectbox("Slope of Peak Exercise ST", options=[0, 1, 2])
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", options=[(1, "Normal"), (2, "Fixed Defect"), (3, "Reversable Defect")], format_func=lambda x: x[1])[0]

st.markdown("---")

# Prediction logic
if st.button("Analyze Results"):
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # Custom Output Messages as requested
    if prediction[0] == 1:
        if probability > 0.8:
            st.error("### Result: High risk of heart disease")
        else:
            st.warning("### Result: Sign of heart disease")
    else:
        if probability < 0.2:
            st.success("### Result: No sign of heart disease")
        else:
            st.info("### Result: Low risk of heart disease")
            
    st.write(f"Confidence Level: {probability:.2%}")

    # ... (Prediction logic aur result display) ...
    st.write(f"Confidence Level: {probability:.2%}")

# --- YAHAN ADD KAREIN ---
st.markdown("---") # Aik line separator ke liye

if st.checkbox("Show Model Analysis Graphs"):
    st.subheader("Model Performance Analysis")
    
    # Check karein ke images folder mein hain ya nahi
    try:
        st.write("**Confusion Matrix**")
        st.image("confusion_matrix.png", caption="Ye graph sahi aur ghalat predictions dikhata hai.")
        
        st.write("**Feature Importance**")
        st.image("feature_importance.png", caption="Ye dikhata hai ke kaunse factors prediction ke liye ahem hain.")
    except:
        st.error("Images not found! Pehle 'train_model.py' run karein taake graphs save ho sakein.")

