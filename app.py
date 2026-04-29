# making ui using streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# saving the modele
with open("heart_disease_model.pkl", "rb") as file:
    model, scaler = pickle.load(file)


# define the columns
categorical_cols = ["Gender", "ChestPainType","FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope", "MajorVessels", "Thalassemia"]
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "ST_Depression"]

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("Heart Disease Prediction App")

st.write("This app predicts the linklihood of heart disease based on the patieent data")

st.markdown("--")

# create input data
col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", min_value=20, max_value=100, value=45)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    ChestPainType = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    RestingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120, step=1)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)

with col2:
    FastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]) 
    RestingECG = st.selectbox("Resting ECG Results", [0, 1, 2])
    MaxHR = st.number_input("Max Heart Rate", min_value=60, max_value=250, value=150, step=1)
    ExerciseAngina = st.selectbox("Exercise Angina",[0, 1])
    ST_Depression = st.number_input("ST_Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    ST_Slope = st.selectbox("ST_Slope", [0, 1, 2])
    MajorVessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    Thalassemia = st.selectbox("Thalassemia", [0, 1, 2])

Gender = 1 if Gender == "Male" else 0

# create data frame 
input_dict ={
      "Age": Age,
    "Gender": Gender,
    "ChestPainType": ChestPainType,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
    "RestingECG": RestingECG,
    "MaxHR": MaxHR,
    "ExerciseAngina": ExerciseAngina,
    "ST_Depression": ST_Depression,
    "ST_Slope": ST_Slope,
    "MajorVessels": MajorVessels,
    "Thalassemia": Thalassemia
}

input_df = pd.DataFrame([input_dict])

input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

expected_encoded = model.feature_names_in_


input_encoded = input_encoded.reindex(columns=expected_encoded,fill_value=0)

# Scale numeric features (Yahan 'scale' ki jagah 'scaler_object' use kiya hai)
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

# prediction button
st.write(input_encoded) 
if st.button("Predict heart disease"):
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("High risk of heart disease")
    else:
        st.success("No sign of heart disease")

   


