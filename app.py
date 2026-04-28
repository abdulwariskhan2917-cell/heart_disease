# making ui using streamlit
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# saving the model
with open("heart_disease_model.pkl", "rb") as file:
    model = pickle.load(file)

# define the columns
categorical_cols = ["sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"] 

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

st.title("Heart Disease Prediction App")

st.write("This app predicts the linklihood of heart disease based on the patieent data")

st.markdown("--")

# create input data

col1, col2 = st.columns(2)
with col1:
    Age = st.number_input("Age", 20 , 100, 45)
    Sex = st.selectbox("Sex", ["M", "F"])
    Chest_pain_type = st.selectbox("Chest Pain Type", [0,1,2,3])
    RestingBP = st.number_input("Resting Blood Pressure (mn Hg)", 80, 200, 120)
    Cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

with col2:
    FastingBP = st.selectbox("Fasting Blood Sugar > 120 mg/dl",[0,1])
    RestingECG = st.selectbox("Resting ECG Result", [0, 1, 2])
    MaxHR = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    ExerciseAngina = st.selectbox("Exercise Induced Angina", [0, 1])
    Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    ST_Slope = st.selectbox("ST Slope", [0, 1, 2])

# perprocess input

Sex = 1 if Sex ==  "Male" else 0

# Numbers ko text mein convert karne ke liye (Mapping)
cp_map = {0: 'TA', 1: 'ATA', 2: 'NAP', 3: 'ASY'}
ecg_map = {0: 'Normal', 1: 'ST', 2: 'LVH'}
slope_map = {0: 'Up', 1: 'Flat', 2: 'Down'}

# create data frame 
input_dict ={
      "Age":[ Age],
    "Sex": [Sex],
    "ChestPainType": [Chest_pain_type],
    "RestingBP": [RestingBP],
    "Cholesterol": [Cholesterol],
    "FastingBS": [FastingBP],
    "RestingECG": [RestingECG],
    "MaxHR": [MaxHR],
    "ExerciseAngina": [ExerciseAngina],
    "Oldpeak": [Oldpeak],
    "ST_Slope": [ST_Slope]
}
input_df = pd.DataFrame(input_dict)


# Sahi column names (Exact match zaroori hai)
categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Model aur Scaler ko tuple se nikalne ka sahi tariqa
if isinstance(model, tuple):
    # Agar tuple hai, toh pehla model hai aur doosra scaler
    scaler_object = model[1] 
    model = model[0]
else:
    # Agar aapne scaler alag se load kiya hai toh uska variable yahan dein
    # Maslan: scaler_object = joblib.load('scaler.pkl')
    scaler_object = scale # Agar 'scale' aapka apna variable hai

# Feature names nikalna
expected_encoded = model.feature_names_in_

# Reindex karna
input_encoded = input_encoded.reindex(columns=expected_encoded, fill_value=0)

# Scale numeric features (Yahan 'scale' ki jagah 'scaler_object' use kiya hai)
input_encoded[numeric_cols] = scaler_object.transform(input_encoded[numeric_cols])

# prediction button
st.write(input_encoded) # Dekhein kya saare columns (ChestPainType_ASY waghaira) ban rahe hain?
if st.button("Predict heart disease"):
    prediction = model.predict(input_encoded)[0]

    if prediction == 1:
        st.error("High risk of heart disease")
    else:
        st.success("No sign of heart disease")

st.caption("Develop by Vikas Sharma | @2026 | Mechine learning project")    


