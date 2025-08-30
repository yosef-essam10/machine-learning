
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model/final_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.title("Heart Disease Prediction 🫀")

st.write("Enter patient data to predict heart disease:")

# User inputs
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1])  
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# Build dataframe with same column names as training
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', ' sex', ' cp', ' trestbps', ' chol', ' fbs',
                                   ' restecg', ' thalach', ' exang', ' oldpeak',
                                   ' slope', ' ca', ' thal'])

# Convert to float
input_data = input_data.astype(float)

# Scale data
X_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(X_scaled)
    if prediction[0] == 1:
        st.error("⚠️ The patient is likely to have heart disease")
    else:
        st.success("✅ The patient is healthy (no heart disease)")
