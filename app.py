
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

# Cargar el modelo y el escalador
age_scaler = joblib.load('age_scaler.joblib')
xgb_model = xgb.Booster()
xgb_model.load_model('xgboost_stroke_model_final.bin')


# Definir un nuevo umbral
new_threshold = 0.165  # Ajusta este valor según sea necesario

# Título de la app
st.title("Stroke Prediction App")

# 1. Solicitud de las variables de entrada
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=120, value=25)
hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])
ever_married = st.selectbox("Ever Married", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# Botón para hacer la predicción
if st.button("Predecir"):
    # Crear el dataframe a partir de las entradas del usuario
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # 2. Preparación de los datos
    input_data = pd.get_dummies(input_data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])

    # Asegurarse de que las columnas dummy que faltan se rellenen con 0
    expected_cols = ['gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
                     'work_type_Private', 'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural', 'Residence_type_Urban',
                     'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes']

    for col in expected_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    # Seleccionar solo las características relevantes (important_features)
    important_features = [
        "age",
        "smoking_status_never smoked",
        "hypertension",
        "work_type_Private",
        "Residence_type_Rural",
        "heart_disease"
    ]

    input_data = input_data[important_features]

    # 4. Escalar la edad usando el escalador previamente entrenado
    input_data['age'] = age_scaler.transform(input_data[['age']])

    # 5. Realizar la predicción usando el modelo XGBoost
    dmatrix = xgb.DMatrix(input_data)
    prediction = xgb_model.predict(dmatrix)

    # Mostrar el resultado considerando el nuevo umbral
    st.subheader("Resultado de la Predicción")
    if prediction[0] >= new_threshold:
        st.write(f"El modelo predice que el paciente está en riesgo de tener un derrame cerebral (1). Probabilidad: {prediction[0]:.2f}")
    else:
        st.write(f"El modelo predice que el paciente no está en riesgo de tener un derrame cerebral (0). Probabilidad: {prediction[0]:.2f}")
