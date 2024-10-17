import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier


# Cargar el modelo y el escalador
age_scaler = joblib.load('age_scaler.joblib')
loaded_model = XGBClassifier()
loaded_model.load_model("xgboost_stroke_model_final.bin")

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
    raw_input_data = pd.DataFrame({
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

    raw_input_data.reset_index(drop=True, inplace=True)

    # 2. Preparación de los datos (generar dummies manualmente)
    # Inicializar columnas dummy
    raw_input_data['smoking_status_never smoked'] = 1 if smoking_status == "never smoked" else 0
    raw_input_data['Residence_type_Rural'] = 1 if residence_type == "Rural" else 0
    raw_input_data['work_type_Private'] = 1 if work_type == "Private" else 0
    
    st.write(raw_input_data)

    # Seleccionar solo las características relevantes (important_features)
    important_features = [
        "age",
        "smoking_status_never smoked",
        "hypertension",
        "work_type_Private",
        "Residence_type_Rural",
        "heart_disease"
    ]


    # Mantener solo las columnas importantes
    input_data_final = raw_input_data[important_features]


    # 4. Escalar la edad usando el escalador previamente entrenado
    input_data_final.loc[:, 'age'] = age_scaler.transform(input_data_final[['age']]).astype(float)
    
    input_data_final.reset_index(drop=True, inplace=True)
    
    st.write(input_data_final)

    # 5. Realizar la predicción usando el modelo XGBoost
    prediction = loaded_model.predict_proba(input_data_final)[:, 1]  # Predicción de probabilidad para la clase positiva
    
    st.write(prediction)

    # Mostrar el resultado considerando el nuevo umbral
    st.subheader("Resultado de la Predicción")
    if prediction[0] >= new_threshold:
        st.write(f"El modelo predice que el paciente está en riesgo de tener un derrame cerebral (1). Probabilidad: {prediction[0]:.2f}")
    else:
        st.write(f"El modelo predice que el paciente no está en riesgo de tener un derrame cerebral (0). Probabilidad: {prediction[0]:.2f}")

