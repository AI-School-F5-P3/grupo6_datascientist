import streamlit as st
from config import APP_TITLE, APP_DESCRIPTION, MODEL_PATH, TRANSFORMER_PATH
import joblib
import pandas as pd

# Título y descripción de la aplicación
st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

# Cargar el modelo y el pipeline de transformación
model = joblib.load(MODEL_PATH)
transformer = joblib.load(TRANSFORMER_PATH)

# Interfaz de usuario con Streamlit para recibir datos
st.subheader("Introduce los datos del paciente")

# Input para las características del paciente
age = st.number_input("Edad", min_value=0, max_value=120)
avg_glucose_level = st.number_input("Nivel promedio de glucosa", min_value=0.0, format="%.2f")
bmi = st.number_input("Índice de masa corporal (BMI)", min_value=0.0, format="%.2f")
hypertension = st.selectbox("¿Hipertensión?", options=["0", "1"])
heart_disease = st.selectbox("¿Enfermedad cardíaca?", options=["0", "1"])
ever_married = st.selectbox("¿Alguna vez casado?", options=["Yes", "No"])
work_type = st.selectbox("Tipo de trabajo", options=["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Tipo de residencia", options=["Urban", "Rural"])
smoking_status = st.selectbox("Estado de fumar", options=["formerly smoked", "never smoked", "smokes", "Unknown"])

# Preparar los datos de entrada para hacer la predicción
input_data = pd.DataFrame({
    'age': [age],
    'avg_glucose_level': [avg_glucose_level],
    'bmi': [bmi],
    'hypertension': [hypertension],
    'heart_disease': [heart_disease],
    'ever_married': [ever_married],
    'work_type': [work_type],
    'Residence_type': [residence_type],
    'smoking_status': [smoking_status]
})

# Función para hacer predicciones
def predict(input_data):
    # Transformar los datos de entrada
    transformed_data = transformer.transform(input_data)
    # Realizar la predicción
    prediction = model.predict(transformed_data)
    return prediction

if st.button("Predecir"):
    # Hacer la predicción
    prediction = predict(input_data)
    st.write(f"La predicción es: {'Derrame cerebral' if prediction[0] == 1 else 'Sin derrame cerebral'}")