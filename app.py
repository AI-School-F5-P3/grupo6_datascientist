import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier

# Cargar el modelo entrenado
model = joblib.load('xgboost_model.pkl')

# Título de la aplicación
st.title("Predicción de Riesgo de Ictus")

# Entradas del usuario
gender = st.selectbox("Género:", ["Female", "Male"])
age = st.number_input("Edad:", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hipertensión:", ["No", "Yes"])
heart_disease = st.selectbox("Enfermedad del corazón:", ["No", "Yes"])
ever_married = st.selectbox("¿Alguna vez casado?:", ["No", "Yes"])
residence_type = st.selectbox("Tipo de residencia:", ["Rural", "Urban"])
avg_glucose_level = st.number_input("Nivel promedio de glucosa:", min_value=0.0, value=85.0)
bmi = st.number_input("Índice de masa corporal (BMI):", min_value=0.0, value=22.0)
work_type = st.selectbox("Tipo de trabajo:", ["Govt_job", "Private", "Self-employed", "children"])
smoking_status = st.selectbox("Estado de fumar:", ["Unknown", "formerly smoked", "never smoked", "smokes"])

# Convertir las entradas en formato adecuado para el modelo
input_data = {
    'gender': 1 if gender == 'Male' else 0,
    'age': age,
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'ever_married': 1 if ever_married == 'Yes' else 0,
    'Residence_type': 1 if residence_type == 'Urban' else 0,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
    'work_type_Private': 1 if work_type == 'Private' else 0,
    'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
    'work_type_children': 1 if work_type == 'children' else 0,
    'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
    'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
    'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
    'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
}

input_df = pd.DataFrame(input_data, index=[0])

# Botón para predecir
if st.button("Predecir"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)[:, 1]

    # Mostrar el resultado
    if prediction[0] == 1:
        st.success("El paciente está en riesgo de sufrir un ictus.")
    else:
        st.success("El paciente no está en riesgo de sufrir un ictus.")

    st.write(f"Probabilidad de riesgo de ictus: {prediction_proba[0]:.2f}")

# Ejecutar la aplicación con: streamlit run app.py

