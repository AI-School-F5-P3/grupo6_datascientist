# Importar librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import config

# Cargar el modelo preentrenado
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Función para hacer las predicciones
def predict_stroke(model, input_data):
    return model.predict(input_data)[0]

# Función para preparar los datos ingresados por el usuario
def preprocess_user_input(df):
    # Crear DataFrame con las características necesarias
    processed_df = pd.DataFrame()
    
    # Copiar las columnas numéricas directamente
    processed_df['age'] = df['age']
    processed_df['elderly_comorbid'] = ((df['age'] > 65) & ((df['hypertension'] + df['heart_disease']) > 0)).astype(int)
    processed_df['hypertension'] = df['hypertension']
    
    # Procesar work_type
    work_type_dummies = pd.get_dummies(df['work_type'], prefix='work_type')
    if 'work_type_Private' in work_type_dummies.columns:
        processed_df['work_type_Private'] = work_type_dummies['work_type_Private']
    else:
        processed_df['work_type_Private'] = 0
        
    # Procesar smoking_status
    smoking_dummies = pd.get_dummies(df['smoking_status'], prefix='smoking_status')
    if 'smoking_status_never smoked' in smoking_dummies.columns:
        processed_df['smoking_status_never smoked'] = smoking_dummies['smoking_status_never smoked']
    else:
        processed_df['smoking_status_never smoked'] = 0

    # Asegurar el orden correcto de las columnas
    expected_columns = ['age', 'elderly_comorbid', 'hypertension', 'work_type_Private', 'smoking_status_never smoked']
    processed_df = processed_df[expected_columns]
    
    return processed_df

# Interfaz de Streamlit
def main():
    st.title(config.APP_TITLE)
    st.write(config.APP_DESCRIPTION)

    # Cargar el modelo
    model = load_model(config.MODEL_PATH)

    # Entradas del usuario para las características
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Edad", min_value=1, max_value=100, value=50)
        hypertension = st.selectbox("Hipertensión", options=[0, 1], help="0: No, 1: Sí")
        heart_disease = st.selectbox("Enfermedad cardíaca", options=[0, 1], help="0: No, 1: Sí")
        work_type = st.selectbox("Tipo de trabajo", options=["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    
    with col2:
        smoking_status = st.selectbox("Estado de fumador", options=["never smoked", "formerly smoked", "smokes"])
        avg_glucose_level = st.number_input("Nivel promedio de glucosa", min_value=50.0, max_value=300.0, value=100.0)
        bmi = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0)

    # Convertir las entradas en un DataFrame
    input_data = pd.DataFrame({
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'work_type': [work_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Preprocesar los datos del usuario
    processed_input = preprocess_user_input(input_data)

    # Botón de predicción
    if st.button("Predecir"):
        # Realizar predicción
        prediction = predict_stroke(model, processed_input)
        
        # Mostrar resultado
        st.subheader("Resultado de la Predicción")
        if prediction == 1:
            st.error("⚠️ Se detecta riesgo elevado de derrame cerebral. Se recomienda consultar con un profesional médico.")
        else:
            st.success("✅ No se detecta riesgo elevado de derrame cerebral.")
        
        # Mostrar recomendaciones generales
        st.subheader("Recomendaciones Generales")
        st.write("""
        Independientemente del resultado, es importante mantener hábitos saludables:
        - Mantener una dieta equilibrada
        - Realizar ejercicio regular
        - No fumar
        - Controlar la presión arterial y el nivel de glucosa
        - Consultar regularmente con su médico
        """)

if __name__ == "__main__":
    main()