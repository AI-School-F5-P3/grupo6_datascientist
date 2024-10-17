import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# Aplicar estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Cargar el modelo y el escalador con caché"""
    try:
        age_scaler = joblib.load('age_scaler.joblib')
        model = XGBClassifier()
        model.load_model("xgboost_stroke_model_final.bin")
        return age_scaler, model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None

def process_input_data(raw_data):
    """Procesar y preparar los datos de entrada"""
    # Generar dummies
    raw_data['smoking_status_never smoked'] = 1 if raw_data['smoking_status'].iloc[0] == "never smoked" else 0
    raw_data['Residence_type_Rural'] = 1 if raw_data['Residence_type'].iloc[0] == "Rural" else 0
    raw_data['work_type_Private'] = 1 if raw_data['work_type'].iloc[0] == "Private" else 0
    
    # Seleccionar características importantes
    important_features = [
        "age", "smoking_status_never smoked", "hypertension",
        "work_type_Private", "Residence_type_Rural", "heart_disease"
    ]
    
    return raw_data[important_features]

def main():
    # Cargar modelo y escalador
    age_scaler, loaded_model = load_model_and_scaler()
    if age_scaler is None or loaded_model is None:
        st.error("No se pudieron cargar los modelos necesarios.")
        return

    # Título y descripción
    st.title("🏥 Predicción de Riesgo de Derrame Cerebral")
    st.markdown("""
        Esta aplicación utiliza machine learning para evaluar el riesgo de derrame cerebral 
        basándose en diversos factores de salud y estilo de vida.
    """)

    # Crear dos columnas para la entrada de datos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Información Personal")
        gender = st.selectbox("Género", ["Male", "Female"])
        age = st.number_input("Edad", min_value=1, max_value=120, value=25)
        bmi = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0)
        ever_married = st.selectbox("Estado Civil", ["Yes", "No"])
        
    with col2:
        st.subheader("Información Médica")
        hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
        avg_glucose_level = st.number_input("Nivel de Glucosa Promedio", min_value=50.0, max_value=400.0, value=100.0)
        smoking_status = st.selectbox("Estado de Fumador", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    # Información adicional
    work_type = st.selectbox("Tipo de Trabajo", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Tipo de Residencia", ["Urban", "Rural"])

    # Botón de predicción
    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando datos..."):
            # Crear DataFrame con los datos de entrada
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

            # Procesar datos
            processed_data = process_input_data(input_data)
            
            # Escalar la edad
            processed_data.loc[:, 'age'] = age_scaler.transform(processed_data[['age']]).astype(float)
            
            # Realizar predicción
            prediction_prob = float(loaded_model.predict_proba(processed_data)[:, 1][0])  # Convertir a float estándar
            
            # Mostrar resultados
            st.subheader("Resultados del Análisis")
            
            # Crear una barra de progreso para el riesgo
            risk_percentage = prediction_prob * 100
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.progress(float(prediction_prob))  # Asegurar que sea float
                st.write(f"Probabilidad de riesgo: {risk_percentage:.1f}%")
            
            # Mostrar el resultado con formato
            if prediction_prob >= 0.165:
                st.error(f"⚠️ Riesgo Alto de Derrame Cerebral ({risk_percentage:.1f}%)")
                st.markdown("""
                    ### Recomendaciones:
                    - Consulte a un médico lo antes posible
                    - Monitoree su presión arterial regularmente
                    - Mantenga un estilo de vida saludable
                """)
            else:
                st.success(f"✅ Riesgo Bajo de Derrame Cerebral ({risk_percentage:.1f}%)")
                st.markdown("""
                    ### Recomendaciones:
                    - Continúe manteniendo hábitos saludables
                    - Realice chequeos médicos regulares
                    - Mantenga una dieta equilibrada
                """)

            # Mostrar factores de riesgo
            st.subheader("Factores de Riesgo Principales")
            col1, col2, col3 = st.columns(3)
            with col1:
                if age > 65:
                    st.warning("Edad: Factor de riesgo elevado")
                if hypertension:
                    st.warning("Hipertensión: Factor de riesgo presente")
            with col2:
                if heart_disease:
                    st.warning("Enfermedad cardíaca: Factor de riesgo presente")
                if avg_glucose_level > 126:
                    st.warning("Glucosa elevada: Factor de riesgo presente")
            with col3:
                if bmi > 30:
                    st.warning("IMC elevado: Factor de riesgo presente")
                if smoking_status == "smokes":
                    st.warning("Fumador activo: Factor de riesgo presente")

if __name__ == "__main__":
    main()