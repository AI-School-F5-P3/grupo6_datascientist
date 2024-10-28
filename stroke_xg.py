import streamlit as st
import pandas as pd
from plotly import graph_objects as go
import tensorflow
import joblib
import pickle
import xgboost as xgb

# Definición de constantes
VALID_WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
VALID_SMOKING_STATUS = ["formerly smoked", "never smoked", "smokes", "Unknown"]
VALID_RESIDENCE_TYPES = ["Urban", "Rural"]

@st.cache_resource
def load_model():
    """Cargar el modelo XGBoost."""
    try:
        model = xgb.Booster()
        model.load_model("models/xgboost_stroke_model_final.bin")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo XGBoost: {str(e)}")
        return None

def create_gauge_chart(value, title):
    """Create a gauge chart for risk visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 16.5], 'color': "lightgreen"},
                {'range': [16.5, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 16.5
            }
        }
    ))
    return fig

def modelo_xgboost():
    """Función principal para la interfaz de predicción de derrame cerebral."""
    # Add custom CSS for white labels
    st.markdown("""
        <style>
        .white-label {
            color: white !important;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - XGBoost</h1>", unsafe_allow_html=True)
    
    model = load_model()
    
    # Initialize session state variables
    if 'risk_score' not in st.session_state:
        st.session_state.risk_score = None
    
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None

    if "prediction_made" not in st.session_state:
        st.session_state.prediction_made = False
    
    st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)
    
    # Crear el formulario
    with st.form(key='patient_form'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p class='white-subheader'>Datos Demográficos</p>", unsafe_allow_html=True)
            st.markdown("<p class='white-label'>Edad</p>", unsafe_allow_html=True)
            age = st.number_input("Edad", min_value=0, max_value=120, value=25, help="Edad del paciente en años", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Género</p>", unsafe_allow_html=True)
            gender = st.selectbox("Género", ["Male", "Female"], help="Género del paciente", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Estado Civil</p>", unsafe_allow_html=True)
            ever_married = st.selectbox("Estado Civil", ["Yes", "No"], help="¿El paciente ha estado alguna vez casado?", label_visibility="collapsed")
        
        with col2:
            st.markdown("<p class='white-subheader'>Estilo de Vida</p>", unsafe_allow_html=True)
            st.markdown("<p class='white-label'>Tipo de Trabajo</p>", unsafe_allow_html=True)
            work_type = st.selectbox("Tipo de Trabajo", VALID_WORK_TYPES, help="Sector laboral principal del paciente", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Estado de Fumador</p>", unsafe_allow_html=True)
            smoking_status = st.selectbox("Estado de Fumador", VALID_SMOKING_STATUS, help="Historial de consumo de tabaco", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Nivel Promedio de Glucosa</p>", unsafe_allow_html=True)
            avg_glucose_level = st.number_input("Nivel Promedio de Glucosa", min_value=0.0, max_value=300.0, value=100.0, help="Nivel promedio de glucosa en sangre", label_visibility="collapsed")
        
        with col3:
            st.markdown("<p class='white-subheader'>Ubicación y Salud</p>", unsafe_allow_html=True)
            st.markdown("<p class='white-label'>Tipo de Residencia</p>", unsafe_allow_html=True)
            residence_type = st.selectbox("Tipo de Residencia", VALID_RESIDENCE_TYPES, help="Área de residencia del paciente", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Índice de Masa Corporal (BMI)</p>", unsafe_allow_html=True)
            bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0, help="Índice de masa corporal del paciente", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Hipertensión</p>", unsafe_allow_html=True)
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene hipertensión diagnosticada?", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Enfermedad Cardíaca</p>", unsafe_allow_html=True)
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene alguna enfermedad cardíaca diagnosticada?", label_visibility="collapsed")

        # Botones de submit dentro del formulario
        col_buttons = st.columns([1, 1])
        with col_buttons[0]:
            predict_button = st.form_submit_button("Realizar Predicción", type="primary")
        with col_buttons[1]:
            save_button = st.form_submit_button("Guardar en BBDD", type="secondary")

        if predict_button:
            try:
                # Store patient data in session state
                st.session_state.patient_data = {
                    'gender': gender,
                    'age': age,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'ever_married': ever_married,
                    'work_type': work_type,
                    'Residence_type': residence_type,
                    'avg_glucose_level': avg_glucose_level,
                    'bmi': bmi,
                    'smoking_status': smoking_status
                }
                
                input_data = pd.DataFrame({
                    'age': [age],
                    'hypertension': [hypertension],
                    'heart_disease': [heart_disease],
                    'work_type': [work_type],
                    'Residence_type': [residence_type],
                    'smoking_status': [smoking_status]
                })
                
                with st.spinner("Analizando factores de riesgo..."):
                    prediction = model.predict_proba(input_data)
                    st.session_state.risk_score = prediction[0][1]

                # Displaying results
                st.markdown("<h2 class='subtitle'>Resultados del Análisis</h2>", unsafe_allow_html=True)
                risk_status = "Alto Riesgo" if st.session_state.risk_score > 0.165 else "Bajo Riesgo"
                risk_color = "red" if st.session_state.risk_score > 0.165 else "green"
                st.markdown(f"""
                    <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>
                        <h3 style='color: white; text-align: center;'>Estado: {risk_status}</h3>
                        <p style='color: white; text-align: center;'>Probabilidad de derrame cerebral: {st.session_state.risk_score:.2%}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.plotly_chart(create_gauge_chart(st.session_state.risk_score, "Riesgo de Derrame Cerebral"))
                st.session_state.prediction_made = True

            except Exception as e:
                st.error(f"Error en el procesamiento: {str(e)}")

        if save_button and st.session_state.prediction_made:
            try:
                if 'patient_data' in st.session_state and 'risk_score' in st.session_state:
                    patient_data = st.session_state.patient_data
                    risk_score_db = 1 if st.session_state.risk_score > 0.165 else 0

                    data_to_insert = {
                        'gender': patient_data['gender'],
                        'age': patient_data['age'],
                        'hypertension': patient_data['hypertension'],
                        'heart_disease': patient_data['heart_disease'],
                        'ever_married': patient_data['ever_married'],
                        'work_type': patient_data['work_type'],
                        'residence_type': patient_data['Residence_type'],
                        'avg_glucose_level': patient_data['avg_glucose_level'],
                        'bmi': patient_data['bmi'],
                        'smoking_status': patient_data['smoking_status'],
                        'stroke': risk_score_db
                    }

                    df_to_insert = pd.DataFrame([data_to_insert])

                    with engine.connect() as connection:
                         df_to_insert.to_sql('patients', connection, if_exists='append', index=False)
                         st.success("Datos guardados en la base de datos correctamente.")
                    
                else:
                    st.warning("No hay datos de paciente o puntuación de riesgo para guardar.")
                    
            except Exception as e:
                st.error(f"Error al guardar en la base de datos: {str(e)}")