import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import plotly.graph_objects as go
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Riesgo de Derrame Cerebral",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E88E5;
            text-align: center;
            margin-bottom: 2rem;
        }
        .subheader {
            font-size: 1.5rem;
            color: #424242;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .info-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            border-bottom: 1px dotted #ccc;
            cursor: help;
        }
        .required-field::after {
            content: " *";
            color: red;
        }
    </style>
""", unsafe_allow_html=True)

# Constantes y configuración
VALID_WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children"]
VALID_SMOKING_STATUS = ["never smoked", "formerly smoked", "smokes", "Unknown"]
VALID_RESIDENCE_TYPES = ["Urban", "Rural"]

@st.cache_resource
def load_model():
    """Cargar el modelo XGBoost."""
    try:
        model = XGBClassifier()
        model.load_model("xgboost_stroke_model_final.bin")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def validate_input_data(input_data):
    """Validar los datos de entrada."""
    errors = []
    
    # Validar edad
    if input_data['age'].iloc[0] < 0 or input_data['age'].iloc[0] > 120:
        errors.append("La edad debe estar entre 0 y 120 años")
    
    # Validar trabajo
    if input_data['work_type'].iloc[0] not in VALID_WORK_TYPES:
        errors.append("Tipo de trabajo no válido")
    
    # Validar estado de fumador
    if input_data['smoking_status'].iloc[0] not in VALID_SMOKING_STATUS:
        errors.append("Estado de fumador no válido")
    
    # Validar tipo de residencia
    if input_data['Residence_type'].iloc[0] not in VALID_RESIDENCE_TYPES:
        errors.append("Tipo de residencia no válido")
    
    # Validar valores binarios
    for field in ['hypertension', 'heart_disease']:
        if input_data[field].iloc[0] not in [0, 1]:
            errors.append(f"El campo {field} debe ser 0 o 1")
    
    if errors:
        raise ValueError("\n".join(errors))

def process_input_data(raw_data):
    """
    Procesar y preparar los datos de entrada para la predicción.
    Solo procesa las variables necesarias para el modelo final.
    """
    # Crear DataFrame con las columnas necesarias
    required_columns = [
        'age',
        'smoking_status_never smoked',
        'hypertension',
        'work_type_Private',
        'Residence_type_Rural',
        'heart_disease'
    ]
    
    processed_data = pd.DataFrame(0, index=[0], columns=required_columns)
    
    # Copiar características numéricas directamente
    processed_data['age'] = raw_data['age'].iloc[0]
    processed_data['hypertension'] = raw_data['hypertension'].iloc[0]
    processed_data['heart_disease'] = raw_data['heart_disease'].iloc[0]
    
    # Procesar variables categóricas
    processed_data['smoking_status_never smoked'] = (raw_data['smoking_status'].iloc[0] == 'never smoked')
    processed_data['work_type_Private'] = (raw_data['work_type'].iloc[0] == 'Private')
    processed_data['Residence_type_Rural'] = (raw_data['Residence_type'].iloc[0] == 'Rural')
    
    return processed_data

def create_gauge_chart(value, title):
    """Crear un gráfico de gauge con Plotly."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 16.5 * 100
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def show_field_info(title, description):
    """Mostrar información sobre un campo con tooltip."""
    st.markdown(f"""
        <div class="tooltip">
            {title}
            <span class="tooltiptext">{description}</span>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🏥 Predictor de Riesgo de Derrame Cerebral</h1>', unsafe_allow_html=True)
    
    # Cargar modelo
    model = load_model()
    if model is None:
        return
    
    # Sidebar con información
    with st.sidebar:
        st.markdown("### ℹ️ Información del Modelo")
        st.info("""
        Este modelo de XGBoost predice el riesgo de derrame cerebral basado en factores clave:
        - Edad
        - Estado de fumador
        - Hipertensión
        - Tipo de trabajo
        - Tipo de residencia
        - Enfermedad cardíaca
        """)
        
        st.markdown("### 🎯 Precisión del Modelo")
        st.progress(0.82)
        st.caption("Precisión: 82%")
        
        st.markdown("### ⚠️ Descargo de Responsabilidad")
        st.warning("""
        Esta herramienta es solo para fines educativos e informativos.
        No sustituye el diagnóstico médico profesional.
        Consulte siempre con un profesional de la salud.
        """)
    
    # Formulario de entrada
    st.markdown('<p class="subheader">📝 Información del Paciente</p>', unsafe_allow_html=True)
    
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Datos Demográficos")
            age = st.number_input("Edad", 
                                min_value=0, 
                                max_value=120, 
                                value=25,
                                help="Edad del paciente en años")
            
            hypertension = st.selectbox(
                "Hipertensión", 
                [0, 1],
                format_func=lambda x: "Sí" if x == 1 else "No",
                help="¿El paciente tiene hipertensión diagnosticada?"
            )
            
            heart_disease = st.selectbox(
                "Enfermedad Cardíaca",
                [0, 1],
                format_func=lambda x: "Sí" if x == 1 else "No",
                help="¿El paciente tiene alguna enfermedad cardíaca diagnosticada?"
            )
        
        with col2:
            st.markdown("##### Estilo de Vida")
            work_type = st.selectbox(
                "Tipo de Trabajo",
                VALID_WORK_TYPES,
                help="Sector laboral principal del paciente"
            )
            
            smoking_status = st.selectbox(
                "Estado de Fumador",
                VALID_SMOKING_STATUS,
                help="Historial de consumo de tabaco"
            )    
        
        with col3:
            st.markdown("##### Ubicación")
            residence_type = st.selectbox(
                "Tipo de Residencia",
                VALID_RESIDENCE_TYPES,
                help="Área de residencia del paciente"
            )
        
        # Botón de predicción
        predict_button = st.form_submit_button("Realizar Predicción", type="primary")
    
    # Procesar predicción cuando se presiona el botón
    if predict_button:
        try:
            # Crear DataFrame con los datos de entrada
            input_data = pd.DataFrame({
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'smoking_status': [smoking_status]
            })
            
            # Validar datos
            validate_input_data(input_data)
            
            # Mostrar spinner durante el procesamiento
            with st.spinner("Analizando factores de riesgo..."):
                # Procesar datos
                processed_data = process_input_data(input_data)
                
                # Realizar predicción
                prediction = model.predict_proba(processed_data)
                risk_score = prediction[0][1]  # Probabilidad de la clase positiva
                
                # Mostrar resultados
                st.markdown("### Resultados del Análisis")
                
                # Mostrar predicción
                # Mostrar predicción
                risk_status = "Alto Riesgo" if risk_score > 0.165 else "Bajo Riesgo"
                risk_color = "red" if risk_score > 0.165 else "green"

                st.markdown(f"<h3 style='color:{risk_color};'>Predicción: {risk_status} de Derrame Cerebral</h3>", unsafe_allow_html=True)

                
                # Mostrar gráfico de gauge
                gauge_chart = create_gauge_chart(risk_score, "Probabilidad de Riesgo de Derrame Cerebral")
                st.plotly_chart(gauge_chart, use_container_width=True)
                
                # Mostrar factores de riesgo principales
                st.markdown("#### Factores de Riesgo Principales")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("✔️ Edad" if age > 65 else "✅ Edad dentro del rango normal")
                    st.info("⚠️ Hipertensión presente" if hypertension else "✅ Sin hipertensión")
                    st.info("⚠️ Enfermedad cardíaca presente" if heart_disease else "✅ Sin enfermedad cardíaca")
                with col2:
                    st.info("⚠️ Fumador activo" if smoking_status == "smokes" else "✅ No fumador o ex-fumador")
                    st.info("ℹ️ Residencia: " + residence_type)
                    st.info("ℹ️ Tipo de trabajo: " + work_type)
        
        except ValueError as e:
            st.error(f"Error en los datos de entrada:\n{str(e)}")
        except Exception as e:
            st.error(f"Error inesperado: {str(e)}")
            st.error("Por favor, contacte al equipo de soporte si el problema persiste.")

if __name__ == "__main__":
    main()