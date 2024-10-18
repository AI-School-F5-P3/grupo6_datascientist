import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import numpy as np
import plotly.graph_objects as go

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
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model = XGBClassifier()
        model.load_model("xgboost_stroke_model_1.bin")
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def process_input_data(raw_data):
    """Procesar y preparar los datos de entrada para la predicción."""
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
    numeric_features = ['age', 'hypertension', 'heart_disease']
    for feature in numeric_features:
        processed_data[feature] = raw_data[feature].iloc[0]
    
    # Procesar estado de fumador
    processed_data['smoking_status_never smoked'] = int(raw_data['smoking_status'].iloc[0] == 'never smoked')
    
    # Procesar tipo de trabajo
    processed_data['work_type_Private'] = int(raw_data['work_type'].iloc[0] == 'Private')
    
    # Procesar tipo de residencia
    processed_data['Residence_type_Rural'] = int(raw_data['Residence_type'].iloc[0] == 'Rural')
    
    return processed_data[required_columns]

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

def main():
    st.markdown('<h1 class="main-header">🏥 Predictor de Riesgo de Derrame Cerebral</h1>', unsafe_allow_html=True)
    
    # Cargar modelo
    loaded_model = load_model()
    if loaded_model is None:
        return
    
    # Sidebar con información
    with st.sidebar:
        st.markdown("### ℹ️ Información del Modelo")
        st.info("""
        Este modelo utiliza técnicas de aprendizaje automático (XGBoost) 
        para predecir el riesgo de derrame cerebral basado en diversos 
        factores de salud y estilo de vida.
        """)
        
        st.markdown("### 🎯 Precisión del Modelo")
        st.progress(0.82)
        st.caption("Precisión: 82%")
        
        st.markdown("### ⚠️ Descargo de Responsabilidad")
        st.warning("""
        Esta herramienta es solo para fines educativos e informativos.
        No sustituye el diagnóstico médico profesional.
        """)
    
    # Entrada manual con diseño mejorado
    st.markdown('<p class="subheader">📝 Información Personal</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Género", ["Male", "Female"])
        age = st.number_input("Edad", min_value=1, max_value=120, value=25)
        hypertension = st.selectbox("Hipertensión", [0, 1], 
                                  format_func=lambda x: "Sí" if x == 1 else "No")
    
    with col2:
        heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], 
                                   format_func=lambda x: "Sí" if x == 1 else "No")
        avg_glucose_level = st.number_input("Nivel de Glucosa Promedio", 
                                          min_value=0.0, value=100.0)
        bmi = st.number_input("Índice de Masa Corporal (IMC)", 
                            min_value=0.0, value=25.0)
    
    with col3:
        ever_married = st.selectbox("¿Alguna vez casado?", ["Yes", "No"])
        work_type = st.selectbox("Tipo de Trabajo", 
                               ["Private", "Self-employed", "Govt_job", "children"])
        smoking_status = st.selectbox("Estado de Fumador", 
                                    ["never smoked", "formerly smoked", "smokes", "Unknown"])
    
    residence_type = st.selectbox("Tipo de Residencia", ["Urban", "Rural"])
    
    input_data = pd.DataFrame({
        'gender': [gender], 'age': [age], 'hypertension': [hypertension],
        'heart_disease': [heart_disease], 'ever_married': [ever_married],
        'work_type': [work_type], 'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level], 'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Analizando factores de riesgo..."):
            # Procesar datos
            processed_data = process_input_data(input_data)

            # Realizar la predicción
            prediction = loaded_model.predict(processed_data)
            risk_score = prediction[0]

            # Mostrar resultados
            st.success(f"Predicción: {'Riesgo de Derrame Cerebral' if risk_score == 1 else 'Sin Riesgo de Derrame Cerebral'}")

            # Crear y mostrar gráfico gauge
            gauge_chart = create_gauge_chart(risk_score, "Nivel de Riesgo de Derrame Cerebral")
            st.plotly_chart(gauge_chart)

# Ejecutar el flujo principal de la aplicación
if __name__ == "__main__":
    main()

