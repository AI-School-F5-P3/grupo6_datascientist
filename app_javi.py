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
    body {
        color: #FFFFFF;
        background-color: #0A2647;
    }
    .stApp {
        background-color: #0A2647;
    }
    .main-container {
        padding: 2rem;
        background-color: #144272;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem;
    }
    .title {
        font-size: 2.2rem;
        font-weight: bold;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.5rem;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1rem;
        text-align: center;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .feature-container {
        background-color: #205295;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        font-size: 0.9rem;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #2C74B3;
        color: #FFFFFF;
    }
    .stButton>button:hover {
        background-color: #205295;
    }
    /* Cambiar el color de las etiquetas a blanco */
    .stSelectbox > div > label, 
    .stNumberInput > div > label,
    .stSlider > div > label {
        color: white !important;
    }
    /* Mantener el texto de entrada en negro para mejor legibilidad */
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        color: black !important;
        background-color: white !important;
    }
    /* Asegurar que las opciones del menú desplegable sean negras sobre fondo blanco */
    .stSelectbox > div > div > ul > li {
        color: black !important;
    }
    /* Estilo para los subtítulos de las secciones */
    .white-subheader {
        color: white;
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 10px;
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
    if input_data['age'].iloc[0] < 0 or input_data['age'].iloc[0] > 120:
        errors.append("La edad debe estar entre 0 y 120 años")
    if input_data['work_type'].iloc[0] not in VALID_WORK_TYPES:
        errors.append("Tipo de trabajo no válido")
    if input_data['smoking_status'].iloc[0] not in VALID_SMOKING_STATUS:
        errors.append("Estado de fumador no válido")
    if input_data['Residence_type'].iloc[0] not in VALID_RESIDENCE_TYPES:
        errors.append("Tipo de residencia no válido")
    for field in ['hypertension', 'heart_disease']:
        if input_data[field].iloc[0] not in [0, 1]:
            errors.append(f"El campo {field} debe ser 0 o 1")
    if errors:
        raise ValueError("\n".join(errors))

def process_input_data(raw_data):
    """Procesar y preparar los datos de entrada para la predicción."""
    required_columns = [
        'age', 'smoking_status_never smoked', 'hypertension', 'work_type_Private',
        'Residence_type_Rural', 'heart_disease'
    ]
    processed_data = pd.DataFrame(0, index=[0], columns=required_columns)
    processed_data['age'] = raw_data['age'].iloc[0]
    processed_data['hypertension'] = raw_data['hypertension'].iloc[0]
    processed_data['heart_disease'] = raw_data['heart_disease'].iloc[0]
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

def main_page():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Bienvenido al Predictor de Riesgo de Derrame Cerebral</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>HOSPITAL F5</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", use_column_width=True)
    
    st.markdown("<p class='description'>Nuestra aplicación utiliza tecnología de vanguardia para evaluar el riesgo de derrame cerebral basado en factores de salud individuales.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='feature-container'>
            <h3 class='feature-title'>Modelo XGBoost</h3>
            <p class='feature-description'>Predicciones precisas con aprendizaje automático avanzado.</p>
        </div>
        <div class='feature-container'>
            <h3 class='feature-title'>Modelo de Red Neuronal</h3>
            <p class='feature-description'>Próximamente: Evaluación con redes neuronales artificiales.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-container'>
            <h3 class='feature-title'>Modelo por Imágenes</h3>
            <p class='feature-description'>Próximamente: Análisis de imágenes médicas.</p>
        </div>
        <div class='feature-container'>
            <h3 class='feature-title'>Información y Prevención</h3>
            <p class='feature-description'>Información crucial sobre prevención y manejo.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<h2 class='subtitle'>Comienza tu Evaluación</h2>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Selecciona una opción en el menú lateral para comenzar a utilizar nuestras herramientas de predicción.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def modelo_xgboost():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - XGBoost</h1>", unsafe_allow_html=True)
    
    model = load_model()
    
    st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<p class='white-subheader'>Datos Demográficos</p>", unsafe_allow_html=True)
            age = st.number_input("Edad", min_value=0, max_value=120, value=25, help="Edad del paciente en años")
            gender = st.selectbox("Género", ["Masculino", "Femenino"], help="Género del paciente")
            ever_married = st.selectbox("Estado Civil", ["Sí", "No"], help="¿El paciente ha estado alguna vez casado?")
        with col2:
            st.markdown("<p class='white-subheader'>Estilo de Vida</p>", unsafe_allow_html=True)
            work_type = st.selectbox("Tipo de Trabajo", VALID_WORK_TYPES, help="Sector laboral principal del paciente")
            smoking_status = st.selectbox("Estado de Fumador", VALID_SMOKING_STATUS, help="Historial de consumo de tabaco")
            avg_glucose_level = st.number_input("Nivel Promedio de Glucosa", min_value=0.0, max_value=300.0, value=100.0, help="Nivel promedio de glucosa en sangre")
        with col3:
            st.markdown("<p class='white-subheader'>Ubicación y Salud</p>", unsafe_allow_html=True)
            residence_type = st.selectbox("Tipo de Residencia", VALID_RESIDENCE_TYPES, help="Área de residencia del paciente")
            bmi = st.selectbox("Índice de Masa Corporal (BMI)", [1, 0], format_func=lambda x: "Sobrepeso" if x == 1 else "Normal", help="¿El paciente tiene sobrepeso? (1: Sí, 0: No)")
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene hipertensión diagnosticada?")
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene alguna enfermedad cardíaca diagnosticada?")
        
        predict_button = st.form_submit_button("Realizar Predicción", type="primary")

    if predict_button:
        try:
            input_data = pd.DataFrame({
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'smoking_status': [smoking_status]
            })
            validate_input_data(input_data)
            
            with st.spinner("Analizando factores de riesgo..."):
                processed_data = process_input_data(input_data)
                prediction = model.predict_proba(processed_data)
                risk_score = prediction[0][1]

            st.markdown("<h2 class='subtitle'>Resultados del Análisis</h2>", unsafe_allow_html=True)
            risk_status = "Alto Riesgo" if risk_score > 0.165 else "Bajo Riesgo"
            risk_color = "red" if risk_score > 0.165 else "green"
            st.markdown(f"""
                <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>
                    <h3 style='color: white; text-align: center;'>Estado: {risk_status}</h3>
                    <p style='color: white; text-align: center;'>Probabilidad de derrame cerebral: {risk_score:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(create_gauge_chart(risk_score, "Riesgo de Derrame Cerebral"))

            st.markdown("<h2 class='subtitle'>Interpretación de Resultados</h2>", unsafe_allow_html=True)
            if risk_score > 0.165:
                st.warning("El paciente presenta un riesgo elevado de sufrir un derrame cerebral. Se recomienda una evaluación médica inmediata y la implementación de medidas preventivas.")
            else:
                st.success("El paciente presenta un riesgo bajo de sufrir un derrame cerebral. Sin embargo, es importante mantener un estilo de vida saludable y realizar chequeos regulares.")

        except Exception as e:
            st.error(f"Error en el procesamiento: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

@st.cache_resource
def load_neural_model():
    """Cargar el modelo de red neuronal y el pipeline."""
    try:
        model = load_model("keras_model_nn.keras")
        pipeline = joblib.load("full_pipeline_nn.joblib")
        return model, pipeline
    except Exception as e:
        st.error(f"Error al cargar el modelo neuronal: {str(e)}")
        return None, None
      
def modelo_red_neuronal():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - Red Neuronal</h1>", unsafe_allow_html=True)
    
    model, pipeline = load_neural_model()
    
    if model is None or pipeline is None:
        st.error("No se pudo cargar el modelo de red neuronal. Por favor, verifica que los archivos del modelo estén presentes.")
        return
    
    st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)
    with st.form("patient_data_form_nn"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p class='white-subheader'>Datos Demográficos</p>", unsafe_allow_html=True)
            age = st.number_input("Edad", min_value=0, max_value=120, value=25, help="Edad del paciente en años")
            gender = st.selectbox("Género", ["Male", "Female"], help="Género del paciente")
            ever_married = st.selectbox("Estado Civil", ["Yes", "No"], help="¿El paciente ha estado alguna vez casado?")
        
        with col2:
            st.markdown("<p class='white-subheader'>Estilo de Vida</p>", unsafe_allow_html=True)
            work_type = st.selectbox("Tipo de Trabajo", VALID_WORK_TYPES, help="Sector laboral principal del paciente")
            smoking_status = st.selectbox("Estado de Fumador", VALID_SMOKING_STATUS, help="Historial de consumo de tabaco")
            avg_glucose_level = st.number_input("Nivel Promedio de Glucosa", min_value=0.0, max_value=300.0, value=100.0, help="Nivel promedio de glucosa en sangre")
        
        with col3:
            st.markdown("<p class='white-subheader'>Ubicación y Salud</p>", unsafe_allow_html=True)
            residence_type = st.selectbox("Tipo de Residencia", VALID_RESIDENCE_TYPES, help="Área de residencia del paciente")
            bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0, help="Índice de masa corporal del paciente")
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene hipertensión diagnosticada?")
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene alguna enfermedad cardíaca diagnosticada?")
        
        predict_button = st.form_submit_button("Realizar Predicción", type="primary")

    if predict_button:
        try:
            # Crear DataFrame con los datos del paciente
            patient_data = pd.DataFrame({
                'age': [age],
                'gender': [gender],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })
            
            # Procesar datos y realizar predicción
            with st.spinner("Analizando factores de riesgo con red neuronal..."):
                processed_data = pipeline.transform(patient_data)
                prediction = model.predict(processed_data)
                risk_score = prediction[0][0]  # Asumiendo que el modelo devuelve probabilidades

            st.markdown("<h2 class='subtitle'>Resultados del Análisis Neural</h2>", unsafe_allow_html=True)
            risk_status = "Alto Riesgo" if risk_score > 0.06 else "Bajo Riesgo"  # Umbral ajustado según el modelo neural
            risk_color = "red" if risk_score > 0.06 else "green"
            
            st.markdown(f"""
                <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>
                    <h3 style='color: white; text-align: center;'>Estado: {risk_status}</h3>
                    <p style='color: white; text-align: center;'>Probabilidad de derrame cerebral: {risk_score:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(create_gauge_chart(risk_score, "Riesgo de Derrame Cerebral (Red Neuronal)"))

            st.markdown("<h2 class='subtitle'>Interpretación de Resultados</h2>", unsafe_allow_html=True)
            if risk_score > 0.06:
                st.warning("""
                    El modelo de red neuronal indica un riesgo elevado de sufrir un derrame cerebral. 
                    Se recomienda una evaluación médica inmediata y la implementación de medidas preventivas.
                    Este resultado está basado en un análisis profundo de múltiples factores de riesgo.
                """)
            else:
                st.success("""
                    El modelo de red neuronal indica un riesgo bajo de sufrir un derrame cerebral. 
                    Sin embargo, es importante mantener un estilo de vida saludable y realizar chequeos regulares.
                    La red neuronal ha evaluado múltiples factores y sus interacciones para llegar a esta conclusión.
                """)

        except Exception as e:
            st.error(f"Error en el procesamiento del modelo neural: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)

def modelo_imagenes():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - Modelo por Imágenes</h1>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Esta sección está en desarrollo. Próximamente se implementará el modelo basado en imágenes.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def informacion_prevencion():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Información y Prevención de Ictus</h1>", unsafe_allow_html=True)
    
    st.markdown("<h2 class='subtitle'>¿Qué es un Ictus?</h2>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Un ictus, también conocido como accidente cerebrovascular (ACV), ocurre cuando el suministro de sangre a una parte del cerebro se interrumpe o se reduce, privando al tejido cerebral de oxígeno y nutrientes.</p>", unsafe_allow_html=True)

    st.markdown("<h2 class='subtitle'>Factores de Riesgo</h2>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Algunos factores de riesgo incluyen:</p>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: white;">
        <li>Hipertensión arterial</li>
        <li>Diabetes</li>
        <li>Colesterol alto</li>
        <li>Tabaquismo</li>
        <li>Obesidad</li>
        <li>Sedentarismo</li>
        <li>Edad avanzada</li>
        <li>Antecedentes familiares</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='subtitle'>Síntomas de Alerta</h2>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Es crucial reconocer los síntomas de un ictus para actuar rápidamente:</p>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: white;">
        <li>Debilidad o entumecimiento repentino en la cara, brazo o pierna, especialmente en un lado del cuerpo</li>
        <li>Confusión súbita o dificultad para hablar o entender</li>
        <li>Problemas repentinos de visión en uno o ambos ojos</li>
        <li>Dificultad repentina para caminar, mareo, pérdida de equilibrio o coordinación</li>
        <li>Dolor de cabeza severo sin causa conocida</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='subtitle'>Prevención</h2>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Algunas medidas para prevenir un ictus incluyen:</p>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: white;">
        <li>Controlar la presión arterial</li>
        <li>Mantener una dieta saludable y equilibrada</li>
        <li>Realizar actividad física regularmente</li>
        <li>Dejar de fumar</li>
        <li>Limitar el consumo de alcohol</li>
        <li>Controlar el estrés</li>
        <li>Realizar chequeos médicos regulares</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<h2 class='subtitle'>Actuación en caso de Ictus</h2>", unsafe_allow_html=True)
    st.markdown("<p class='description'>Si sospechas que alguien está sufriendo un ictus, recuerda el acrónimo FAST:</p>", unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: white;">
        <li><strong style="color: white;">F</strong>ace (Cara): Pide a la persona que sonría. ¿Un lado de la cara cae?</li>
        <li><strong style="color: white;">A</strong>rms (Brazos): Pide que levante ambos brazos. ¿Uno de los brazos cae?</li>
        <li><strong style="color: white;">S</strong>peech (Habla): Pide que repita una frase simple. ¿El habla es confusa o extraña?</li>
        <li><strong style="color: white;">T</strong>ime (Tiempo): Si observas cualquiera de estos signos, llama inmediatamente a emergencias.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<p class='description'>Recuerda, en caso de ictus, cada minuto cuenta. La atención médica inmediata puede marcar la diferencia entre la recuperación y la discapacidad permanente.</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Cargar el modelo
    model = load_model()

    # Menú desplegable en la barra lateral
    st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Menú</h1>", unsafe_allow_html=True)
    menu_option = st.sidebar.selectbox(
        "Seleccione una opción",
        ["Página Principal", "Modelo XGBoost", "Modelo red neuronal", "Modelo por imágenes", "Información y prevención Ictus"]
    )

    if menu_option == "Página Principal":
        main_page()
    elif menu_option == "Modelo XGBoost":
        modelo_xgboost()
    elif menu_option == "Modelo red neuronal":
        modelo_red_neuronal()
    elif menu_option == "Modelo por imágenes":
        modelo_imagenes()
    elif menu_option == "Información y prevención Ictus":
        informacion_prevencion()

if __name__ == "__main__":
    main()