import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import plotly.graph_objects as go
import numpy as np
from stroke_info import mostrar_informacion_prevencion
from vision_aux import *
import joblib
import tensorflow
from stroke_neuronal import StrokePredictor
import streamlit as st
from stroke_xg import load_model, modelo_xgboost



vision_model_path = 'models/vision_stroke_95.pth'
vision_model = load_model_image(vision_model_path)


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
        model.load_model("models/xgboost_stroke_model_final.bin")
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


def modelo_imagenes():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - Modelo por Imágenes</h1>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Cargue un Scan para analizar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Open the uploaded image directly
            image = Image.open(uploaded_file)
            
            # Define the path where the image will be saved temporarily
            temp_file_path = f"./temp_image_{uploaded_file.name}"
            
            # Save the image with the same format as uploaded
            image.save(temp_file_path, format=image.format)

            # Preprocess the image using the saved path
            image_tensor = preprocess_image(temp_file_path)
            
            # Center the image and button using columns
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                # Display the centered image
                st.image(temp_file_path, caption='Scan cargado', use_column_width=False, width=400)
                

                st.markdown(
                """
                <style>
                .stButton > button {
                    width: 115px;
                    margin-left: 150px;  /* Adjust this value to move button left/right */
                    margin-top: 10px;   /* Add some space between image and button */
                    height: 50px;       /* Set button height */
                    font-size: 16px;    /* Set font size */
                }
                </style>
                """, 
                unsafe_allow_html=True
                )

                # Center the button
                if st.button('Predicción'):
                    predicted_class, probability = predict_image(vision_model, image_tensor)

                    st.write("")
                    st.write("")
                    print_outcome(predicted_class, probability)




            # # Display the uploaded image
            # st.image(temp_file_path, caption='Scan cargado', use_column_width=False,  width=250)

            # if st.button('Predicción'):
            #     predicted_class, probability = predict(vision_model, image_tensor)
            #     print_outcome(predicted_class, probability)

            # Clean up: remove the temporary file after processing
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as e:
            print_error(str(e))

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
        modelo_xgboost(model)
    elif menu_option == "Modelo red neuronal":
        predictor = StrokePredictor()
        predictor.mostrar_prediccion_derrame()  
    elif menu_option == "Modelo por imágenes":
        modelo_imagenes()
    elif menu_option == "Información y prevención Ictus":
        mostrar_informacion_prevencion()

if __name__ == "__main__":
    main()
    
    