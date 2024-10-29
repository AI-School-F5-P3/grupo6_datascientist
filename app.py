import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
import plotly.graph_objects as go
import numpy as np
import tensorflow
from stroke_info import mostrar_informacion_prevencion
from vision_aux import load_model_image
from db_aux import *
from stroke_neuronal import StrokePredictor
from stroke_xg import load_model, modelo_xgboost
from stroke_pictures import modelo_imagenes


# Constantes
VALID_WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children"]
VALID_SMOKING_STATUS = ["never smoked", "formerly smoked", "smokes", "Unknown"]
VALID_RESIDENCE_TYPES = ["Urban", "Rural"]
VISION_MODEL_PATH = 'models/vision_stroke_95.pth'

# Estilos CSS
CSS_STYLES = """
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
    .stSelectbox > div > label, 
    .stNumberInput > div > label,
    .stSlider > div > label {
        color: white !important;
    }
    .stSelectbox > div > div > div,
    .stNumberInput > div > div > input {
        color: black !important;
        background-color: white !important;
    }
    .stSelectbox > div > div > ul > li {
        color: black !important;
    }
    .white-subheader {
        color: white;
        font-size: 1.1em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
"""

# Contenido HTML para las características
FEATURES_HTML = {
    'left': """
    <div class='feature-container'>
        <h3 class='feature-title'>Modelo XGBoost</h3>
        <p class='feature-description'>Predicciones precisas con aprendizaje automático avanzado.</p>
    </div>
    <div class='feature-container'>
        <h3 class='feature-title'>Modelo de Red Neuronal</h3>
        <p class='feature-description'>Próximamente: Evaluación con redes neuronales artificiales.</p>
    </div>
    """,
    'right': """
    <div class='feature-container'>
        <h3 class='feature-title'>Modelo por Imágenes</h3>
        <p class='feature-description'>Próximamente: Análisis de imágenes médicas.</p>
    </div>
    <div class='feature-container'>
        <h3 class='feature-title'>Información y Prevención</h3>
        <p class='feature-description'>Información crucial sobre prevención y manejo.</p>
    </div>
    """
}

def setup_page():
    """Configura la página inicial de Streamlit"""
    st.set_page_config(
        page_title="Predictor de Riesgo de Derrame Cerebral",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown(CSS_STYLES, unsafe_allow_html=True)

def load_models():
    """Carga todos los modelos necesarios"""
    return {
        'vision': load_model_image(VISION_MODEL_PATH)
    }

def main_page():
    """Renderiza la página principal de la aplicación"""
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Bienvenido al Predictor de Riesgo de Derrame Cerebral</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subtitle'>HOSPITAL F5</h2>", unsafe_allow_html=True)
    
    # Logo
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("logo.png", use_column_width=True)
    
    # Descripción
    st.markdown(
        "<p class='description'>Nuestra aplicación utiliza tecnología de vanguardia para evaluar "
        "el riesgo de derrame cerebral basado en factores de salud individuales.</p>",
        unsafe_allow_html=True
    )
    
    # Características
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(FEATURES_HTML['left'], unsafe_allow_html=True)
    with col2:
        st.markdown(FEATURES_HTML['right'], unsafe_allow_html=True)
    
    # Sección final
    st.markdown("<h2 class='subtitle'>Comienza tu Evaluación</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='description'>Selecciona una opción en el menú lateral para comenzar "
        "a utilizar nuestras herramientas de predicción.</p>",
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

def create_sidebar_menu():
    """Crea y retorna la selección del menú lateral"""
    st.sidebar.markdown("<h1 style='text-align: center; color: black;'>Menú</h1>", unsafe_allow_html=True)
    return st.sidebar.selectbox(
        "Seleccione una opción",
        ["Página Principal", "Modelo XGBoost", "Modelo red neuronal", 
         "Modelo por imágenes", "Información y prevención Ictus"]
    )

def handle_menu_selection(menu_option, models):
    """Maneja la selección del menú y muestra la página correspondiente"""
    if menu_option == "Página Principal":
        main_page()
    elif menu_option == "Modelo XGBoost":
        modelo_xgboost()
    elif menu_option == "Modelo red neuronal":
        predictor = StrokePredictor()
        predictor.mostrar_prediccion_derrame()
    elif menu_option == "Modelo por imágenes":
        modelo_imagenes(models['vision'])
    elif menu_option == "Información y prevención Ictus":
        mostrar_informacion_prevencion()

def main():
    """Función principal que ejecuta la aplicación"""
    setup_page()
    models = load_models()
    menu_option = create_sidebar_menu()
    handle_menu_selection(menu_option, models)

if __name__ == "__main__":
    main()