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

# def modelo_xgboost():
#     st.markdown("<div class='main-container'>", unsafe_allow_html=True)
#     st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - XGBoost</h1>", unsafe_allow_html=True)
    
#     model = load_model()
    
#     st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)
#     with st.form("patient_data_form"):
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.markdown("<p class='white-subheader'>Datos Demográficos</p>", unsafe_allow_html=True)
#             age = st.number_input("Edad", min_value=0, max_value=120, value=25, help="Edad del paciente en años")
#             gender = st.selectbox("Género", ["Masculino", "Femenino"], help="Género del paciente")
#             ever_married = st.selectbox("Estado Civil", ["Sí", "No"], help="¿El paciente ha estado alguna vez casado?")
#         with col2:
#             st.markdown("<p class='white-subheader'>Estilo de Vida</p>", unsafe_allow_html=True)
#             work_type = st.selectbox("Tipo de Trabajo", VALID_WORK_TYPES, help="Sector laboral principal del paciente")
#             smoking_status = st.selectbox("Estado de Fumador", VALID_SMOKING_STATUS, help="Historial de consumo de tabaco")
#             avg_glucose_level = st.number_input("Nivel Promedio de Glucosa", min_value=0.0, max_value=300.0, value=100.0, help="Nivel promedio de glucosa en sangre")
#         with col3:
#             st.markdown("<p class='white-subheader'>Ubicación y Salud</p>", unsafe_allow_html=True)
#             residence_type = st.selectbox("Tipo de Residencia", VALID_RESIDENCE_TYPES, help="Área de residencia del paciente")
#             bmi = st.selectbox("Índice de Masa Corporal (BMI)", [1, 0], format_func=lambda x: "Sobrepeso" if x == 1 else "Normal", help="¿El paciente tiene sobrepeso? (1: Sí, 0: No)")
#             hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene hipertensión diagnosticada?")
#             heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene alguna enfermedad cardíaca diagnosticada?")
        
#         predict_button = st.form_submit_button("Realizar Predicción", type="primary")

#     if predict_button:
#         try:
#             input_data = pd.DataFrame({
#                 'age': [age],
#                 'hypertension': [hypertension],
#                 'heart_disease': [heart_disease],
#                 'work_type': [work_type],
#                 'Residence_type': [residence_type],
#                 'smoking_status': [smoking_status]
#             })
#             validate_input_data(input_data)
            
#             with st.spinner("Analizando factores de riesgo..."):
#                 processed_data = process_input_data(input_data)
#                 prediction = model.predict_proba(processed_data)
#                 risk_score = prediction[0][1]

#             st.markdown("<h2 class='subtitle'>Resultados del Análisis</h2>", unsafe_allow_html=True)
#             risk_status = "Alto Riesgo" if risk_score > 0.165 else "Bajo Riesgo"
#             risk_color = "red" if risk_score > 0.165 else "green"
#             st.markdown(f"""
#                 <div style='background-color: {risk_color}; padding: 10px; border-radius: 5px;'>
#                     <h3 style='color: white; text-align: center;'>Estado: {risk_status}</h3>
#                     <p style='color: white; text-align: center;'>Probabilidad de derrame cerebral: {risk_score:.2%}</p>
#                 </div>
#             """, unsafe_allow_html=True)

#             st.plotly_chart(create_gauge_chart(risk_score, "Riesgo de Derrame Cerebral"))

#             st.markdown("<h2 class='subtitle'>Interpretación de Resultados</h2>", unsafe_allow_html=True)
#             if risk_score > 0.165:
#                 st.warning("El paciente presenta un riesgo elevado de sufrir un derrame cerebral. Se recomienda una evaluación médica inmediata y la implementación de medidas preventivas.")
#             else:
#                 st.success("El paciente presenta un riesgo bajo de sufrir un derrame cerebral. Sin embargo, es importante mantener un estilo de vida saludable y realizar chequeos regulares.")

#         except Exception as e:
#             st.error(f"Error en el procesamiento: {str(e)}")
    
#     st.markdown("</div>", unsafe_allow_html=True)

def modelo_xgboost():
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
    
    st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p class='white-subheader'>Datos Demográficos</p>", unsafe_allow_html=True)
            st.markdown("<p class='white-label'>Edad</p>", unsafe_allow_html=True)
            age = st.number_input("Edad", min_value=0, max_value=120, value=25, help="Edad del paciente en años", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Género</p>", unsafe_allow_html=True)
            gender = st.selectbox("Género", ["Masculino", "Femenino"], help="Género del paciente", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Estado Civil</p>", unsafe_allow_html=True)
            ever_married = st.selectbox("Estado Civil", ["Sí", "No"], help="¿El paciente ha estado alguna vez casado?", label_visibility="collapsed")
        
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
            bmi = st.selectbox("Índice de Masa Corporal (BMI)", [1, 0], format_func=lambda x: "Sobrepeso" if x == 1 else "Normal", help="¿El paciente tiene sobrepeso? (1: Sí, 0: No)", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Hipertensión</p>", unsafe_allow_html=True)
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene hipertensión diagnosticada?", label_visibility="collapsed")
            st.markdown("<p class='white-label'>Enfermedad Cardíaca</p>", unsafe_allow_html=True)
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No", help="¿El paciente tiene alguna enfermedad cardíaca diagnosticada?", label_visibility="collapsed")
        
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
        modelo_xgboost()
    elif menu_option == "Modelo red neuronal":
        predictor = StrokePredictor()
        predictor.mostrar_prediccion_derrame()  
    elif menu_option == "Modelo por imágenes":
        modelo_imagenes()
    elif menu_option == "Información y prevención Ictus":
        mostrar_informacion_prevencion()

if __name__ == "__main__":
    main()
    
    