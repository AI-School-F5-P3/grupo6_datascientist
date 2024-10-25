import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any


# Opciones válidas para los campos select
VALID_WORK_TYPES = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
VALID_SMOKING_STATUS = ["formerly smoked", "never smoked", "smokes", "Unknown"]
VALID_RESIDENCE_TYPES = ["Urban", "Rural"]


class StrokePredictor:
    """
    Clase para manejar la predicción de riesgo de derrame cerebral usando una red neuronal.
    Proporciona una interfaz de usuario para introducir datos del paciente y visualizar resultados.
    """

    @staticmethod
    @st.cache_resource
    def _load_neural_model() -> Tuple[Optional[Any], Optional[Any]]:
        """
        Carga el modelo de red neuronal y el pipeline.

        Returns:
            Tuple[Optional[Any], Optional[Any]]: Modelo neural y pipeline de preprocesamiento
        """
        try:
            model_nn = tf.keras.models.load_model("models/keras_model_nn.keras")
            pipeline_nn = joblib.load("models/full_pipeline_nn.joblib")
            return model_nn, pipeline_nn
        except Exception as e:
            st.error(f"Error al cargar el modelo neuronal: {str(e)}")
            return None, None

    def _load_models(self) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Método interno para cargar los modelos usando la función cacheada.

        Returns:
            Tuple[Optional[Any], Optional[Any]]: Modelo neural y pipeline de preprocesamiento
        """
        return self._load_neural_model()

    def __init__(self):
        """Inicializa el predictor con estilos y configuraciones."""
        self.styles = """
            <style>
            .white-label {
                color: white !important;
                font-weight: 500;
                margin-bottom: 0.5rem;
            }
            .white-subheader {
                color: white !important;
                font-weight: 600;
                font-size: 1.1rem;
                margin-bottom: 1rem;
            }
            .result-container {
                padding: 1.5rem;
                border-radius: 10px;
                margin: 1rem 0;
                text-align: center;
            }
            .high-risk {
                background-color: rgba(255, 0, 0, 0.7);
            }
            .low-risk {
                background-color: rgba(0, 128, 0, 0.7);
            }
            </style>
        """

    def _create_form_section(self, column, title: str, fields: Dict[str, Dict[str, Any]]):
        """
        Crea una sección del formulario con campos específicos.

        Args:
            column: Columna de Streamlit donde se mostrarán los campos
            title (str): Título de la sección
            fields (Dict): Diccionario con la configuración de los campos
        """
        with column:
            st.markdown(f"<p class='white-subheader'>{title}</p>", unsafe_allow_html=True)
            for field_name, config in fields.items():
                st.markdown(f"<p class='white-label'>{config['label']}</p>", unsafe_allow_html=True)
                if config['type'] == 'number':
                    value = st.number_input(
                        field_name,
                        min_value=config['min'],
                        max_value=config['max'],
                        value=config['default'],
                        help=config['help'],
                        label_visibility="collapsed"
                    )
                elif config['type'] == 'select':
                    value = st.selectbox(
                        field_name,
                        config['options'],
                        help=config['help'],
                        format_func=config.get('format_func', lambda x: x),
                        label_visibility="collapsed"
                    )
                config['value'] = value

    def create_gauge_chart(self, risk_score: float, title: str) -> go.Figure:
        """
        Crea un gráfico de indicador tipo gauge para visualizar el riesgo.

        Args:
            risk_score (float): Puntuación de riesgo entre 0 y 1
            title (str): Título del gráfico

        Returns:
            go.Figure: Figura de Plotly con el gráfico gauge
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score * 100,  # Convierte el riesgo a porcentaje
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 20], 'color': "green"},
                    {'range': [20, 40], 'color': "yellow"},
                    {'range': [40, 60], 'color': "orange"},
                    {'range': [60, 80], 'color': "red"},
                    {'range': [80, 100], 'color': "darkred"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score * 100
                }
            }
        ))
        fig.update_layout(height=250)
        return fig

    def _show_results(self, risk_score: float):
        """
        Muestra los resultados de la predicción.

        Args:
            risk_score (float): Puntuación de riesgo calculada
        """
        st.markdown("<h2 class='subtitle'>Resultados del Análisis Neural</h2>", unsafe_allow_html=True)

        risk_status = "Alto Riesgo" if risk_score > 0.06 else "Bajo Riesgo"
        risk_class = "high-risk" if risk_score > 0.06 else "low-risk"

        st.markdown(f"""
            <div class='result-container {risk_class}'>
                <h3 style='color: white;'>Estado: {risk_status}</h3>
                <p style='color: white;'>Probabilidad de derrame cerebral: {risk_score:.2%}</p>
            </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(self.create_gauge_chart(risk_score, "Riesgo de Derrame Cerebral (Red Neuronal)"))

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

    def predict(self):
        """Método principal que maneja toda la interfaz de predicción."""
        st.markdown(self.styles, unsafe_allow_html=True)
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - Red Neuronal</h1>", unsafe_allow_html=True)

        model_nn, pipeline_nn = self._load_models()
        if model_nn is None or pipeline_nn is None:
            st.error("No se pudo cargar el modelo de red neuronal. Por favor, verifica que los archivos del modelo estén presentes.")
            return

        st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)

        with st.form("patient_data_form_nn"):
            col1, col2, col3 = st.columns(3)

            # Configuración de campos para cada sección
            demographic_fields = {
                'age': {'type': 'number', 'label': 'Edad', 'min': 0, 'max': 120, 'default': 25, 'help': 'Edad del paciente en años'},
                'gender': {'type': 'select', 'label': 'Género', 'options': ["Male", "Female"], 'help': 'Género del paciente'},
                'ever_married': {'type': 'select', 'label': 'Estado Civil', 'options': ["Yes", "No"], 'help': '¿El paciente ha estado alguna vez casado?'}
            }

            lifestyle_fields = {
                'work_type': {'type': 'select', 'label': 'Tipo de Trabajo', 'options': VALID_WORK_TYPES, 'help': 'Sector laboral principal del paciente'},
                'smoking_status': {'type': 'select', 'label': 'Estado de Fumador', 'options': VALID_SMOKING_STATUS, 'help': 'Historial de consumo de tabaco'},
                'avg_glucose_level': {'type': 'number', 'label': 'Nivel Promedio de Glucosa', 'min': 0.0, 'max': 300.0, 'default': 100.0, 'help': 'Nivel promedio de glucosa en sangre'}
            }

            health_fields = {
                'residence_type': {'type': 'select', 'label': 'Tipo de Residencia', 'options': VALID_RESIDENCE_TYPES, 'help': 'Área de residencia del paciente'},
                'bmi': {'type': 'number', 'label': 'Índice de Masa Corporal (BMI)', 'min': 10.0, 'max': 50.0, 'default': 25.0, 'help': 'Índice de masa corporal del paciente'},
                'hypertension': {'type': 'select', 'label': 'Hipertensión', 'options': [0, 1], 'format_func': lambda x: "Sí" if x == 1 else "No", 'help': '¿El paciente tiene hipertensión diagnosticada?'},
                'heart_disease': {'type': 'select', 'label': 'Enfermedad Cardíaca', 'options': [0, 1], 'format_func': lambda x: "Sí" if x == 1 else "No", 'help': '¿El paciente tiene alguna enfermedad cardíaca diagnosticada?'}
            }

            self._create_form_section(col1, "Datos Demográficos", demographic_fields)
            self._create_form_section(col2, "Estilo de Vida", lifestyle_fields)
            self._create_form_section(col3, "Ubicación y Salud", health_fields)

            # Botón de envío del formulario
            predict_button = st.form_submit_button("Realizar Predicción")

        if predict_button:
            try:
                # Recolectar valores de los campos
                input_data = {field: config['value'] for field, config in {**demographic_fields, **lifestyle_fields, **health_fields}.items()}

                patient_data = pd.DataFrame([input_data])  # Transforma el diccionario a un DataFrame

                with st.spinner("Analizando factores de riesgo con red neuronal..."):
                    processed_data = pipeline_nn.transform(patient_data)
                    prediction = model_nn.predict(processed_data)
                    risk_score = prediction[0][0]

                self._show_results(risk_score)

            except Exception as e:
                st.error(f"Error en el procesamiento del modelo neural: {str(e)}")

        st.markdown("</div>", unsafe_allow_html=True)

    def mostrar_prediccion_derrame(self):
        """Muestra la interfaz del predictor de derrame cerebral."""
        self.predict()  # Llama a la función predict() para mostrar la interfaz
        
if __name__ == "__main__":
    predictor = StrokePredictor()
    predictor.mostrar_prediccion_derrame()
    
