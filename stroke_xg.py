import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

VALID_WORK_TYPES = ['Private', 'Self-employed', 'Government', 'Children', 'Never worked']
VALID_SMOKING_STATUS = ['never smoked', 'smokes', 'formerly smoked']
VALID_RESIDENCE_TYPES = ['Urban', 'Rural']

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

def modelo_xgboost(model):
    """Función para la interfaz de predicción XGBoost."""
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Predictor de Riesgo de Derrame Cerebral - XGBoost</h1>", unsafe_allow_html=True)

    st.markdown("<h2 class='subtitle'>📝 Información del Paciente</h2>", unsafe_allow_html=True)
    with st.form("patient_data_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Edad", min_value=0, max_value=120, value=25)
            gender = st.selectbox("Género", ["Masculino", "Femenino"])
            ever_married = st.selectbox("Estado Civil", ["Sí", "No"])

        with col2:
            work_type = st.selectbox("Tipo de Trabajo", VALID_WORK_TYPES)
            smoking_status = st.selectbox("Estado de Fumador", VALID_SMOKING_STATUS)
            avg_glucose_level = st.number_input("Nivel Promedio de Glucosa", min_value=0.0, max_value=300.0, value=100.0)

        with col3:
            residence_type = st.selectbox("Tipo de Residencia", VALID_RESIDENCE_TYPES)
            bmi = st.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=50.0, value=25.0, step=0.1, format="%.1f", help="Introduce el índice de masa corporal entre 10 y 50")
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")

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

            # Aquí puedes agregar más visualizaciones si lo deseas

            st.markdown("<h2 class='subtitle'>Interpretación de Resultados</h2>", unsafe_allow_html=True)
            if risk_score > 0.165:
                st.warning("El paciente presenta un riesgo elevado de sufrir un derrame cerebral. Se recomienda una evaluación médica inmediata.")
            else:
                st.success("El paciente presenta un riesgo bajo de sufrir un derrame cerebral. Mantener un estilo de vida saludable es importante.")

        except Exception as e:
            st.error(f"Error en el procesamiento: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)
