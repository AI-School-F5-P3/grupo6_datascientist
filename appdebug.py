import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBClassifier
import numpy as np

@st.cache_resource
def load_model_and_scaler():
    try:
        scaler = joblib.load('scaler.pkl')
        model = XGBClassifier()
        model.load_model("xgboost_stroke_model_final.bin")
        return scaler, model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None, None

def process_input_data(raw_data):
    """Procesar y preparar los datos de entrada, incluyendo todas las variables."""
    processed_data = pd.DataFrame(index=[0])
    
    # Agregar las características
    processed_data['gender_Female'] = 1 if raw_data['gender'].iloc[0] == "Female" else 0
    processed_data['gender_Male'] = 1 if raw_data['gender'].iloc[0] == "Male" else 0
    processed_data['age'] = raw_data['age'].iloc[0]
    processed_data['hypertension'] = raw_data['hypertension'].iloc[0]
    processed_data['heart_disease'] = raw_data['heart_disease'].iloc[0]
    processed_data['ever_married_No'] = 1 if raw_data['ever_married'].iloc[0] == "No" else 0
    processed_data['ever_married_Yes'] = 1 if raw_data['ever_married'].iloc[0] == "Yes" else 0
    processed_data['work_type_Govt_job'] = 1 if raw_data['work_type'].iloc[0] == "Govt_job" else 0
    processed_data['work_type_Private'] = 1 if raw_data['work_type'].iloc[0] == "Private" else 0
    processed_data['work_type_Self-employed'] = 1 if raw_data['work_type'].iloc[0] == "Self-employed" else 0
    processed_data['work_type_children'] = 1 if raw_data['work_type'].iloc[0] == "children" else 0
    processed_data['Residence_type_Rural'] = 1 if raw_data['Residence_type'].iloc[0] == "Rural" else 0
    processed_data['Residence_type_Urban'] = 1 if raw_data['Residence_type'].iloc[0] == "Urban" else 0
    processed_data['avg_glucose_level'] = raw_data['avg_glucose_level'].iloc[0]
    processed_data['bmi'] = raw_data['bmi'].iloc[0]
    processed_data['smoking_status_Unknown'] = 1 if raw_data['smoking_status'].iloc[0] == "Unknown" else 0
    processed_data['smoking_status_formerly smoked'] = 1 if raw_data['smoking_status'].iloc[0] == "formerly smoked" else 0
    processed_data['smoking_status_never smoked'] = 1 if raw_data['smoking_status'].iloc[0] == "never smoked" else 0
    processed_data['smoking_status_smokes'] = 1 if raw_data['smoking_status'].iloc[0] == "smokes" else 0
    
    return processed_data

def main():
    st.title("🏥 Predicción de Riesgo de Derrame Cerebral")
    
    scaler, loaded_model = load_model_and_scaler()
    if scaler is None or loaded_model is None:
        return
    
    # Crear datos de prueba para caso de alto riesgo
    test_high_risk = {
        'gender': ["Male"],
        'age': [75],
        'hypertension': [1],
        'heart_disease': [1],
        'ever_married': ["Yes"],
        'work_type': ["Self-employed"],
        'Residence_type': ["Rural"],
        'avg_glucose_level': [200],
        'bmi': [28],
        'smoking_status': ["smokes"]
    }

    # Crear datos de prueba para caso de bajo riesgo
    test_low_risk = {
        'gender': ["Female"],
        'age': [25],
        'hypertension': [0],
        'heart_disease': [0],
        'ever_married': ["No"],
        'work_type': ["Private"],
        'Residence_type': ["Urban"],
        'avg_glucose_level': [90],
        'bmi': [22],
        'smoking_status': ["never smoked"]
    }

    test_case = st.radio(
        "Seleccionar modo de entrada:",
        ["Manual", "Caso de Alto Riesgo (Test)", "Caso de Bajo Riesgo (Test)"]
    )
    
    if test_case == "Caso de Alto Riesgo (Test)":
        input_data = pd.DataFrame(test_high_risk)
    elif test_case == "Caso de Bajo Riesgo (Test)":
        input_data = pd.DataFrame(test_low_risk)
    else:
        # Entrada manual
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Género", ["Male", "Female"])
            age = st.number_input("Edad", min_value=1, max_value=120, value=25)
            hypertension = st.selectbox("Hipertensión", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            heart_disease = st.selectbox("Enfermedad Cardíaca", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            avg_glucose_level = st.number_input("Nivel de Glucosa Promedio", min_value=0.0, value=100.0)
            bmi = st.number_input("Índice de Masa Corporal (IMC)", min_value=0.0, value=25.0)
            ever_married = st.selectbox("¿Alguna vez casado?", ["Yes", "No"])

        with col2:
            work_type = st.selectbox("Tipo de Trabajo", ["Private", "Self-employed", "Govt_job", "children"])
            residence_type = st.selectbox("Tipo de Residencia", ["Urban", "Rural"])
            smoking_status = st.selectbox("Estado de Fumador", ["never smoked", "formerly smoked", "smokes", "Unknown"])

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

    if st.button("Realizar Predicción", type="primary"):
        with st.spinner("Procesando datos..."):
            # Procesar los datos
            processed_data = process_input_data(input_data)
            
            # Escalar las variables necesarias
            features_to_scale = [
                'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
                'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes',
                'work_type_Govt_job', 'work_type_Private', 'work_type_Self-employed', 'work_type_children',
                'Residence_type_Rural', 'Residence_type_Urban', 'smoking_status_Unknown',
                'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
            ]
            processed_data_scaled = scaler.transform(processed_data[features_to_scale])
            
            # Convertir a DataFrame
            input_data_scaled = pd.DataFrame(processed_data_scaled, columns=features_to_scale)

            # Seleccionar solo las columnas clave para la predicción
            selected_features = [
                "age", "smoking_status_never smoked", "hypertension",
                "work_type_Private", "Residence_type_Rural", "heart_disease"
            ]
            prediction_input = input_data_scaled[selected_features]
            
            # Realizar predicción
            prediction_prob = float(loaded_model.predict_proba(prediction_input)[:, 1][0])
            threshold = 0.165  # Umbral ajustado durante el entrenamiento
            
            # Mostrar resultados
            st.subheader("Resultados del Análisis")
            risk_percentage = prediction_prob * 100
            
            col1, col2, _ = st.columns([1, 2, 1])
            with col2:
                st.progress(float(prediction_prob))
                st.write(f"Probabilidad de riesgo: {risk_percentage:.1f}%")
            
            if prediction_prob >= threshold:
                st.error(f"⚠️ Riesgo Alto de Derrame Cerebral ({risk_percentage:.1f}%)")
                st.markdown("""### Recomendaciones:
                    - Consulte a un médico lo antes posible
                    - Monitoree su presión arterial regularmente
                    - Mantenga un estilo de vida saludable
                """)
            else:
                st.success(f"✅ Riesgo Bajo de Derrame Cerebral ({risk_percentage:.1f}%)")
                st.markdown("""### Recomendaciones:
                    - Continúe manteniendo hábitos saludables
                    - Realice chequeos médicos regulares
                    - Mantenga una dieta equilibrada
                """)
                
if __name__ == "__main__":
    main()