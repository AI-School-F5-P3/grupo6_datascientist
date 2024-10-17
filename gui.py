

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load the saved transformer and model
loaded_transformer = joblib.load('models/transformer.joblib')
loaded_model = load_model('models/stroke_model.keras')

def predict_stroke(input_data):
    columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
               'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
    df = pd.DataFrame([input_data], columns=columns)

    selected_columns = ['age', 'avg_glucose_level', 'bmi', 'work_type', 'smoking_status']
    df_selected = df[selected_columns]

    preprocessed_data = loaded_transformer.transform(df_selected)

    prediction_prob = loaded_model.predict(preprocessed_data)
    prediction = 1 if prediction_prob[0][0] >= 0.1 else 0

    return prediction, prediction_prob[0][0]

def main():
    st.title("Stroke Prediction App")

    # Create two columns
    col1, col2 = st.columns(2)

    # User input - Column 1
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        age = st.number_input("Age", min_value=0.0, max_value=120.0, value=30.0)
        hypertension = st.selectbox("Hypertension", ['Yes', 'No'])
        heart_disease = st.selectbox("Heart Disease", ['Yes', 'No'])
        ever_married = st.selectbox("Ever Married", ['Yes', 'No'])

    # User input - Column 2
    with col2:
        work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Govt_job', 'children'])
        residence_type = st.selectbox("Residence Type", ['Urban', 'Rural'])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
        smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

    # Prediction button
    if st.button("Predict Stroke Risk"):
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension':  1 if hypertension == 'Yes' else 0,
            'heart_disease': 1 if heart_disease == 'Yes' else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        prediction, probability = predict_stroke(input_data)

        st.subheader("Prediction Results:")
        if prediction == 1:
            st.warning(f"High risk of stroke. Probability: {probability:.2%}")
        else:
            st.success(f"Low risk of stroke. Probability: {probability:.2%}")

        

if __name__ == "__main__":
    main()