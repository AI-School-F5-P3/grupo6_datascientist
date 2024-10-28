import streamlit as st
from PIL import Image
import os
from vision_aux import preprocess_image, predict_image, print_outcome, print_error

def modelo_imagenes(vision_model):
    """
    Función que implementa la interfaz de predicción basada en imágenes.
    
    Args:
        vision_model: Modelo de visión por computadora pre-entrenado para la detección de ictus
    """
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

            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        except Exception as e:
            print_error(str(e))