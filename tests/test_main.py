import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
from app import (
    VALID_WORK_TYPES,
    VALID_SMOKING_STATUS, 
    VALID_RESIDENCE_TYPES,
    load_models,
    create_sidebar_menu,
    handle_menu_selection,
    setup_page
)

def test_valid_constants():
    """Test que las constantes definidas son válidas y completas"""
    # Test work types
    assert len(VALID_WORK_TYPES) == 4
    assert "Private" in VALID_WORK_TYPES
    assert "Self-employed" in VALID_WORK_TYPES
    assert "Govt_job" in VALID_WORK_TYPES
    assert "children" in VALID_WORK_TYPES
    
    # Test smoking status
    assert len(VALID_SMOKING_STATUS) == 4
    assert "never smoked" in VALID_SMOKING_STATUS
    assert "formerly smoked" in VALID_SMOKING_STATUS
    assert "smokes" in VALID_SMOKING_STATUS
    assert "Unknown" in VALID_SMOKING_STATUS
    
    # Test residence types
    assert len(VALID_RESIDENCE_TYPES) == 2
    assert "Urban" in VALID_RESIDENCE_TYPES
    assert "Rural" in VALID_RESIDENCE_TYPES

@patch('app.load_model_image')
def test_load_models(mock_load_model_image):
    """Test que la carga de modelos funciona correctamente"""
    # Configure el mock
    mock_model = MagicMock()
    mock_load_model_image.return_value = mock_model
    
    # Ejecutar la función
    models = load_models()
    
    # Verificar que se llamó a load_model_image con la ruta correcta
    mock_load_model_image.assert_called_once_with('models/vision_stroke_95.pth')
    
    # Verificar que el diccionario de modelos contiene la clave correcta
    assert 'vision' in models
    assert models['vision'] == mock_model

@patch('streamlit.sidebar.selectbox')
def test_create_sidebar_menu(mock_selectbox):
    """Test que el menú lateral se crea con las opciones correctas"""
    # Configure el mock
    mock_selectbox.return_value = "Página Principal"
    
    # Ejecutar la función
    result = create_sidebar_menu()
    
    # Verificar que se llamó a selectbox con las opciones correctas
    mock_selectbox.assert_called_once()
    options = mock_selectbox.call_args[0][1]
    assert len(options) == 5
    assert "Página Principal" in options
    assert "Modelo XGBoost" in options
    assert "Modelo red neuronal" in options
    assert "Modelo por imágenes" in options
    assert "Información y prevención Ictus" in options
    
    # Verificar el valor retornado
    assert result == "Página Principal"

@patch('main_page')
@patch('modelo_xgboost')
@patch('StrokePredictor')
@patch('modelo_imagenes')
@patch('mostrar_informacion_prevencion')
def test_handle_menu_selection(mock_info, mock_imagenes, 
                             mock_predictor, mock_xgboost, 
                             mock_main):
    """Test que handle_menu_selection llama a la función correcta según la opción"""
    models = {'vision': MagicMock()}
    
    # Test Página Principal
    handle_menu_selection("Página Principal", models)
    mock_main.assert_called_once()
    
    # Test Modelo XGBoost
    handle_menu_selection("Modelo XGBoost", models)
    mock_xgboost.assert_called_once()
    
    # Test Modelo red neuronal
    predictor_instance = MagicMock()
    mock_predictor.return_value = predictor_instance
    handle_menu_selection("Modelo red neuronal", models)
    mock_predictor.assert_called_once()
    predictor_instance.mostrar_prediccion_derrame.assert_called_once()
    
    # Test Modelo por imágenes
    handle_menu_selection("Modelo por imágenes", models)
    mock_imagenes.assert_called_once_with(models['vision'])
    
    # Test Información y prevención
    handle_menu_selection("Información y prevención Ictus", models)
    mock_info.assert_called_once()

@patch('streamlit.set_page_config')
@patch('streamlit.markdown')
def test_setup_page(mock_markdown, mock_set_page_config):
    """Test que la configuración de la página se realiza correctamente"""
    setup_page()
    
    # Verificar que se llamó a set_page_config con los parámetros correctos
    mock_set_page_config.assert_called_once_with(
        page_title="Predictor de Riesgo de Derrame Cerebral",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    from unittest import mock
    mock_markdown.assert_called_with(mock.ANY, unsafe_allow_html=True)

    css_call = mock_markdown.call_args[0][0]
    assert "<style>" in css_call

if __name__ == '__main__':
    pytest.main([__file__])