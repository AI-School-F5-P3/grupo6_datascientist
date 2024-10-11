# config.py

# Parámetros de la aplicación
APP_TITLE = "Stroke Prediction Model"
APP_DESCRIPTION = """
Esta aplicación predice la probabilidad de sufrir un derrame cerebral
basado en un modelo de Machine Learning.
"""

# Otros parámetros del proyecto
MODEL_PATH = 'models/final_model.joblib'
TRANSFORMER_PATH = 'models/transformer_pipeline.joblib'

# Configuración de la base de datos
DB_CONFIG = {
    'user': 'root',
    'password': 'password123',
    'host': 'localhost',
    'database': 'ml_db',
    'port': 3306
}
