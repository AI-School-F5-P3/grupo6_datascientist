import joblib
import pandas as pd
from config import MODEL_PATH, TRANSFORMER_PATH

class ModelPipeline:
    def __init__(self):
        # Cargar el modelo y el pipeline de transformación
        self.model = joblib.load(MODEL_PATH)
        self.transformer = joblib.load(TRANSFORMER_PATH)

    def preprocess(self, input_data):
        """
        Transforma los datos de entrada usando el transformer.
        """
        transformed_data = self.transformer.transform(input_data)
        return transformed_data

    def predict(self, input_data):
        """
        Hace predicciones en los datos procesados.
        """
        transformed_data = self.preprocess(input_data)
        prediction = self.model.predict(transformed_data)
        return prediction

    def predict_proba(self, input_data):
        """
        Obtiene la probabilidad de las predicciones.
        """
        transformed_data = self.preprocess(input_data)
        prediction_proba = self.model.predict_proba(transformed_data)
        return prediction_proba