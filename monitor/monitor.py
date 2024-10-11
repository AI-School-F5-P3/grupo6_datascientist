# monitor/monitor.py

import mysql.connector
from config import DB_CONFIG
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import datetime

class ModelMonitor:
    def __init__(self):
        # Conexión a la base de datos usando los detalles de configuración
        self.conn = mysql.connector.connect(
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            database=DB_CONFIG['database'],
            port=DB_CONFIG['port']
        )
        self.cursor = self.conn.cursor()

    def log_prediction(self, input_data, prediction, prediction_proba, actual=None):
        """
        Registra la predicción en la base de datos.
        """
        timestamp = datetime.datetime.now()
        query = """
            INSERT INTO predictions (timestamp, input_data, prediction, prediction_proba, actual)
            VALUES (%s, %s, %s, %s, %s)
        """
        self.cursor.execute(query, (timestamp, str(input_data), prediction, prediction_proba, actual))
        self.conn.commit()

    def evaluate_performance(self, y_true, y_pred):
        """
        Calcula métricas de rendimiento como ROC-AUC y precisión.
        """
        auc_score = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        return {"auc": auc_score, "accuracy": accuracy}

    def close_connection(self):
        """
        Cierra la conexión a la base de datos.
        """
        self.cursor.close()
        self.conn.close()
