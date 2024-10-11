import mysql.connector

class Database:
    def __init__(self, db_config):
        self.connection = mysql.connector.connect(**db_config)
    
    def insert_data(self, table, data):
        cursor = self.connection.cursor()
        placeholders = ', '.join(['%s'] * len(data))
        query = f"INSERT INTO {table} VALUES ({placeholders})"
        cursor.execute(query, tuple(data.values()))
        self.connection.commit()
        cursor.close()

    def fetch_data(self, query):
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def close(self):
        self.connection.close()

# Uso:
# db = Database(db_config={'user': 'root', 'password': '', 'host': 'localhost', 'database': 'ml_db'})
# db.insert_data('predictions', {'id': 1, 'prediction': 0.9})