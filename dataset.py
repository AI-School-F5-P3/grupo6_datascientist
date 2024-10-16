import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Función para generar fechas aleatorias
def random_date(start, end):
    return start + timedelta(
        seconds=np.random.randint(0, int((end - start).total_seconds()))
    )

# Configurar el generador de números aleatorios para reproducibilidad
np.random.seed(42)

# Número de registros
n = 1000

# Generar datos
data = {
    'id': range(1, n+1),
    'edad': np.random.randint(18, 80, n),
    'genero': np.random.choice(['M', 'F'], n),
    'ciudad': np.random.choice(['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao'], n),
    'fecha_registro': [random_date(datetime(2020, 1, 1), datetime(2023, 10, 9)) for _ in range(n)],
    'ultima_compra': [random_date(datetime(2023, 1, 1), datetime(2023, 10, 9)) for _ in range(n)],
    'total_compras': np.random.uniform(100, 5000, n).round(2),
    'categoria_favorita': np.random.choice(['Electrónica', 'Ropa', 'Hogar', 'Belleza', 'Deportes'], n),
    'dispositivo': np.random.choice(['móvil', 'desktop', 'tablet'], n),
    'tiempo_en_sitio': np.random.uniform(5, 120, n).round(1),
    'es_premium': np.random.choice([0, 1], n, p=[0.8, 0.2])
}

# Crear DataFrame
df = pd.DataFrame(data)

# Asegurar que la última compra no sea anterior a la fecha de registro
df['ultima_compra'] = df.apply(lambda row: max(row['ultima_compra'], row['fecha_registro']), axis=1)

# Guardar en CSV
df.to_csv('tienda_online_1000.csv', index=False)

print("Dataset creado y guardado como 'tienda_online_1000.csv'")