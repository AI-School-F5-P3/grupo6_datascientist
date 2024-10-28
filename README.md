# PROYECTO DATA SCIENTIST: Detección de Riesgo de Ictus (Grupo 6)

## Descripción
Este proyecto tiene como objetivo desarrollar un prototipo de modelo de inteligencia artificial para predecir el riesgo de ictus en pacientes, basado en datos proporcionados por el hospital F5. La herramienta solicita información relevante sobre el paciente mediante una interfaz de línea de comandos y devuelve una predicción de riesgo de ictus como paso preliminar antes de la consulta médica. El modelo busca ofrecer una clasificación eficiente y fiable, con un nivel de sobreajuste (overfitting) controlado por debajo del 5%.

## Características principales
- **Predicción del riesgo de ictus** en función de indicadores de salud y estilo de vida.
- **Interfaz grafica Streamlit** para solicitar datos al usuario y mostrar el resultado.
- **Análisis de rendimiento del modelo** que incluye métricas clave de clasificación y evaluación de overfitting.
- **Estructura organizada en ramas** con commits claros y organizados para facilitar el seguimiento del desarrollo.

## Tecnologías utilizadas
- **Scikit-learn**: Para la implementación de algoritmos de aprendizaje supervisado y evaluación del modelo.
- **Pandas**: Para el manejo y procesamiento de datos.
- **Streamlit**: Interface grafica de usuario realizada con Streamlit
- **Git y GitHub**: Control de versiones y colaboración.
- **Trello** (u otras herramientas organizativas): Para la planificación y gestión del proyecto.

## Requisitos previos
Para ejecutar este proyecto necesitarás:
- Python 3.8 o superior
- Bibliotecas listadas en `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  
## Instalación y configuración

## Clona el repositorio:
git clone https://github.com/tu_usuario/proyecto_data_scientist_ictus.git

## Navega al directorio del proyecto:
cd proyecto_data_scientist_ictus

## Instala las dependencias necesarias:
pip install -r requirements.txt

## Uso

1. Ejecuta el script principal para iniciar el programa:
python app.py

2. Introduce los datos solicitados la interface grafica del modelo usado, como edad, sexo, historial médico, etc.
3. El sistema devolverá una predicción sobre el riesgo de ictus del paciente.

## Estructura del proyecto

**archivos raiz:** Script principal que ejecuta el programa.
**test/:** Contiene los test realizados sobre la app.
**eda_training/:** Almacena la configuraciones de cada modelo.
**Models:** Almacena los modelos entrenados.
