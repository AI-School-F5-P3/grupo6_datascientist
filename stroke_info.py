import streamlit as st
from typing import List, Dict

class StrokeInformation:
    """
    A class to manage and display stroke-related information using Streamlit.
    Provides structured information about stroke prevention, symptoms, and emergency response.
    """
    
    def __init__(self):
        """Initialize the styling configurations for the component."""
        self.styles = {
            'container': """
                <style>
                    .main-container {
                        padding: 2rem;
                        background-color: rgba(0, 0, 0, 0.1);
                        border-radius: 10px;
                        margin: 1rem 0;
                    }
                    .title {
                        color: #ffffff;
                        font-size: 2.5rem;
                        text-align: center;
                        margin-bottom: 2rem;
                    }
                    .subtitle {
                        color: #ffffff;
                        font-size: 1.8rem;
                        margin: 1.5rem 0;
                    }
                    .description {
                        color: #ffffff;
                        font-size: 1.1rem;
                        line-height: 1.6;
                        margin: 1rem 0;
                    }
                    ul {
                        margin-left: 2rem;
                        margin-bottom: 1.5rem;
                    }
                    li {
                        margin: 0.5rem 0;
                        line-height: 1.4;
                    }
                </style>
            """
        }
        
    def _create_section(self, title: str, content: str, is_list: bool = False, items: List[str] = None) -> None:
        """
        Create a section with a title and content.
        
        Args:
            title (str): The section title
            content (str): The main content or description
            is_list (bool): Whether the content should be displayed as a list
            items (List[str]): List items if is_list is True
        """
        st.markdown(f"<h2 class='subtitle'>{title}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p class='description'>{content}</p>", unsafe_allow_html=True)
        
        if is_list and items:
            items_html = "\n".join([f"<li>{item}</li>" for item in items])
            st.markdown(f"<ul style='color: white;'>{items_html}</ul>", unsafe_allow_html=True)

    def _create_fast_section(self) -> None:
        """Create the FAST protocol section with formatted content."""
        fast_items = [
            "<strong style='color: white;'>F</strong>ace (Cara): Pide a la persona que sonría. ¿Un lado de la cara cae?",
            "<strong style='color: white;'>A</strong>rms (Brazos): Pide que levante ambos brazos. ¿Uno de los brazos cae?",
            "<strong style='color: white;'>S</strong>peech (Habla): Pide que repita una frase simple. ¿El habla es confusa o extraña?",
            "<strong style='color: white;'>T</strong>ime (Tiempo): Si observas cualquiera de estos signos, llama inmediatamente a emergencias."
        ]
        
        st.markdown("<h2 class='subtitle'>Actuación en caso de Ictus</h2>", unsafe_allow_html=True)
        st.markdown("<p class='description'>Si sospechas que alguien está sufriendo un ictus, recuerda el acrónimo FAST:</p>", unsafe_allow_html=True)
        st.markdown(f"<ul style='color: white;'>{''.join([f'<li>{item}</li>' for item in fast_items])}</ul>", unsafe_allow_html=True)

    def display_information(self) -> None:
        """Display all stroke-related information in a structured format."""
        st.markdown(self.styles['container'], unsafe_allow_html=True)
        st.markdown("<div class='main-container'>", unsafe_allow_html=True)
        
        # Título principal
        st.markdown("<h1 class='title'>Información y Prevención de Ictus</h1>", unsafe_allow_html=True)
        
        # Definición
        self._create_section(
            "¿Qué es un Ictus?",
            "Un ictus, también conocido como accidente cerebrovascular (ACV), ocurre cuando el suministro de sangre a una parte del cerebro se interrumpe o se reduce, privando al tejido cerebral de oxígeno y nutrientes."
        )
        
        # Factores de riesgo
        risk_factors = [
            "Hipertensión arterial", "Diabetes", "Colesterol alto", "Tabaquismo",
            "Obesidad", "Sedentarismo", "Edad avanzada", "Antecedentes familiares"
        ]
        self._create_section(
            "Factores de Riesgo",
            "Algunos factores de riesgo incluyen:",
            True,
            risk_factors
        )
        
        # Síntomas
        symptoms = [
            "Debilidad o entumecimiento repentino en la cara, brazo o pierna, especialmente en un lado del cuerpo",
            "Confusión súbita o dificultad para hablar o entender",
            "Problemas repentinos de visión en uno o ambos ojos",
            "Dificultad repentina para caminar, mareo, pérdida de equilibrio o coordinación",
            "Dolor de cabeza severo sin causa conocida"
        ]
        self._create_section(
            "Síntomas de Alerta",
            "Es crucial reconocer los síntomas de un ictus para actuar rápidamente:",
            True,
            symptoms
        )
        
        # Prevención
        prevention_measures = [
            "Controlar la presión arterial",
            "Mantener una dieta saludable y equilibrada",
            "Realizar actividad física regularmente",
            "Dejar de fumar",
            "Limitar el consumo de alcohol",
            "Controlar el estrés",
            "Realizar chequeos médicos regulares"
        ]
        self._create_section(
            "Prevención",
            "Algunas medidas para prevenir un ictus incluyen:",
            True,
            prevention_measures
        )
        
        # Sección FAST
        self._create_fast_section()
        
        # Mensaje final
        st.markdown(
            "<p class='description'>Recuerda, en caso de ictus, cada minuto cuenta. "
            "La atención médica inmediata puede marcar la diferencia entre la recuperación "
            "y la discapacidad permanente.</p>",
            unsafe_allow_html=True
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

def mostrar_informacion_prevencion():
    """
    Función principal para mostrar la información sobre prevención de ictus.
    Esta función inicializa y muestra toda la información utilizando la clase StrokeInformation.
    """
    stroke_info = StrokeInformation()
    stroke_info.display_information()

if __name__ == "__main__":
    mostrar_informacion_prevencion()