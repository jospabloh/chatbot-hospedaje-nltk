"""Standalone Gradio chat UI for the Casa Almendro rule-based chatbot (uses chatbot_hospedaje.py unchanged)."""
import sys
sys.path.insert(0, ".")  # chatbot_hospedaje.py in the same folder
import gradio as gr
from chatbot_hospedaje import RentalChatbot

bot = RentalChatbot()

def respond(message, history):
    intent, answer = bot.process(message)
    return answer

demo = gr.ChatInterface(
    fn=respond,
    title="Casa Almendro · Asistente de huéspedes",
    description="Chatbot basado en reglas (Python + NLTK). Datos de ejemplo. Responde en español o inglés.",
    examples=["¿Cuánto cuesta la noche?", "¿Está disponible el 14 de diciembre?", "No puedo abrir la puerta",
              "What time is check-in?", "Is November 20 available?", "Do you have wifi?"],
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
