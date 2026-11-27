"""Regresion del 27-nov: la respuesta de horarios debe dar check-in (15:00) Y check-out (11:00) cuando se pregunta por ambos.

Causa raiz: en `conocimiento.reply_for` la intencion "horarios" elegia la respuesta de check-out en cuanto el texto
contenia cualquier palabra de salida, aunque tambien preguntara por la entrada ("entrada y salida", "check-in y check-out").
Tambien se prueba la redaccion de la regla de mascotas.
"""
import pytest

from casaguacate import conocimiento as kb
from casaguacate.agente import HybridAgent, Services
from tests.conftest import FakeIntents

# Preguntas que piden la entrada, o la entrada y la salida: la respuesta debe traer las dos horas.
PIDEN_ENTRADA = [
    ("¿A qué hora es el check-in?", "es"), ("¿A qué hora es la entrada y la salida?", "es"),
    ("¿A qué hora es el check-in y el check-out?", "es"), ("¿Cuál es el horario de check-in/check-out?", "es"),
    ("¿A qué hora es el check in y la salida?", "es"), ("¿A qué hora puedo llegar?", "es"),
    ("what time is check-in?", "en"), ("what time is check-in and check-out?", "en"),
    ("What are the check in and check out times?", "en"), ("when is arrival and departure?", "en"),
]
# Preguntas solo de salida: se contesta el check-out (con sus indicaciones).
SOLO_SALIDA = [("¿A qué hora es la salida?", "es"), ("hora del check out", "es"), ("What time is check-out?", "en"),
               ("when do we have to leave", "en")]


@pytest.mark.parametrize("text,lang", PIDEN_ENTRADA)
def test_horarios_con_entrada_trae_check_in_y_check_out(text, lang):
    reply = kb.reply_for("horarios", text, lang)
    assert "15:00" in reply and "11:00" in reply, reply


@pytest.mark.parametrize("text,lang", SOLO_SALIDA)
def test_horarios_solo_salida_trae_el_check_out(text, lang):
    assert "11:00" in kb.reply_for("horarios", text, lang)


@pytest.mark.parametrize("text,lang", PIDEN_ENTRADA)
def test_el_agente_responde_ambas_horas_en_horarios(text, lang):
    """De punta a punta con un clasificador simulado que acierta "horarios": la respuesta final lleva 15:00 y 11:00."""
    services = Services(intents=FakeIntents(default=("horarios", 0.9)))
    result = HybridAgent(services).run(text)
    assert result.final_intent == "horarios" and result.action == "RESPONDER"
    assert "15:00" in result.reply and "11:00" in result.reply, result.reply


def test_es_cuando_pregunta_por_ambos_no_se_queda_solo_con_check_out():
    """Si alguien cambia el orden de la logica y vuelve a responder solo salida, esto falla."""
    solo_salida = kb.reply_for("horarios", "¿A qué hora es la salida?", "es")
    ambas = kb.reply_for("horarios", "¿A qué hora es la entrada y la salida?", "es")
    assert ambas != solo_salida and "15:00" in ambas


def test_mascotas_se_lee_natural_en_espanol():
    reply = kb.reply_for("reglas", "¿Aceptan mascotas?", "es")
    assert "no se permiten mascotas" in reply
    assert "se aceptan" not in reply and "mascotas ni visitas" not in reply, reply


def test_mascotas_en_ingles_sigue_diciendo_que_no_se_permiten():
    assert "pets and visitors are not allowed" in kb.reply_for("reglas", "Can I bring my dog?", "en")
