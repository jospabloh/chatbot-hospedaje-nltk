"""Unit tests of the reactive layer (safety rules)."""
import csv

import pytest

from casaguacate import config
from casaguacate.reglas import check_blocking, check_tone

URGENT = [
    "Hay un incendio en la cocina", "huele a gas en el pasillo", "hay humo en la habitación", "necesito una ambulancia",
    "mi esposo se desmayó", "se está inundando el baño", "Tengo una emergencia", "me quedé encerrada en el baño",
    "there is smoke in the room", "I smell gas", "there's a fire", "my husband fainted", "we have an emergency",
    "someone broke in", "I am locked in the bathroom",
]
NOT_URGENT = [
    "¿Se puede fumar?", "Can I smoke inside?", "¿Tienen asador?", "can I use the BBQ area", "¿Aceptan mascotas?",
    "¿A qué hora es el check-in?", "no me abre la puerta", "the door won't open", "¿Hay fuego artificial permitido?",
    "robot aspirador", "What time is check-out?",
]


@pytest.mark.parametrize("text", URGENT)
def test_urgencia_se_detecta_en_ambos_idiomas(text):
    hit = check_blocking(text)
    assert hit is not None and hit.rule == "urgencia"


@pytest.mark.parametrize("text", NOT_URGENT)
def test_frases_normales_no_disparan_urgencia(text):
    hit = check_blocking(text)
    assert hit is None or hit.rule != "urgencia"


def test_la_urgencia_tiene_prioridad_sobre_pedir_humano():
    assert check_blocking("Tengo una emergencia, quiero hablar con la anfitriona").rule == "urgencia"


@pytest.mark.parametrize("text", ["mi tarjeta es 4111 1111 1111 1111", "pago con 4111-1111-1111-1111 gracias",
                                  "the card 5500000000000004 expires 12/28", "mi cvv es 123"])
def test_datos_de_tarjeta_se_detectan(text):
    assert check_blocking(text).rule == "datos_sensibles"


@pytest.mark.parametrize("text", ["¿Está disponible el 14 de diciembre de 2026?", "mi teléfono es 8888 8888", "somos 4 personas"])
def test_fechas_y_telefonos_no_son_tarjetas(text):
    hit = check_blocking(text)
    assert hit is None or hit.rule != "datos_sensibles"


@pytest.mark.parametrize("text", ["Quiero hablar con Lore", "quiero hablar con la anfitriona", "necesito hablar con una persona",
                                  "can I talk to the host", "I want to speak with a human", "Perfect, I want to book, talk to Lore"])
def test_pedir_humano(text):
    assert check_blocking(text).rule == "pide_humano"


def test_las_frases_fuera_de_alcance_del_avance2_se_detectan(tmp_path):
    with open(config.DATA_DIR / "fuera_de_alcance.csv", encoding="utf-8") as f:
        phrases = [row["texto"] for row in csv.DictReader(f, delimiter="|")]
    assert len(phrases) == 8
    missed = [p for p in phrases if not (check_blocking(p) and check_blocking(p).rule == "fuera_de_alcance")]
    assert not missed, missed


@pytest.mark.parametrize("text", ["Do you sell concert tickets?", "any restaurant nearby?", "how is the weather there"])
def test_fuera_de_alcance_en_ingles(text):
    assert check_blocking(text).rule == "fuera_de_alcance"


@pytest.mark.parametrize("text", ["es el colmo, el portón no abre y nadie contesta", "llevo veinte minutos esperando",
                                  "esto es inaceptable", "we have been waiting outside for an hour and nobody answers",
                                  "this is unacceptable", "I want to file a complaint"])
def test_tono_molesto(text):
    assert check_tone(text).rule == "tono_molesto"


@pytest.mark.parametrize("text", ["muchas gracias por todo", "thank you, the place is lovely", "¿a qué hora es el check-in?",
                                  "estoy esperando la confirmación de mi reserva"])
def test_tono_neutral_o_positivo_no_se_marca(text):
    assert check_tone(text) is None


@pytest.mark.parametrize("text", ["can we check out later, our flight is at night", "late check-out please", "Is early check-in possible?",
                                  "¿podemos salir más tarde el domingo?", "queremos quedarnos más tarde", "¿se puede hacer late checkout?"])
def test_politica_de_la_anfitriona_check_out_y_check_in(text):
    assert check_blocking(text).rule == "politica_anfitriona"


@pytest.mark.parametrize("text", ["What time is check-out?", "¿A qué hora es la salida?", "we will arrive late tonight",
                                  "vamos a llegar tarde", "what time is check in", "¿a qué hora es el check in?"])
def test_horarios_normales_no_son_politica(text):
    hit = check_blocking(text)
    assert hit is None or hit.rule != "politica_anfitriona"
