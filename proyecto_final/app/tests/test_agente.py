"""Unit tests of the hybrid agent with fake services (deterministic)."""
import re

import pytest
from PIL import Image

from casaguacate import conocimiento as kb
from casaguacate.agente import ConversationState, answer_text

PHOTO = Image.new("RGB", (64, 64), (10, 20, 30))
FORBIDDEN = ["MXN", "Las Palmas", "Almendro", "Marina", "+52"]


def components(result):
    return [step["componente"] for step in result.trace]


def test_urgencia_no_llama_a_ningun_modelo(agent, intents, vision):
    result = agent.run("Hay un incendio en la cocina", PHOTO)
    assert result.action == "ESCALAR_URGENTE" and result.notify_host
    assert "911" in result.reply
    assert intents.calls == [] and vision.calls == 0
    assert components(result) == ["Agente", "Capa reactiva"]


def test_urgencia_en_ingles_responde_en_ingles(agent):
    result = agent.run("there is smoke in the room")
    assert result.action == "ESCALAR_URGENTE" and result.lang == "en" and "call 911" in result.reply


def test_datos_de_tarjeta_se_protegen_sin_clasificar(agent, intents):
    result = agent.run("mi tarjeta es 4111 1111 1111 1111")
    assert result.action == "PROTEGER_DATOS" and intents.calls == []
    assert "4111" not in result.reply


def test_pedir_humano_escala_a_la_anfitriona(agent):
    result = agent.run("quiero hablar con la anfitriona")
    assert result.action == "ESCALAR" and result.final_intent == "contactar_anfitrion" and "Lore" in result.reply


def test_fuera_de_alcance_no_inventa_y_escala(agent, intents):
    result = agent.run("recomiendan algun restaurante por aca")
    assert result.action == "ESCALAR" and result.final_intent == "fuera_de_alcance" and intents.calls == []


def test_pregunta_clara_se_responde_con_datos_reales(agent):
    result = agent.run("a que hora es el check in")
    assert result.action == "RESPONDER" and "15:00" in result.reply
    assert components(result) == ["Agente", "Capa reactiva", "Chatbot NLP", "Capa deliberativa"]


def test_pregunta_en_ingles_responde_en_ingles(agent):
    result = agent.run("what time is check in")
    assert result.lang == "en" and result.action == "RESPONDER" and "Check-in from" in result.reply


def test_texto_ambiguo_de_acceso_pide_foto_y_luego_la_usa(agent):
    first = agent.run("no me abre")
    assert first.action == "PEDIR_FOTO" and first.state.pidio_foto and first.state.intencion_pendiente == "cerradura"
    agent.services.vision.label = "porton"
    second = agent.run("", PHOTO, first.state)
    assert second.action == "RESPONDER" and second.final_intent == "porton_estacionamiento"
    assert "portón" in second.reply and second.state.pidio_foto is False
    assert components(second)[-2:] == ["Clasificador visual", "Capa deliberativa"]


def test_no_pide_la_foto_dos_veces(agent):
    first = agent.run("no me abre")
    second = agent.run("no me abre", None, first.state)
    assert second.action == "RESPONDER" and second.final_intent == "cerradura"


def test_la_foto_corrige_al_texto_cuando_son_distintos(agent):
    agent.services.vision.label = "porton"
    result = agent.run("no me abre la puerta", PHOTO)
    assert result.final_intent == "porton_estacionamiento" and result.decided_by == "imagen"


def test_texto_y_foto_coinciden(agent):
    result = agent.run("no me abre la puerta", PHOTO)
    assert result.final_intent == "cerradura" and result.decided_by == "texto e imagen coinciden"


def test_texto_ambiguo_con_foto_confiable_decide_la_foto(agent):
    agent.services.vision.label = "caseta"
    result = agent.run("mira esto", PHOTO)
    assert result.final_intent == "acceso_fraccionamiento" and result.action == "RESPONDER"


def test_foto_interior_con_texto_vago_va_a_la_anfitriona(agent):
    agent.services.vision.label = "interior"
    result = agent.run("mira esto", PHOTO)
    assert result.action == "ESCALAR" and "foto" in result.reply.lower()


def test_foto_no_confiable_se_trata_como_otro_y_escala(agent):
    agent.services.vision.conf = 0.40
    result = agent.run("mira esto", PHOTO)
    assert result.action == "ESCALAR" and "No logro identificar" in result.reply


def test_foto_fuera_de_dominio_no_estorba_una_pregunta_clara(agent):
    agent.services.vision.conf = 0.30
    result = agent.run("tienen alberca", PHOTO)
    assert result.action == "RESPONDER" and "No tenemos piscina" in result.reply


def test_dos_mensajes_incomprensibles_escalan(agent):
    first = agent.run("asdfgh")
    assert first.action == "ACLARAR" and first.state.fallos == 1
    second = agent.run("asdfgh", None, first.state)
    assert second.action == "ESCALAR" and second.state.fallos == 0 and second.state.escalada


def test_un_mensaje_entendido_reinicia_los_fallos(agent):
    first = agent.run("asdfgh")
    second = agent.run("hola", None, first.state)
    assert second.action == "RESPONDER" and second.state.fallos == 0


@pytest.mark.parametrize("text", ["is there a late check in", "tenemos llegada tarde"])
def test_llegada_tarde_no_es_politica_de_la_anfitriona(agent, text):
    # a late arrival is a normal question (arrival instructions); only early check-in and early/late check-out need the host
    result = agent.run(text)
    assert result.action == "RESPONDER" and result.final_intent == "llegada_tarde" and not result.notify_host


def test_pedir_foto_reinicia_los_fallos_consecutivos(agent):
    # unrecognized -> understood (asks for a photo) -> unrecognized are NOT two consecutive failures
    first = agent.run("asdfgh")
    second = agent.run("hace mucho calor y no enfria", None, first.state)
    assert second.action == "PEDIR_FOTO" and second.state.fallos == 0
    third = agent.run("asdfgh", None, second.state)
    assert third.action == "ACLARAR" and third.state.fallos == 1 and not third.notify_host


def test_tono_molesto_responde_y_avisa(agent):
    result = agent.run("el porton no abre y nadie contesta")
    assert result.action == "RESPONDER_Y_AVISAR" and result.notify_host
    assert "portón" in result.reply and "Lore" in result.reply


def test_tono_molesto_sin_resolver_escala(agent):
    result = agent.run("es el colmo asdfgh")
    assert result.action == "ESCALAR"


def test_late_checkout_lo_decide_la_anfitriona(agent):
    result = agent.run("me puedo quedar mas tarde el domingo")
    assert result.action == "ESCALAR" and result.final_intent == "early_late_checkout"


def test_problema_pide_foto_primero_y_luego_escala(agent):
    first = agent.run("hace mucho calor y no enfria")
    assert first.action == "PEDIR_FOTO"
    agent.services.vision.label = "controles"
    second = agent.run("aqui esta", PHOTO, first.state)
    assert second.action == "ESCALAR" and second.final_intent == "reportar_problema"


def test_mensaje_vacio_pide_aclaracion(agent):
    assert agent.run("   ").action == "ACLARAR"


def test_el_idioma_se_conserva_en_el_estado_y_cambia_con_el_huesped(agent):
    first = agent.run("what time is check in")
    assert first.state.idioma == "en"
    second = agent.run("cuanto cuesta la noche", None, first.state)
    assert second.lang == "es" and second.state.idioma == "es"


def test_la_traza_expone_el_intercambio_entre_componentes(agent):
    result = agent.run("no me abre la puerta", PHOTO)
    assert components(result) == ["Agente", "Capa reactiva", "Chatbot NLP", "Clasificador visual", "Capa deliberativa"]
    assert result.trace[2]["datos"]["etiqueta"] == "cerradura" and result.trace[3]["datos"]["etiqueta"] == "cerradura"
    assert [s["paso"] for s in result.trace] == [1, 2, 3, 4, 5]


def test_el_agente_no_modifica_el_estado_recibido(agent):
    state = ConversationState()
    agent.run("no me abre", None, state)
    assert state.turnos == 0 and state.pidio_foto is False


@pytest.mark.parametrize("data", [{"idioma": "fr"}, {"turnos": -1}, {"turnos": True}, {"fallos": "2"}, {"pidio_foto": 1},
                                  {"extra": 1}, {"intencion_pendiente": 5}, {"intencion_pendiente": "x" * 50}, "texto"])
def test_estado_invalido_se_rechaza(data):
    with pytest.raises(ValueError):
        ConversationState.from_dict(data)


def test_las_respuestas_no_contienen_datos_prohibidos_ni_codigos():
    forbidden_codes = re.compile(r"(?<![/\d])\d{4,}(?![/\d])")
    for lang in ("es", "en"):
        replies = [kb.reply_for(i, "hola", lang) for i in list(kb.INTENT_TO_REPLY) + ["horarios", "saludo_cortesia"]]
        replies += [kb.message(k, lang) for k in kb.MESSAGES[lang]]
        for reply in filter(None, replies):
            assert not any(word in reply for word in FORBIDDEN), reply
            assert not forbidden_codes.search(reply), reply
            assert "[HANDOFF" not in reply and "[TRANSFERENCIA" not in reply


def test_answer_text_componente_chatbot_solo(services):
    out = answer_text(services, "cuanto cuesta la noche")
    assert out["resuelto_por_el_bot"] and out["intencion"]["etiqueta"] == "tarifas" and "reservations.casaguacate" in out["respuesta"]
    low = answer_text(services, "asdfgh")
    assert not low["resuelto_por_el_bot"]


def test_pagos_se_responden_aunque_el_corpus_no_tenga_intencion_de_pagos(agent):
    result = agent.run("como puedo pagar")
    assert result.action == "RESPONDER" and "SINPE" in result.reply
