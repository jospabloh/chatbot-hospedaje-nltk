"""Integration tests of the Flask API (test client, fake services)."""
import io
import json

import pytest

from casaguacate import config
from casaguacate.api import create_app
from tests.helpers import make_image


def post_agent(client, text="", photo=None, state=None, name="foto.png"):
    data = {"texto": text}
    if state is not None:
        data["estado"] = json.dumps(state)
    if photo is not None:
        data["file"] = (io.BytesIO(photo), name)
    return client.post("/agent", data=data, content_type="multipart/form-data")


def test_health(client):
    body = client.get("/health").get_json()
    assert body["estado"] == "ok" and body["modelos"]["intenciones"] and body["modelos"]["vision"] == "fake"
    assert "X-Request-ID" in client.get("/health").headers


def test_health_degradado_sin_vision(services):
    services.vision = None
    response = create_app(services, {"RATE_LIMIT_PER_MIN": 0}).test_client().get("/health")
    assert response.status_code == 200 and response.get_json()["estado"] == "degradado"


def test_chat_responde(client):
    response = client.post("/chat", json={"texto": "cuanto cuesta la noche"})
    body = response.get_json()
    assert response.status_code == 200 and body["intencion"]["etiqueta"] == "tarifas" and body["idioma"] == "es"
    assert body["resuelto_por_el_bot"] is True


def test_chat_acepta_idioma_explicito(client):
    assert client.post("/chat", json={"texto": "a que hora es el check in", "idioma": "en"}).get_json()["idioma"] == "en"


@pytest.mark.parametrize("payload,status,code", [
    ({}, 400, "texto_requerido"), ({"texto": ""}, 400, "texto_vacio"), ({"texto": 3}, 400, "texto_invalido"),
    ({"texto": "x" * 600}, 400, "texto_demasiado_largo"), ({"texto": "hola", "idioma": "fr"}, 400, "idioma_invalido")])
def test_chat_errores_de_validacion(client, payload, status, code):
    response = client.post("/chat", json=payload)
    body = response.get_json()
    assert response.status_code == status and body["error"]["codigo"] == code and body["request_id"]


def test_chat_json_invalido_y_tipo_no_soportado(client):
    assert client.post("/chat", data="no json", content_type="application/json").get_json()["error"]["codigo"] == "json_invalido"
    response = client.post("/chat", data="texto=hola", content_type="text/plain")
    assert response.status_code == 415 and response.get_json()["error"]["codigo"] == "tipo_no_soportado"


def test_image_clasifica(client, png_bytes):
    response = client.post("/image", data={"file": (io.BytesIO(png_bytes), "puerta.png")}, content_type="multipart/form-data")
    body = response.get_json()
    assert response.status_code == 200 and body["etiqueta"] == "cerradura" and body["modelo"] == "fake"
    assert body["supera_umbral"] is True and "latencia_ms" in body


@pytest.mark.parametrize("name,data,status,code", [
    ("foto.gif", make_image("PNG"), 415, "formato_no_permitido"),
    ("foto.jpg", b"basura", 415, "imagen_corrupta"),
    ("foto.png", make_image("PNG", size=(10, 10)), 422, "imagen_muy_pequena")])
def test_image_errores(client, name, data, status, code):
    response = client.post("/image", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")
    assert response.status_code == status and response.get_json()["error"]["codigo"] == code


def test_image_sin_archivo(client):
    response = client.post("/image", data={}, content_type="multipart/form-data")
    assert response.status_code == 400 and response.get_json()["error"]["codigo"] == "archivo_requerido"


def test_image_demasiado_grande_devuelve_413(client):
    big = b"0" * (config.MAX_IMAGE_BYTES + 400 * 1024)
    response = client.post("/image", data={"file": (io.BytesIO(big), "foto.jpg")}, content_type="multipart/form-data")
    assert response.status_code == 413 and response.get_json()["error"]["codigo"] == "archivo_demasiado_grande"


def test_image_sin_modelo_devuelve_503(services, png_bytes):
    services.vision = None
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 0}).test_client()
    response = client.post("/image", data={"file": (io.BytesIO(png_bytes), "a.png")}, content_type="multipart/form-data")
    assert response.status_code == 503 and response.get_json()["error"]["codigo"] == "modelo_no_disponible"


def test_agent_json_solo_texto(client):
    body = client.post("/agent", json={"texto": "a que hora es el check in"}).get_json()
    assert body["decision"]["accion"] == "RESPONDER" and body["decision"]["idioma"] == "es"
    assert [p["componente"] for p in body["traza"]] == ["Agente", "Capa reactiva", "Chatbot NLP", "Capa deliberativa"]
    assert body["estado"]["turnos"] == 1 and body["tiempos_ms"]["total"] >= 0


def test_agent_multipart_con_foto_y_estado(client, png_bytes):
    first = post_agent(client, "no me abre").get_json()
    assert first["decision"]["accion"] == "PEDIR_FOTO"
    second = post_agent(client, "", png_bytes, first["estado"]).get_json()
    assert second["decision"]["accion"] == "RESPONDER" and second["decision"]["intencion_final"] == "cerradura"
    assert "Clasificador visual" in [p["componente"] for p in second["traza"]] and second["estado"]["turnos"] == 2


def test_agent_urgencia(client):
    body = client.post("/agent", json={"texto": "hay un incendio"}).get_json()
    assert body["decision"]["accion"] == "ESCALAR_URGENTE" and body["decision"]["avisar_anfitriona"] is True


@pytest.mark.parametrize("payload,code", [({}, "entrada_vacia"), ({"texto": "   "}, "entrada_vacia"),
                                           ({"texto": "hola", "estado": {"idioma": "fr"}}, "estado_invalido"),
                                           ({"texto": "hola", "estado": {"turnos": -5}}, "estado_invalido"),
                                           ({"texto": 7}, "texto_invalido")])
def test_agent_errores(client, payload, code):
    response = client.post("/agent", json=payload)
    assert response.status_code == 400 and response.get_json()["error"]["codigo"] == code


def test_agent_foto_invalida_no_llega_al_modelo(client, vision):
    response = post_agent(client, "hola", b"no soy imagen", name="a.jpg")
    assert response.status_code == 415 and vision.calls == 0


def test_rutas_y_metodos_inexistentes(client):
    response = client.get("/nada")
    assert response.status_code == 404 and response.get_json()["error"]["codigo"] == "ruta_no_encontrada"
    response = client.get("/chat")
    assert response.status_code == 405 and response.get_json()["error"]["codigo"] == "metodo_no_permitido"


def test_error_interno_no_filtra_detalles(services):
    class Broken:
        def predict(self, text, lang):
            raise RuntimeError("secreto interno")
    services.intents = Broken()
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 0}).test_client()
    response = client.post("/agent", json={"texto": "a que hora es el check in"})
    body = response.get_json()
    assert response.status_code == 500 and body["error"]["codigo"] == "error_interno" and "secreto" not in json.dumps(body)


def test_metricas_cuentan_peticiones_errores_y_latencia(client):
    client.post("/chat", json={"texto": "cuanto cuesta la noche"})
    client.post("/chat", json={"texto": ""})
    client.post("/agent", json={"texto": "no me abre la puerta"})
    client.get("/nada")
    m = client.get("/metrics").get_json()
    assert m["por_ruta"]["/chat"]["peticiones"] == 2 and m["por_ruta"]["/chat"]["errores_cliente_4xx"] == 1
    assert m["por_ruta"]["/agent"]["latencia_ms"]["p95"] >= 0 and m["por_ruta"]["otras"]["por_codigo"]["404"] == 1
    assert m["eventos"]["accion:RESPONDER"] == 1 and m["errores_5xx_total"] == 0 and m["peticiones_total"] >= 4


def test_token_de_acceso(services):
    client = create_app(services, {"API_TOKEN": "secreto-de-prueba", "RATE_LIMIT_PER_MIN": 0}).test_client()
    assert client.get("/health").status_code == 200   # health stays public for the tunnel check
    denied = client.post("/chat", json={"texto": "hola"})
    assert denied.status_code == 401 and denied.get_json()["error"]["codigo"] == "token_invalido"
    assert client.post("/chat", json={"texto": "hola"}, headers={"X-API-Token": "otro"}).status_code == 401
    assert client.post("/chat", json={"texto": "hola"}, headers={"X-API-Token": "secreto-de-prueba"}).status_code == 200


def test_cada_huesped_tiene_su_propio_limite_aunque_compartan_ip(services):
    # behind Streamlit Cloud all guests share one IP: with a valid token each session id gets its own budget
    client = create_app(services, {"API_TOKEN": "t0ken-de-prueba", "RATE_LIMIT_PER_MIN": 3}).test_client()

    def chat(session_id):
        headers = {"X-API-Token": "t0ken-de-prueba", "CF-Connecting-IP": "9.9.9.9", "X-Session-Id": session_id}
        return client.post("/chat", json={"texto": "hola"}, headers=headers).status_code
    assert [chat("huesped-aaaa") for _ in range(4)] == [200, 200, 200, 429]
    assert [chat("huesped-bbbb") for _ in range(3)] == [200, 200, 200]


def test_sin_token_el_id_de_sesion_no_se_toma_en_cuenta(services):
    # an anonymous client could invent session ids to dodge the limit: without a token the limit is by IP
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 2}).test_client()
    codes = [client.post("/chat", json={"texto": "hola"}, headers={"CF-Connecting-IP": "1.1.1.1", "X-Session-Id": f"sesion-{i:04d}"}).status_code
             for i in range(3)]
    assert codes == [200, 200, 429]


def test_metrics_no_consume_el_limite_del_chat(services):
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 3}).test_client()
    headers = {"CF-Connecting-IP": "4.4.4.4"}
    assert [client.get("/metrics", headers=headers).status_code for _ in range(3)] == [200, 200, 200]
    assert [client.post("/chat", json={"texto": "hola"}, headers=headers).status_code for _ in range(3)] == [200, 200, 200]


def test_limite_de_peticiones_por_cliente(services):
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 3}).test_client()
    codes = [client.post("/chat", json={"texto": "hola"}, headers={"CF-Connecting-IP": "1.2.3.4"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    assert client.post("/chat", json={"texto": "hola"}, headers={"CF-Connecting-IP": "5.6.7.8"}).status_code == 200
    assert client.get("/health").status_code == 200


def test_el_log_no_contiene_el_texto_del_huesped(client):
    import logging

    class Collector(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    collector = Collector()
    logging.getLogger("casaguacate.api").addHandler(collector)
    try:
        client.post("/agent", json={"texto": "mi nombre es Fulano de Tal y no me abre la puerta"})
    finally:
        logging.getLogger("casaguacate.api").removeHandler(collector)
    logged = " ".join(collector.lines)
    assert "Fulano" not in logged and "no me abre" not in logged
    assert '"accion"' in logged and '"request_id"' in logged and '"chars"' in logged


def test_escalar_deja_registro_y_conteo_del_aviso_a_la_anfitriona(client):
    client.post("/agent", json={"texto": "hay un incendio"})
    client.post("/agent", json={"texto": "a que hora es el check in"})
    events = client.get("/metrics").get_json()["eventos"]
    assert events["avisar_anfitriona:si"] == 1 and events["accion:ESCALAR_URGENTE"] == 1
