"""Tests of the Streamlit client and app (friendly errors when the API fails, normal flow with a stubbed API)."""
import sys
from pathlib import Path

import pytest
import requests

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
sys.path.insert(0, str(UI_DIR))

import cliente_api  # noqa: E402
from cliente_api import ApiClient, ApiResult  # noqa: E402


class FakeResponse:
    def __init__(self, status, payload=None, raises_json=False):
        self.status_code, self._payload, self._raises = status, payload, raises_json

    def json(self):
        if self._raises:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


def client_with(response=None, error=None, token=""):
    session = FakeSession(response, error)
    return ApiClient("http://api.test/", token, session=session), session


def test_exito_devuelve_los_datos_y_manda_el_token():
    client, session = client_with(FakeResponse(200, {"estado": "ok"}), token="abc")
    result = client.health()
    assert result.ok and result.data["estado"] == "ok"
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("GET", "http://api.test/health") and kwargs["headers"] == {"X-API-Token": "abc"}


@pytest.mark.parametrize("error,fragment", [(requests.ConnectionError("x"), "No pude conectarme"), (requests.Timeout("x"), "tardó demasiado")])
def test_fallas_de_red_son_mensajes_amigables(error, fragment):
    result = client_with(error=error)[0].agent("hola")
    assert not result.ok and fragment in result.error and "Traceback" not in result.error


@pytest.mark.parametrize("status,fragment", [(401, "token"), (413, "demasiado grande"), (415, "JPG o PNG"), (422, "pequeña"),
                                              (429, "Espera un minuto"), (503, "no está disponible"), (500, "problema interno"),
                                              (502, "problema interno")])
def test_codigos_de_error_de_la_api(status, fragment):
    result = client_with(FakeResponse(status, {"error": {"codigo": "x", "mensaje": "detalle"}}))[0].agent("hola")
    assert not result.ok and result.status == status and fragment in result.error


def test_error_400_incluye_el_detalle_de_la_api():
    result = client_with(FakeResponse(400, {"error": {"codigo": "texto_vacio", "mensaje": "El campo 'texto' no puede estar vacio."}}))[0].agent("")
    assert "no puede estar vacio" in result.error


def test_respuesta_que_no_es_json_no_rompe():
    assert "no pude interpretar" in client_with(FakeResponse(200, raises_json=True))[0].agent("hola").error
    assert "internal problem" in client_with(FakeResponse(500, raises_json=True))[0].agent("hola", lang="en").error


def test_mensajes_en_ingles():
    assert "could not reach" in client_with(error=requests.ConnectionError())[0].health("en").error


def test_agent_envia_foto_y_estado_como_multipart():
    client, session = client_with(FakeResponse(200, {"decision": {}}))
    client.agent("hola", {"turnos": 1}, b"bytes", "a.png")
    _, url, kwargs = session.calls[0]
    assert url.endswith("/agent") and kwargs["files"]["file"][0] == "a.png" and '"turnos": 1' in kwargs["data"]["estado"]


# ---------------------------------------------------------------- Streamlit app
streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def run_app(monkeypatch, api_url="http://127.0.0.1:9"):
    monkeypatch.setenv("API_URL", api_url)
    return streamlit_testing.AppTest.from_file(str(UI_DIR / "streamlit_app.py"), default_timeout=30).run()


def test_la_app_abre_sin_api_y_avisa_sin_traceback(monkeypatch):
    app = run_app(monkeypatch)
    assert not app.exception
    assert any("Sin conexión" in e.value for e in app.sidebar.error)
    assert [t.label for t in app.tabs] == ["Asistente", "Cómo funciona", "Estado del servicio", "Privacidad y ética"]


def test_enviar_mensaje_con_la_api_caida_muestra_aviso_amigable(monkeypatch):
    app = run_app(monkeypatch)
    app.chat_input[0].set_value("¿A qué hora es el check-in?").run()
    assert not app.exception
    assert any("No pude conectarme" in w.value for w in app.warning)


def test_flujo_normal_con_api_simulada(monkeypatch):
    decision = {"accion": "RESPONDER", "respuesta": "Check-in desde las 15:00", "avisar_anfitriona": False, "motivo": "prueba",
                "intencion_final": "horarios", "decidido_por": "texto", "idioma": "es"}
    trace = [{"paso": 1, "componente": "Agente", "evento": "percepcion", "datos": {"idioma": "es"}, "ms": 0.1}]
    monkeypatch.setattr(cliente_api.ApiClient, "health", lambda self, lang="es": ApiResult(True, {"estado": "ok"}))
    monkeypatch.setattr(cliente_api.ApiClient, "agent", lambda self, text, state=None, photo=None, filename="f", lang="es":
                        ApiResult(True, {"decision": decision, "traza": trace, "estado": {"turnos": 1}}))
    app = run_app(monkeypatch)
    assert any("Conectado" in s.value for s in app.sidebar.success)
    app.chat_input[0].set_value("¿A qué hora es el check-in?").run()
    assert not app.exception
    assert any("15:00" in m.value for m in app.markdown)
    assert any("Respondió" in c.value for c in app.caption)


def test_cambio_de_idioma_de_la_interfaz(monkeypatch):
    app = run_app(monkeypatch)
    app.sidebar.selectbox[0].select("en").run()
    assert not app.exception and "Cas'aguacate guest assistant" in app.title[0].value
