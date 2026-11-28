"""Tests of the integrated mode: without API_URL the Streamlit app hosts the real Flask API in a background thread
and talks to it over real HTTP on loopback (fake services keep the tests fast; the HTTP layer is real)."""
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from casaguacate.agente import Services
from tests.conftest import DEFAULT_TABLE, FakeIntents, FakeVision

UI_DIR = Path(__file__).resolve().parents[1] / "ui"
APP_DIR = UI_DIR.parent
sys.path.insert(0, str(UI_DIR))

import api_integrada  # noqa: E402


def fake_services() -> Services:
    return Services(intents=FakeIntents(dict(DEFAULT_TABLE)), vision=FakeVision())


@pytest.fixture
def embedded():
    api = api_integrada.start_embedded_api(fake_services(), rate_limit_per_min=0)
    yield api
    api.stop()


def test_sin_api_url_la_api_embebida_responde_health_por_http_real(embedded):
    assert embedded.url.startswith("http://127.0.0.1:") and embedded.alive()
    response = requests.get(embedded.url + "/health", timeout=10)   # /health is public, as in the external API
    assert response.status_code == 200 and response.json()["estado"] == "ok"


def test_chat_por_http_exige_token_y_responde_con_el_token(embedded):
    body = {"texto": "cuanto cuesta la noche"}
    assert requests.post(embedded.url + "/chat", json=body, timeout=10).status_code == 401
    assert requests.post(embedded.url + "/chat", json=body, headers={"X-API-Token": "otro"}, timeout=10).status_code == 401
    ok = requests.post(embedded.url + "/chat", json=body, headers={"X-API-Token": embedded.token}, timeout=10)
    assert ok.status_code == 200 and ok.json()["intencion"]["etiqueta"] == "tarifas"


def test_cada_arranque_usa_token_y_puerto_propios():
    first = api_integrada.start_embedded_api(fake_services())
    second = api_integrada.start_embedded_api(fake_services())
    try:
        assert first.token != second.token and len(first.token) >= 32 and first.url != second.url
    finally:
        first.stop(), second.stop()


def test_la_api_embebida_aplica_limite_de_peticiones():
    api = api_integrada.start_embedded_api(fake_services(), rate_limit_per_min=2)
    try:
        codes = [requests.post(api.url + "/chat", json={"texto": "hola"}, headers={"X-API-Token": api.token}, timeout=10).status_code
                 for _ in range(3)]
        assert codes == [200, 200, 429]
    finally:
        api.stop()


# ---------------------------------------------------------------- Streamlit app
streamlit_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.fixture
def started(monkeypatch):
    """Replaces the embedded-API factory with one that uses fake services and records every start (and stops them)."""
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("API_TOKEN", raising=False)
    real_start, apis = api_integrada.start_embedded_api, []

    def recording_start(services=None, rate_limit_per_min=None):
        api = real_start(fake_services(), rate_limit_per_min=0)
        apis.append(api)
        return api

    monkeypatch.setattr(api_integrada, "start_embedded_api", recording_start)
    yield apis
    for api in apis:
        api.stop()
    st.cache_resource.clear()


def run_ui():
    return streamlit_testing.AppTest.from_file(str(UI_DIR / "streamlit_app.py"), default_timeout=30).run()


def test_la_app_sin_api_url_levanta_la_api_una_sola_vez_y_el_chat_funciona(started):
    app = run_ui()
    assert not app.exception and len(started) == 1
    assert any("Conectado" in s.value for s in app.sidebar.success)
    app.chat_input[0].set_value("a que hora es el check in").run()
    assert not app.exception and len(started) == 1       # the second run reuses the same embedded API
    assert any("Respondió" in c.value for c in app.caption)
    assert any("integrada" in c.value for c in app.sidebar.caption)


def test_el_modo_integrado_se_indica_en_ambos_idiomas(started):
    app = run_ui()
    assert any("API REST integrada en la app" in c.value for c in app.sidebar.caption)
    app.sidebar.selectbox[0].select("en").run()
    assert any("REST API integrated in the app" in c.value for c in app.sidebar.caption)


def test_con_api_url_configurada_no_se_levanta_nada(started, monkeypatch):
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    app = run_ui()
    assert not app.exception and started == []
    assert any("API REST externa" in c.value for c in app.sidebar.caption)
    assert any("Sin conexión" in e.value for e in app.sidebar.error)


def test_en_modo_integrado_no_hay_campo_para_cambiar_la_url(started, monkeypatch):
    monkeypatch.setenv("ALLOW_API_URL_OVERRIDE", "1")
    app = run_ui()
    assert not app.exception and not [w for w in app.sidebar.text_input if "API" in w.label]


# ---------------------------------------------------------------- lazy TensorFlow
def test_cargar_servicios_perezosos_no_importa_tensorflow():
    code = ("import sys; sys.path.insert(0, %r); from casaguacate.servicios import load_services; "
            "s = load_services(lazy_vision=True); "
            "assert s.vision.backbone and not s.vision.loaded; "
            "assert 'tensorflow' not in sys.modules, 'TensorFlow se importo al arrancar'; print('ok')" % str(APP_DIR))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr[-800:]


def test_vision_perezosa_carga_el_modelo_una_sola_vez_en_la_primera_foto(monkeypatch):
    from casaguacate import vision
    from casaguacate.servicios import LazyVision
    loads = []

    class Loaded:
        def predict(self, image):
            return "prediccion"

    monkeypatch.setattr(vision.ImageClassifier, "load", classmethod(lambda cls, directory=None: loads.append(directory) or Loaded()))
    lazy = LazyVision()
    assert loads == [] and not lazy.loaded
    assert lazy.predict(object()) == "prediccion" and lazy.predict(object()) == "prediccion"
    assert len(loads) == 1 and lazy.loaded


def test_vision_perezosa_que_falla_degrada_igual_que_sin_modelo(monkeypatch, png_bytes):
    """If the first load fails (e.g. the ImageNet weights cannot be downloaded) the API answers 503, /health says
    "degradado" and the load is not retried on every photo, the same as when the eager load fails at start-up."""
    import io

    from casaguacate import vision
    from casaguacate.api import create_app
    from casaguacate.servicios import LazyVision
    attempts = []

    def failing_load(cls, directory=None):
        attempts.append(directory)
        raise OSError("sin red para descargar los pesos")

    monkeypatch.setattr(vision.ImageClassifier, "load", classmethod(failing_load))
    services = Services(intents=FakeIntents(dict(DEFAULT_TABLE)), vision=LazyVision())
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 0}).test_client()
    assert client.get("/health").get_json()["estado"] == "ok"            # nothing loaded yet: the head is there

    first = client.post("/image", data={"file": (io.BytesIO(png_bytes), "a.png")}, content_type="multipart/form-data")
    assert first.status_code == 503 and first.get_json()["error"]["codigo"] == "modelo_no_disponible"
    agent = client.post("/agent", data={"texto": "no abre la puerta", "file": (io.BytesIO(png_bytes), "a.png")},
                        content_type="multipart/form-data")
    assert agent.status_code == 503 and agent.get_json()["error"]["codigo"] == "modelo_no_disponible"
    second = client.post("/image", data={"file": (io.BytesIO(png_bytes), "a.png")}, content_type="multipart/form-data")
    assert second.status_code == 503 and len(attempts) == 1             # not retried on every photo
    health = client.get("/health").get_json()
    assert health["estado"] == "degradado" and health["modelos"]["vision"] is False


def test_vision_perezosa_que_falla_dentro_del_agente_devuelve_503(monkeypatch, png_bytes):
    """First photo arrives through /agent (not /image): the failure inside agent.run also becomes a 503."""
    import io

    from casaguacate import vision
    from casaguacate.api import create_app
    from casaguacate.servicios import LazyVision

    def failing_load(cls, directory=None):
        raise OSError("sin red")

    monkeypatch.setattr(vision.ImageClassifier, "load", classmethod(failing_load))
    services = Services(intents=FakeIntents(dict(DEFAULT_TABLE)), vision=LazyVision())
    client = create_app(services, {"RATE_LIMIT_PER_MIN": 0}).test_client()
    response = client.post("/agent", data={"texto": "no abre la puerta", "file": (io.BytesIO(png_bytes), "a.png")},
                           content_type="multipart/form-data")
    assert response.status_code == 503 and response.get_json()["error"]["codigo"] == "modelo_no_disponible"
