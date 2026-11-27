"""Shared fixtures: fake services (fast, deterministic) and helpers to build images."""
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from casaguacate.agente import HybridAgent, Services  # noqa: E402
from casaguacate.texto import TextPrediction  # noqa: E402
from tests.helpers import make_image  # noqa: E402
from casaguacate.vision import ImagePrediction  # noqa: E402


class FakeIntents:
    """Returns scripted predictions by normalized text; records the calls."""

    def __init__(self, table=None, default=("saludo_cortesia", 0.1)):
        self.table, self.default, self.calls = table or {}, default, []

    def predict(self, text, lang):
        self.calls.append((text, lang))
        label, conf = self.table.get(text.lower().strip(" ?!¿."), self.default)
        return TextPrediction(label, conf, [(label, conf)])


class FakeVision:
    backbone = "fake"

    def __init__(self, label="cerradura", conf=0.95, threshold=0.60):
        self.label, self.conf, self.threshold, self.calls = label, conf, threshold, 0

    def predict(self, image):
        self.calls += 1
        shown = self.label if self.conf >= self.threshold else "otro"
        return ImagePrediction(shown, self.label, self.conf, [(self.label, self.conf)])


DEFAULT_TABLE = {
    "no me abre": ("cerradura", 0.55),
    "no me abre la puerta": ("cerradura", 0.80),
    "a que hora es el check in": ("horarios", 0.87),
    "what time is check in": ("horarios", 0.83),
    "cuanto cuesta la noche": ("tarifas", 0.80),
    "como puedo pagar": ("tarifas", 0.60),
    "mira esto": ("saludo_cortesia", 0.19),
    "asdfgh": ("saludo_cortesia", 0.10),
    "me puedo quedar mas tarde el domingo": ("early_late_checkout", 0.79),
    "hace mucho calor y no enfria": ("reportar_problema", 0.72),
    "el porton no abre y nadie contesta": ("porton_estacionamiento", 0.78),
    "is there a late check in": ("llegada_tarde", 0.82),
    "tenemos llegada tarde": ("llegada_tarde", 0.80),
    "hola": ("saludo_cortesia", 0.90),
    "gracias": ("saludo_cortesia", 0.85),
    "tienen alberca": ("servicios", 0.70),
    "como puedo cancelar mi reservacion": ("cancelacion", 0.85),
    "how can i cancel my booking": ("cancelacion", 0.78),
}


@pytest.fixture
def intents():
    return FakeIntents(dict(DEFAULT_TABLE))


@pytest.fixture
def vision():
    return FakeVision()


@pytest.fixture
def services(intents, vision):
    return Services(intents=intents, vision=vision)


@pytest.fixture
def agent(services):
    return HybridAgent(services)


@pytest.fixture
def app(services):
    from casaguacate.api import create_app
    return create_app(services, {"RATE_LIMIT_PER_MIN": 0})


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def png_bytes():
    return make_image("PNG")


@pytest.fixture
def jpg_bytes():
    return make_image("JPEG")
