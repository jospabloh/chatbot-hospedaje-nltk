"""Loads the exported artifacts into the Services container used by the agent and the API."""
from __future__ import annotations

import logging

from . import config
from .agente import Services
from .texto import IntentClassifier, ToneClassifier

log = logging.getLogger("casaguacate.api")


def load_services() -> Services:
    """Intents are required (an error here stops the start-up). Tone and vision degrade gracefully."""
    intents = IntentClassifier.load()
    tone = vision = None
    if (config.ARTIFACTS_DIR / "tono.joblib").exists():  # optional: by default the tone comes from the reactive rules
        tone = ToneClassifier.load()
    try:
        from .vision import ImageClassifier
        vision = ImageClassifier.load()
    except Exception as exc:  # noqa: BLE001 - /health reports "degradado" and /image answers 503
        log.warning('{"evento": "modelo_no_cargado", "modelo": "vision", "tipo": "%s"}', type(exc).__name__)
    return Services(intents=intents, vision=vision, tone=tone)
