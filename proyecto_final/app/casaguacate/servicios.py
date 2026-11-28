"""Loads the exported artifacts into the Services container used by the agent and the API."""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from . import config
from .agente import Services, VisionUnavailable
from .texto import IntentClassifier, ToneClassifier

log = logging.getLogger("casaguacate.api")


class LazyVision:
    """Vision classifier that imports TensorFlow and builds the backbone only on the first photo.

    It exposes the same interface the API and the agent use (`backbone`, `predict`). The head and the metadata are
    checked at creation time (cheap, no TensorFlow), so /health still reports the vision model as available.
    Used by `load_services(lazy_vision=True)`, which the Streamlit integrated mode calls to keep memory low
    until a guest actually sends a photo.
    """

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or config.ARTIFACTS_DIR)
        meta = json.loads((self.directory / "vision_meta.json").read_text(encoding="utf-8"))
        self.backbone = config.VISION_BACKBONE or meta["despliegue"]
        if not (self.directory / f"vision_head_{self.backbone}.joblib").exists():
            raise FileNotFoundError(f"missing vision head for {self.backbone}")
        self._classifier = None
        self._error: str | None = None   # type of the exception if the first load failed (then it is not retried)
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """True once TensorFlow and the backbone are in memory (used by tests and diagnostics)."""
        return self._classifier is not None

    @property
    def available(self) -> bool:
        """False after a failed load; agente.vision_available reads it so /health reports "degradado" and
        /image and /agent answer 503, as when the eager load fails at start-up."""
        return self._error is None

    def predict(self, image):
        """Classifies a PIL image, loading the real classifier the first time (thread-safe, loads once).

        If that load fails the model is marked unavailable, the failure is logged once and VisionUnavailable is
        raised (the API answers 503); later photos get VisionUnavailable without retrying the load.
        """
        if self._classifier is None:
            with self._lock:
                if self._error is not None:
                    raise VisionUnavailable(self._error)
                if self._classifier is None:
                    try:
                        from .vision import ImageClassifier
                        self._classifier = ImageClassifier.load(self.directory)
                    except Exception as exc:  # noqa: BLE001 - same degradation as the eager load in load_services
                        self._error = type(exc).__name__
                        log.warning('{"evento": "modelo_no_cargado", "modelo": "vision", "tipo": "%s"}', self._error)
                        raise VisionUnavailable(self._error) from exc
        return self._classifier.predict(image)


def load_services(lazy_vision: bool = False) -> Services:
    """Intents are required (an error here stops the start-up). Tone and vision degrade gracefully.

    With lazy_vision=True TensorFlow is not imported until the first photo (see LazyVision); the default keeps the
    eager load that gunicorn / servidor.py use.
    """
    intents = IntentClassifier.load()
    tone = vision = None
    if (config.ARTIFACTS_DIR / "tono.joblib").exists():  # optional: by default the tone comes from the reactive rules
        tone = ToneClassifier.load()
    try:
        if lazy_vision:
            vision = LazyVision()
        else:
            from .vision import ImageClassifier
            vision = ImageClassifier.load()
    except Exception as exc:  # noqa: BLE001 - /health reports "degradado" and /image answers 503
        log.warning('{"evento": "modelo_no_cargado", "modelo": "vision", "tipo": "%s"}', type(exc).__name__)
    return Services(intents=intents, vision=vision, tone=tone)
