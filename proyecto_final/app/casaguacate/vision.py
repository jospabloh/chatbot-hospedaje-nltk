"""Computer vision component: frozen ImageNet backbone + a small 6-class head ("otro" included).

The backbone is loaded with keras.applications (ImageNet weights are downloaded once to ~/.keras); only the
head (a few KB) is stored in artefactos/. TensorFlow is imported lazily so the rest of the package (and the
unit tests with fake services) does not need it.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from . import config

CLASSES = ["cerradura", "porton", "caseta", "controles", "interior"]
ALL_CLASSES = CLASSES + ["otro"]
IMG_SIZE = 160
BACKBONES = ("resnet50", "mobilenetv2", "efficientnetb0")


def build_extractor(name: str):
    """Keras model image (160x160x3, 0-255) -> global-average-pooled feature vector."""
    import tensorflow as tf
    from tensorflow.keras import applications as apps

    builders = {"resnet50": (apps.ResNet50, apps.resnet50.preprocess_input),
                "mobilenetv2": (apps.MobileNetV2, apps.mobilenet_v2.preprocess_input),
                "efficientnetb0": (apps.EfficientNetB0, apps.efficientnet.preprocess_input)}
    if name not in builders:
        raise ValueError(f"unknown backbone: {name}")
    build, prepare = builders[name]
    base = build(weights="imagenet", include_top=False, pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    prepared = tf.keras.layers.Lambda(prepare, name="preprocesar")(inputs)
    return tf.keras.Model(inputs, base(prepared)), base


def to_array(image) -> np.ndarray:
    """PIL image -> float32 array (160, 160, 3) stretched like in the Avance 2 notebook."""
    return np.asarray(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)


@dataclass
class ImagePrediction:
    label: str                  # "otro" when the confidence is below the threshold
    top_label: str              # most probable class, before applying the threshold
    confidence: float
    top: list = field(default_factory=list)

    @property
    def reliable(self) -> bool:
        return self.label != "otro"

    def as_dict(self) -> dict:
        return {"etiqueta": self.label, "etiqueta_modelo": self.top_label, "confianza": round(self.confidence, 3),
                "supera_umbral": self.reliable,
                "top3": [{"etiqueta": k, "confianza": round(p, 3)} for k, p in self.top]}


def decide(prob: np.ndarray, classes: list[str], threshold: float) -> ImagePrediction:
    order = np.argsort(prob)[::-1]
    top_label, conf = str(classes[order[0]]), float(prob[order[0]])
    label = top_label if conf >= threshold else "otro"
    return ImagePrediction(label, top_label, conf, [(str(classes[i]), float(prob[i])) for i in order[:3]])


class ImageClassifier:
    def __init__(self, backbone: str, head, threshold: float = config.IMAGE_THRESHOLD):
        self.backbone, self.head, self.threshold = backbone, head, threshold
        self.extractor, _ = build_extractor(backbone)
        self._lock = threading.Lock()
        self.predict(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8), warmup=True)

    @classmethod
    def load(cls, directory: Path | None = None) -> "ImageClassifier":
        directory = Path(directory or config.ARTIFACTS_DIR)
        meta = json.loads((directory / "vision_meta.json").read_text(encoding="utf-8"))
        backbone = config.VISION_BACKBONE or meta["despliegue"]
        return cls(backbone, joblib.load(directory / f"vision_head_{backbone}.joblib"), meta.get("umbral", config.IMAGE_THRESHOLD))

    def features(self, arrays: np.ndarray) -> np.ndarray:
        with self._lock:
            return np.asarray(self.extractor(arrays, training=False))

    def predict(self, image, warmup: bool = False) -> ImagePrediction:
        arr = image if isinstance(image, np.ndarray) else to_array(image)
        prob = self.head.predict_proba(self.features(arr[None, ...].astype("float32")))[0]
        return decide(prob, list(self.head.classes_), self.threshold)
