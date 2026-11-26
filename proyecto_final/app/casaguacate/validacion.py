"""Input validation for the REST API. Every failure raises ApiError with a stable code and a clear message."""
from __future__ import annotations

import io
import json
import re

from PIL import Image, ImageOps, UnidentifiedImageError

from . import config
from .agente import ConversationState

Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS  # Pillow raises DecompressionBombError above 2x this value
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def validate_text(value, required: bool = True) -> str:
    if value is None:
        if required:
            raise ApiError(400, "texto_requerido", "Falta el campo 'texto'.")
        return ""
    if not isinstance(value, str):
        raise ApiError(400, "texto_invalido", "El campo 'texto' debe ser una cadena de texto.")
    cleaned = _CONTROL.sub("", value).strip()
    if required and not cleaned:
        raise ApiError(400, "texto_vacio", "El campo 'texto' no puede estar vacio.")
    if len(cleaned) > config.MAX_TEXT_CHARS:
        raise ApiError(400, "texto_demasiado_largo", f"El texto excede {config.MAX_TEXT_CHARS} caracteres.")
    return cleaned


def validate_lang(value) -> str | None:
    if value in (None, ""):
        return None
    if value not in ("es", "en"):
        raise ApiError(400, "idioma_invalido", "El campo 'idioma' debe ser 'es' o 'en'.")
    return value


def validate_state(value) -> ConversationState:
    if isinstance(value, str):
        if not value.strip():
            return ConversationState()
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ApiError(400, "estado_invalido", "El campo 'estado' no es JSON valido.")
    try:
        return ConversationState.from_dict(value)
    except (ValueError, TypeError) as exc:
        raise ApiError(400, "estado_invalido", str(exc))


def validate_image(file_storage) -> Image.Image:
    """Checks extension, size, real image content (not just the extension) and dimensions."""
    name = (file_storage.filename or "").strip()
    if not name:
        raise ApiError(400, "archivo_sin_nombre", "El archivo no tiene nombre.")
    if "." not in name or name.rsplit(".", 1)[1].lower() not in config.ALLOWED_IMAGE_EXTENSIONS:
        raise ApiError(415, "formato_no_permitido", "Formato no permitido. Usa JPG, JPEG o PNG.")
    data = file_storage.read(config.MAX_IMAGE_BYTES + 1)
    if not data:
        raise ApiError(400, "archivo_vacio", "El archivo esta vacio.")
    if len(data) > config.MAX_IMAGE_BYTES:
        raise ApiError(413, "archivo_demasiado_grande", f"La imagen excede {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        if probe.format not in ("JPEG", "PNG"):
            raise ApiError(415, "formato_no_permitido", "El contenido no es una imagen JPG o PNG.")
        probe.verify()
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, ValueError):
        raise ApiError(415, "imagen_corrupta", "No se pudo leer la imagen: el archivo esta danado o no es una imagen.")
    if min(image.size) < config.MIN_IMAGE_SIDE:
        raise ApiError(422, "imagen_muy_pequena", f"La imagen debe medir al menos {config.MIN_IMAGE_SIDE} px por lado.")
    return image
