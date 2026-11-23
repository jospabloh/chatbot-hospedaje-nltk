"""HTTP client of the Streamlit app for the Flask API, with friendly (bilingual) error messages.

It never raises: every call returns an ApiResult, so the interface can always show something useful
instead of a traceback.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import requests

ERRORS = {
    "es": {
        "conexion": "No pude conectarme con el asistente. Puede que el servidor esté apagado o la dirección haya cambiado. Intenta de nuevo en un momento o escribe directamente a reservations.casaguacate@gmail.com.",
        "timeout": "El asistente tardó demasiado en responder. Intenta de nuevo; si persiste, escribe a reservations.casaguacate@gmail.com.",
        "401": "El asistente no aceptó la conexión (falta o no es válido el token de acceso). Avisa a quien administra la app.",
        "413": "La foto es demasiado grande. Prueba con una de menos de 5 MB.",
        "415": "No pude leer esa foto. Usa una imagen JPG o PNG que no esté dañada.",
        "422": "La foto es demasiado pequeña. Prueba con una imagen más grande.",
        "429": "Hay muchas consultas seguidas. Espera un minuto e intenta de nuevo.",
        "503": "Una parte del asistente no está disponible por ahora. Intenta de nuevo en unos minutos.",
        "5xx": "El asistente tuvo un problema interno. Intenta de nuevo en un momento.",
        "400": "No pude procesar el mensaje: {detalle}",
        "respuesta": "El asistente respondió algo que no pude interpretar. Intenta de nuevo.",
    },
    "en": {
        "conexion": "I could not reach the assistant. The server may be off or its address may have changed. Try again in a moment or write to reservations.casaguacate@gmail.com.",
        "timeout": "The assistant took too long to answer. Try again; if it persists, write to reservations.casaguacate@gmail.com.",
        "401": "The assistant did not accept the connection (the access token is missing or invalid). Let the app administrator know.",
        "413": "The photo is too large. Try one under 5 MB.",
        "415": "I could not read that photo. Use a JPG or PNG image that is not damaged.",
        "422": "The photo is too small. Try a larger image.",
        "429": "Too many requests in a row. Wait a minute and try again.",
        "503": "Part of the assistant is unavailable right now. Try again in a few minutes.",
        "5xx": "The assistant had an internal problem. Try again in a moment.",
        "400": "I could not process the message: {detalle}",
        "respuesta": "The assistant answered something I could not read. Try again.",
    },
}


@dataclass
class ApiResult:
    ok: bool
    data: dict = field(default_factory=dict)
    error: str = ""
    status: int = 0


class ApiClient:
    def __init__(self, base_url: str, token: str = "", timeout: float = 45.0, session=None, token_url: str | None = None):
        """token_url is the only address the token may be sent to (the one set by the server configuration);
        by default it is base_url. If base_url differs, the token is never attached."""
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.token_url = (token_url if token_url is not None else base_url).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def _request(self, method: str, path: str, lang: str = "es", **kwargs) -> ApiResult:
        messages = ERRORS.get(lang, ERRORS["es"])
        send_token = bool(self.token) and self.base_url == self.token_url
        headers = {"X-API-Token": self.token} if send_token else {}
        try:
            response = self.session.request(method, self.base_url + path, headers=headers, timeout=self.timeout,
                                           allow_redirects=False, **kwargs)
        except requests.Timeout:
            return ApiResult(False, error=messages["timeout"])
        except requests.RequestException:
            return ApiResult(False, error=messages["conexion"])
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if response.status_code >= 400:
            if response.status_code >= 500 and response.status_code != 503:
                text = messages["5xx"]
            elif str(response.status_code) in messages and response.status_code != 400:
                text = messages[str(response.status_code)]
            else:
                detail = ((payload or {}).get("error") or {}).get("mensaje", "") if isinstance(payload, dict) else ""
                text = messages["400"].format(detalle=detail or response.status_code)
            return ApiResult(False, error=text, status=response.status_code, data=payload if isinstance(payload, dict) else {})
        if not isinstance(payload, dict):
            return ApiResult(False, error=messages["respuesta"], status=response.status_code)
        return ApiResult(True, data=payload, status=response.status_code)

    def health(self, lang: str = "es") -> ApiResult:
        return self._request("GET", "/health", lang)

    def metrics(self, lang: str = "es") -> ApiResult:
        return self._request("GET", "/metrics", lang)

    def agent(self, text: str, state: dict | None = None, photo: bytes | None = None, filename: str = "foto.jpg",
              lang: str = "es") -> ApiResult:
        import json
        data = {"texto": text, "estado": json.dumps(state or {})}
        files = {"file": (filename, photo)} if photo else None
        return self._request("POST", "/agent", lang, data=data, files=files)
