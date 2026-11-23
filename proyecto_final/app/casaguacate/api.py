"""Flask REST API: /health, /chat, /image, /agent, /metrics.

Design notes
- Application factory: tests inject fake services; production loads the exported artifacts once at start-up.
- Stateless: the conversation state is sent and returned by the client; nothing about the guest is stored.
- Privacy: logs are one JSON line per request WITHOUT the message text or the photo (only length, language,
  intent and action).
- Optional protection for a public tunnel: API_TOKEN (header X-API-Token) and a per-client rate limit.
"""
from __future__ import annotations

import hmac
import json
import logging
import sys
import threading
import time
import uuid
from collections import defaultdict, deque

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from . import config
from .agente import HybridAgent, Services, answer_text
from .metricas import Metrics
from .texto import detect_language
from .validacion import ApiError, validate_image, validate_lang, validate_state, validate_text

KNOWN_PATHS = {"/health", "/chat", "/image", "/agent", "/metrics", "/"}
PUBLIC_PATHS = {"/health", "/"}

logger = logging.getLogger("casaguacate.api")


def configure_logging() -> None:
    if logger.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(config.LOG_LEVEL)
    logger.propagate = False


def log_event(level: int, **fields) -> None:
    logger.log(level, json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}, ensure_ascii=False))


class RateLimiter:
    """Sliding one-minute window per client (in memory: run a single worker process)."""

    def __init__(self, limit: int):
        self.limit, self.hits, self.lock = limit, defaultdict(deque), threading.Lock()

    def allow(self, client: str) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            q = self.hits[client]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.limit:
                return False
            q.append(now)
            if len(self.hits) > 5000:  # forget idle clients
                for key in [k for k, v in self.hits.items() if not v or now - v[-1] > 60]:
                    del self.hits[key]
            return True


def client_id() -> str:
    forwarded = request.headers.get("CF-Connecting-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    return forwarded or request.remote_addr or "desconocido"


def create_app(services: Services | None = None, settings: dict | None = None) -> Flask:
    configure_logging()
    app = Flask(__name__)
    cfg = {"API_TOKEN": config.API_TOKEN, "RATE_LIMIT_PER_MIN": config.RATE_LIMIT_PER_MIN, **(settings or {})}
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_IMAGE_BYTES + 256 * 1024
    app.config["JSON_AS_ASCII"] = False
    app.json.ensure_ascii = False
    metrics, limiter = Metrics(), RateLimiter(cfg["RATE_LIMIT_PER_MIN"])
    if services is None:
        from .servicios import load_services
        services = load_services()
    agent = HybridAgent(services)
    app.extensions["casaguacate"] = {"metrics": metrics, "services": services, "agent": agent}

    def error_response(status: int, code: str, message: str):
        return jsonify({"error": {"codigo": code, "mensaje": message}, "request_id": getattr(g, "request_id", None)}), status

    @app.before_request
    def start_request():
        g.start, g.request_id, g.log = time.perf_counter(), request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12], {}
        if request.path in PUBLIC_PATHS or request.method == "OPTIONS":
            return None
        token = cfg["API_TOKEN"]
        if token and not hmac.compare_digest(request.headers.get("X-API-Token", ""), token):
            raise ApiError(401, "token_invalido", "Falta el token de acceso o no es valido (cabecera X-API-Token).")
        if not limiter.allow(client_id()):
            metrics.count("limite_de_peticiones", "excedido")
            raise ApiError(429, "demasiadas_peticiones", "Demasiadas peticiones. Espera un minuto e intenta de nuevo.")
        return None

    @app.after_request
    def finish_request(response):
        ms = (time.perf_counter() - g.get("start", time.perf_counter())) * 1000
        path = request.path if request.path in KNOWN_PATHS else "otras"
        metrics.record(path, response.status_code, ms)
        response.headers["X-Request-ID"] = g.get("request_id", "")
        log_event(logging.ERROR if response.status_code >= 500 else logging.INFO, evento="peticion", request_id=g.get("request_id"),
                  metodo=request.method, ruta=path, estado=response.status_code, ms=round(ms, 1), **g.get("log", {}))
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return error_response(exc.status, exc.code, exc.message)

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        known = {404: ("ruta_no_encontrada", "La ruta solicitada no existe."),
                 405: ("metodo_no_permitido", "Metodo no permitido para esta ruta."),
                 413: ("archivo_demasiado_grande", "La peticion es demasiado grande."),
                 400: ("peticion_invalida", "La peticion no es valida.")}
        code, message = known.get(exc.code, ("error_http", exc.name))
        return error_response(exc.code or 500, code, message)

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        log_event(logging.ERROR, evento="error", request_id=g.get("request_id"), tipo=type(exc).__name__, ruta=request.path)
        return error_response(500, "error_interno", "Ocurrio un error interno. Intenta de nuevo en un momento.")

    def body() -> dict:
        """JSON object or form fields, whichever the client sent."""
        if request.is_json:
            data = request.get_json(silent=True)
            if not isinstance(data, dict):
                raise ApiError(400, "json_invalido", "El cuerpo debe ser un objeto JSON.")
            return data
        if request.mimetype in ("multipart/form-data", "application/x-www-form-urlencoded"):
            return request.form.to_dict()
        raise ApiError(415, "tipo_no_soportado", "Usa application/json o multipart/form-data.")

    def uploaded_image():
        file = request.files.get("file")
        return validate_image(file) if file is not None else None

    @app.get("/")
    def index():
        return jsonify({"servicio": "API del asistente de Cas'aguacate", "version": config.VERSION,
                        "endpoints": {"GET /health": "estado", "POST /chat": "chatbot NLP (texto)",
                                      "POST /image": "clasificador visual (foto)", "POST /agent": "agente (texto y foto opcional)",
                                      "GET /metrics": "metricas basicas"}})

    @app.get("/health")
    def health():
        vision = services.vision
        ok = services.intents is not None
        return jsonify({"estado": "ok" if ok and vision is not None else "degradado", "version": config.VERSION,
                        "modelos": {"intenciones": ok, "tono": services.tone is not None,
                                    "vision": getattr(vision, "backbone", bool(vision))},
                        "uptime_s": round(time.time() - metrics.started, 1)}), (200 if ok else 503)

    @app.post("/chat")
    def chat():
        data = body()
        text = validate_text(data.get("texto"))
        lang = validate_lang(data.get("idioma")) or detect_language(text)
        result = answer_text(services, text, lang)
        g.log = {"idioma": lang, "chars": len(text), "intencion": result["intencion"]["etiqueta"]}
        return jsonify(result)

    @app.post("/image")
    def image():
        if services.vision is None:
            raise ApiError(503, "modelo_no_disponible", "El clasificador visual no esta disponible en este momento.")
        if "file" not in request.files:
            raise ApiError(400, "archivo_requerido", "Falta el archivo en el campo 'file' (multipart/form-data).")
        img = validate_image(request.files["file"])
        started = time.perf_counter()
        prediction = services.vision.predict(img)
        result = {"modelo": getattr(services.vision, "backbone", "desconocido"), **prediction.as_dict(),
                  "latencia_ms": round((time.perf_counter() - started) * 1000, 1)}
        g.log = {"clase": prediction.label, "ancho": img.size[0], "alto": img.size[1]}
        return jsonify(result)

    @app.post("/agent")
    def agent_endpoint():
        data = body()
        text = validate_text(data.get("texto"), required=False)
        state = validate_state(data.get("estado"))
        img = uploaded_image() if not request.is_json else None
        if not text and img is None:
            raise ApiError(400, "entrada_vacia", "Envia al menos 'texto' o una foto en 'file'.")
        if img is not None and services.vision is None:
            raise ApiError(503, "modelo_no_disponible", "El clasificador visual no esta disponible en este momento.")
        result = agent.run(text, img, state)
        decision = result.as_dict()
        metrics.count("accion", result.action)
        metrics.count("idioma", result.lang)
        if result.notify_host:  # the notification channel (e-mail / WhatsApp) is future work: here it is flagged, logged and counted
            metrics.count("avisar_anfitriona", "si")
            log_event(logging.WARNING, evento="aviso_anfitriona", request_id=g.get("request_id"), accion=result.action)
        g.log = {"idioma": result.lang, "chars": len(text), "tiene_foto": img is not None, "accion": result.action,
                 "intencion": result.final_intent}
        return jsonify(decision)

    @app.get("/metrics")
    def metrics_endpoint():
        return jsonify(metrics.snapshot())

    return app
