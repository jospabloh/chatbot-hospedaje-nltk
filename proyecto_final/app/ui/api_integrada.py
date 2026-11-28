"""Integrated mode: the Flask REST API runs inside the same process as the Streamlit app.

When no API_URL is configured, the UI starts the SAME API (casaguacate.api.create_app) in a background thread bound
to 127.0.0.1 on a free port, protected by a random per-process token, and keeps consuming it over real HTTP with
the usual ApiClient. Nothing else changes: routes, validation, rate limit, metrics and /health are the real ones.
With API_URL configured (external API) nothing is started and this module is not even imported by the heavy parts.
"""
from __future__ import annotations

import logging
import secrets
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]


@dataclass
class EmbeddedApi:
    """A running embedded API: its loopback URL, its access token and the server/thread that serve it."""

    url: str
    token: str
    server: object
    thread: threading.Thread

    def alive(self) -> bool:
        """True while the serving thread runs; the app uses it to restart the API if the thread ever dies."""
        return self.thread.is_alive()

    def stop(self) -> None:
        """Shuts the server down (used by the tests; in production the daemon thread ends with the process)."""
        self.server.shutdown()
        self.thread.join(timeout=5)


def build_services():
    """Real services for the embedded API, with TensorFlow loaded lazily (only on the first photo).

    Called by start_embedded_api when no services are injected; tests replace it with fakes.
    """
    from casaguacate.servicios import load_services

    return load_services(lazy_vision=True)


def start_embedded_api(services=None, rate_limit_per_min: int | None = None) -> EmbeddedApi:
    """Starts the Flask API in a daemon thread on 127.0.0.1:<free port> and returns how to reach it.

    A random token is generated for this process and given both to the API (settings override, so the environment
    is not touched) and, through the returned object, to the UI client: the API is never open, even on loopback.
    Used by streamlit_app.embedded_api (cached once per process) and by the integrated-mode tests.
    """
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    from werkzeug.serving import make_server

    from casaguacate.api import create_app

    token = secrets.token_urlsafe(32)
    settings = {"API_TOKEN": token}
    if rate_limit_per_min is not None:
        settings["RATE_LIMIT_PER_MIN"] = rate_limit_per_min
    app = create_app(services if services is not None else build_services(), settings)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # the API already logs one JSON line per request
    server = make_server("127.0.0.1", 0, app, threaded=True)   # port 0: the OS picks a free port
    thread = threading.Thread(target=server.serve_forever, name="api-integrada", daemon=True)
    thread.start()
    return EmbeddedApi(f"http://127.0.0.1:{server.server_port}", token, server, thread)
