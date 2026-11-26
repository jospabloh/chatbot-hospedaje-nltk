"""Basic in-memory monitoring: request counts, errors and latency per endpoint (read at /metrics)."""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 500):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: Counter = Counter()                  # (path, status) -> n
        self.latency: dict = defaultdict(lambda: deque(maxlen=window))   # path -> ms
        self.events: Counter = Counter()                    # free counters: actions, languages, rate limits...

    def record(self, path: str, status: int, ms: float) -> None:
        with self._lock:
            self.requests[(path, status)] += 1
            self.latency[path].append(ms)

    def count(self, name: str, value: str) -> None:
        with self._lock:
            self.events[f"{name}:{value}"] += 1

    def snapshot(self) -> dict:
        with self._lock:
            per_path: dict = {}
            for (path, status), n in self.requests.items():
                entry = per_path.setdefault(path, {"peticiones": 0, "errores_cliente_4xx": 0, "errores_servidor_5xx": 0,
                                                   "por_codigo": {}})
                entry["peticiones"] += n
                entry["por_codigo"][str(status)] = n
                if 400 <= status < 500:
                    entry["errores_cliente_4xx"] += n
                elif status >= 500:
                    entry["errores_servidor_5xx"] += n
            for path, values in self.latency.items():
                arr = np.array(values)
                per_path[path]["latencia_ms"] = {"media": round(float(arr.mean()), 1), "p50": round(float(np.percentile(arr, 50)), 1),
                                                 "p95": round(float(np.percentile(arr, 95)), 1), "max": round(float(arr.max()), 1),
                                                 "muestras": int(arr.size)}
            total = sum(v["peticiones"] for v in per_path.values())
            return {"uptime_s": round(time.time() - self.started, 1), "peticiones_total": total,
                    "errores_5xx_total": sum(v["errores_servidor_5xx"] for v in per_path.values()),
                    "errores_4xx_total": sum(v["errores_cliente_4xx"] for v in per_path.values()),
                    "por_ruta": per_path, "eventos": dict(self.events)}
