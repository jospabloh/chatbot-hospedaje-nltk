"""Smoke test against a RUNNING API (local, or the public Cloudflare Tunnel URL after deploying).

    python scripts/prueba_humo.py --api http://127.0.0.1:5000
    python scripts/prueba_humo.py --api https://<tu-tunel>.trycloudflare.com --token <API_TOKEN>

Checks health, the three inference endpoints, error handling and metrics; prints one line per check and exits
with 1 if any fails. Use it as the deployment evidence ("capturas de despliegue").
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import requests
from PIL import Image

DATA = Path(__file__).resolve().parents[3] / "avance2" / "datos"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:5000")
    ap.add_argument("--token", default="")
    args = ap.parse_args()
    base, headers = args.api.rstrip("/"), ({"X-API-Token": args.token} if args.token else {})
    results = []

    def check(name, fn):
        started = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(ok)
        print(f"{'OK   ' if ok else 'FALLA'} {name} ({(time.perf_counter() - started) * 1000:.0f} ms) {detail}")

    photo = io.BytesIO()
    Image.open(DATA / "fotos_reales" / "real_1.jpg").convert("RGB").save(photo, format="JPEG")

    def health():
        r = requests.get(f"{base}/health", timeout=20)
        return r.status_code == 200 and r.json()["estado"] in ("ok", "degradado"), f"-> {r.json()}"

    def chat():
        r = requests.post(f"{base}/chat", json={"texto": "¿Cuánto cuesta la noche?"}, headers=headers, timeout=30)
        return r.status_code == 200 and r.json()["intencion"]["etiqueta"] == "tarifas", f"-> {r.json()['intencion']['etiqueta']}"

    def image():
        r = requests.post(f"{base}/image", files={"file": ("real_1.jpg", photo.getvalue(), "image/jpeg")}, headers=headers, timeout=60)
        return r.status_code == 200, f"-> {r.json().get('etiqueta')} ({r.json().get('confianza')}) con {r.json().get('modelo')}"

    def agent_text():
        r = requests.post(f"{base}/agent", json={"texto": "What time is check-in?"}, headers=headers, timeout=30)
        d = r.json()["decision"]
        return r.status_code == 200 and d["accion"] == "RESPONDER" and d["idioma"] == "en", f"-> {d['accion']} ({d['idioma']})"

    def agent_photo():
        r = requests.post(f"{base}/agent", data={"texto": "mira esto", "estado": "{}"}, headers=headers, timeout=60,
                          files={"file": ("real_1.jpg", photo.getvalue(), "image/jpeg")})
        names = [s["componente"] for s in r.json()["traza"]]
        return r.status_code == 200 and "Clasificador visual" in names, f"-> {r.json()['decision']['accion']}, traza: {names}"

    def agent_urgent():
        r = requests.post(f"{base}/agent", json={"texto": "hay humo en la habitación"}, headers=headers, timeout=30)
        return r.json()["decision"]["accion"] == "ESCALAR_URGENTE", f"-> {r.json()['decision']['accion']}"

    def bad_input():
        r = requests.post(f"{base}/agent", json={"texto": "x" * 600}, headers=headers, timeout=30)
        return r.status_code == 400 and "error" in r.json(), f"-> {r.status_code} {r.json()['error']['codigo']}"

    def bad_photo():
        r = requests.post(f"{base}/image", files={"file": ("a.jpg", b"no soy imagen", "image/jpeg")}, headers=headers, timeout=30)
        return r.status_code == 415, f"-> {r.status_code} {r.json()['error']['codigo']}"

    def metrics():
        r = requests.get(f"{base}/metrics", headers=headers, timeout=20)
        return r.status_code == 200 and r.json()["peticiones_total"] >= 1, f"-> {json.dumps({k: r.json()[k] for k in ('peticiones_total', 'errores_5xx_total')})}"

    for name, fn in [("GET /health", health), ("POST /chat", chat), ("POST /image", image), ("POST /agent texto (EN)", agent_text),
                     ("POST /agent texto + foto", agent_photo), ("POST /agent urgencia", agent_urgent),
                     ("Validacion: texto demasiado largo", bad_input), ("Validacion: foto corrupta", bad_photo), ("GET /metrics", metrics)]:
        check(name, fn)
    print(f"\n{sum(results)} de {len(results)} comprobaciones correctas")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
