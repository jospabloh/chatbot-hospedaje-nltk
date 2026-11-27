#!/bin/sh
# Script de arranque de la API en la nube.
# Sirve para: lanzar gunicorn en el puerto que pide el proveedor (variable PORT, 7860 en Hugging Face Spaces).
# Un solo worker: las metricas y el limitador de tasa viven en memoria (los hilos si se pueden repetir).
# Variables que lee la API: API_TOKEN (secreto del proveedor), RATE_LIMIT_PER_MIN, VISION_BACKBONE, ARTIFACTS_DIR.
exec gunicorn -w 1 --threads 4 --timeout 120 -b "0.0.0.0:${PORT:-7860}" servidor:app
