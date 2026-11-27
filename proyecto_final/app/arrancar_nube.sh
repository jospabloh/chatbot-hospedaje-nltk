#!/bin/sh
# Script de arranque de la API de Cas'aguacate en la nube.
# Sirve para: lanzar gunicorn en el puerto que pide el proveedor (variable PORT, 7860 en Hugging Face Spaces).
# Un solo worker: las metricas y el limitador de tasa viven en memoria (los hilos si se pueden repetir).
# Variables que lee la API: API_TOKEN (secreto del proveedor), RATE_LIMIT_PER_MIN, VISION_BACKBONE, ARTIFACTS_DIR.
set -eu

# Candado de seguridad: sin API_TOKEN la API quedaria abierta a todo internet, asi que se niega a arrancar.
# Se exige una longitud minima de 32 caracteres para evitar tokens triviales (el mensaje no muestra el valor).
: "${API_TOKEN:?Falta API_TOKEN: defínelo como secreto del proveedor}"
if [ "${#API_TOKEN}" -lt 32 ]; then
  echo "API_TOKEN debe tener al menos 32 caracteres; la API no arranca." >&2
  exit 1
fi

exec gunicorn -w 1 --threads 4 --timeout 120 -b "0.0.0.0:${PORT:-7860}" servidor:app
