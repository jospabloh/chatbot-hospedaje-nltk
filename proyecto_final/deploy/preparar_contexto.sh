#!/bin/sh
# Arma la carpeta de construccion de la imagen de la API (contexto de Docker / contenido del Space).
# Sirve para: copiar solo lo que la API necesita en runtime (sin pruebas, UI ni evidencias) junto con el Dockerfile.
# Uso: proyecto_final/deploy/preparar_contexto.sh <carpeta-destino>
#   Construir local:  docker build -t casaguacate-api <carpeta-destino>
set -e
DEST="${1:?Falta carpeta destino}"
AQUI="$(cd "$(dirname "$0")" && pwd)"; APP="$AQUI/../app"; RAIZ="$AQUI/../.."
mkdir -p "$DEST"
cp -R "$APP/casaguacate" "$APP/artefactos" "$DEST/"
cp "$APP/servidor.py" "$APP/requirements.txt" "$APP/arrancar_nube.sh" "$RAIZ/chatbot_hospedaje.py" "$DEST/"
cp "$AQUI/Dockerfile" "$DEST/Dockerfile"
find "$DEST" -name __pycache__ -type d -prune -exec rm -rf {} +
