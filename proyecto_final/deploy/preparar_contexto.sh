#!/bin/sh
# Arma la carpeta de construccion de la imagen de la API de Cas'aguacate (contexto de Docker / contenido del Space).
# Sirve para: copiar solo lo que la API necesita en runtime (sin pruebas, UI, evidencias, .DS_Store ni *.pyc)
# junto con el Dockerfile. Exige una carpeta destino vacia o inexistente para no mezclar archivos viejos.
# Uso: proyecto_final/deploy/preparar_contexto.sh <carpeta-destino>
#   Construir local:  docker build -t casaguacate-api <carpeta-destino>
set -eu
DEST="${1:?Falta carpeta destino}"
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST")" ]; then
  echo "La carpeta destino debe estar vacia: $DEST" >&2
  exit 1
fi
AQUI="$(cd "$(dirname "$0")" && pwd)"; APP="$AQUI/../app"; RAIZ="$AQUI/../.."
mkdir -p "$DEST"
cp -R "$APP/casaguacate" "$APP/artefactos" "$DEST/"
cp "$APP/servidor.py" "$APP/requirements.txt" "$APP/arrancar_nube.sh" "$RAIZ/chatbot_hospedaje.py" "$DEST/"
cp "$AQUI/Dockerfile" "$DEST/Dockerfile"
# Solo se despliega la cabeza de vision EfficientNetB0 (VISION_BACKBONE por defecto); las otras no se usan.
rm -f "$DEST/artefactos/vision_head_resnet50.joblib" "$DEST/artefactos/vision_head_mobilenetv2.joblib"
find "$DEST" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$DEST" \( -name .DS_Store -o -name '*.pyc' \) -type f -delete
