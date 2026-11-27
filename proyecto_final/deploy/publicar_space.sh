#!/bin/bash
# Publica la API de Cas'aguacate en un Hugging Face Space (SDK Docker) y le pone el API_TOKEN como secreto.
# Sirve para: dejar la API en la nube con URL fija y gratuita. Se corre una vez y cada vez que cambie el codigo.
#
# Requisitos (lo unico que hace la persona duena de la cuenta de Hugging Face):
#   1. Tener cuenta en huggingface.co y correr una vez `hf auth login` (token con permiso de escritura; el token
#      se pega en el prompt de ese comando, no queda en el historial). Alternativa: variable HF_TOKEN.
#   2. Tener el valor de API_TOKEN en un archivo (por defecto ~/haru-work/pf-deploy/.env, linea API_TOKEN=...).
# Uso: proyecto_final/deploy/publicar_space.sh [nombre-del-space]
# El usuario de Hugging Face se toma de la sesion (whoami). Si falta huggingface_hub, se crea un venv aparte.
set -euo pipefail

SPACE="${1:-casaguacate-api}"
ENV_FILE="${API_ENV_FILE:-$HOME/haru-work/pf-deploy/.env}"
AQUI="$(cd "$(dirname "$0")" && pwd)"

# Lee y limpia API_TOKEN del archivo .env: sin "export", sin comillas, sin espacios ni \r; exige >= 32 caracteres.
# Nunca imprime el valor.
[ -r "$ENV_FILE" ] || { echo "ERROR: no existe o no se puede leer $ENV_FILE" >&2; exit 1; }
TOKEN_VAL="$(grep -E '^[[:space:]]*(export[[:space:]]+)?API_TOKEN=' "$ENV_FILE" | head -1 \
  | sed -E 's/^[[:space:]]*(export[[:space:]]+)?API_TOKEN=//' | tr -d "\r\"' \t")" || true
if [ "${#TOKEN_VAL}" -lt 32 ]; then
  echo "ERROR: API_TOKEN falta en $ENV_FILE o tiene menos de 32 caracteres." >&2; exit 1
fi

# Python con huggingface_hub: el de la variable HF_PYTHON, el del sistema si ya lo tiene, o un venv propio.
PY="${HF_PYTHON:-python3}"
if ! "$PY" -c "import huggingface_hub" 2>/dev/null; then
  VENV="${HF_VENV:-$HOME/haru-work/pf-hf-venv}"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/python" -c "import huggingface_hub" 2>/dev/null || "$VENV/bin/pip" install -q huggingface_hub
  PY="$VENV/bin/python"
fi

# Carpeta temporal del Space; se borra siempre al salir (con exito o con error).
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
"$AQUI/preparar_contexto.sh" "$STAGE/space"
cp "$AQUI/README_space.md" "$STAGE/space/README.md"

# El valor del secreto viaja solo por entorno de este comando (no por argumentos).
TOKEN_VAL="$TOKEN_VAL" "$PY" "$AQUI/publicar_space.py" "$STAGE/space" "$SPACE"
