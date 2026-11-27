#!/bin/zsh
# Publica la API en un Hugging Face Space (SDK Docker) y le pone el API_TOKEN como secreto.
# Sirve para: dejar la API en la nube con URL fija y gratuita. Se corre una vez y cada vez que cambie el codigo.
#
# Requisitos (los unicos pasos de la persona duena de la cuenta de Hugging Face):
#   1. Tener cuenta en huggingface.co y un token con permiso de escritura: https://huggingface.co/settings/tokens
#   2. Exportar HF_TOKEN=<ese token> y HF_USER=<usuario de Hugging Face> en la terminal (no se guardan en el repo).
#   3. Tener el valor de API_TOKEN en un archivo (por defecto ~/haru-work/pf-deploy/.env, linea API_TOKEN=...).
# Uso: HF_USER=usuario HF_TOKEN=hf_xxx proyecto_final/deploy/publicar_space.sh [nombre-del-space]
set -e
: "${HF_USER:?Falta HF_USER}"; : "${HF_TOKEN:?Falta HF_TOKEN}"
SPACE="${1:-casaguacate-api}"
ENV_FILE="${API_ENV_FILE:-$HOME/haru-work/pf-deploy/.env}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"

# 1) Crea el Space si no existe (si ya existe, la API responde 409 y se sigue).
curl -s -o /dev/null -X POST https://huggingface.co/api/repos/create \
  -H "Authorization: Bearer $HF_TOKEN" -H "Content-Type: application/json" \
  -d "{\"type\":\"space\",\"name\":\"$SPACE\",\"sdk\":\"docker\",\"private\":false}" || true

# 2) Arma el contenido del Space con preparar_contexto.sh (solo lo que necesita la API) y la cabecera del Space.
"$AQUI/preparar_contexto.sh" "$STAGE"
cp "$AQUI/README_space.md" "$STAGE/README.md"

# 3) Sube el secreto API_TOKEN leyendolo del archivo, sin imprimirlo.
export TOKEN_VAL="$(grep -E '^(export )?API_TOKEN=' "$ENV_FILE" | head -1 | sed -E 's/^(export )?API_TOKEN=//; s/^"//; s/"$//')"
python3 - "$HF_USER" "$SPACE" <<'PY'
import json, os, sys, urllib.request
user, space = sys.argv[1], sys.argv[2]
req = urllib.request.Request(f"https://huggingface.co/api/spaces/{user}/{space}/secrets",
    data=json.dumps({"key": "API_TOKEN", "value": os.environ["TOKEN_VAL"]}).encode(), method="POST",
    headers={"Authorization": "Bearer " + os.environ["HF_TOKEN"], "Content-Type": "application/json"})
print("secreto API_TOKEN:", urllib.request.urlopen(req).status)
PY
unset TOKEN_VAL

# 4) Empuja el contenido al repositorio git del Space (esto dispara la construccion).
cd "$STAGE" && git init -q -b main && git add -A
git -c user.name="Despliegue Casa Aguacate" -c user.email="despliegue@localhost" commit -q -m "Despliegue de la API Casa Aguacate"
git push -f "https://$HF_USER:$HF_TOKEN@huggingface.co/spaces/$HF_USER/$SPACE" main
echo "URL publica: https://${HF_USER}-${SPACE}.hf.space  (la construccion tarda unos minutos)"
