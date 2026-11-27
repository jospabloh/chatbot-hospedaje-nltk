#!/bin/bash
# Prueba de humo de la API de Cas'aguacate contra su URL publica (nube).
# Sirve para: comprobar de punta a punta que la API desplegada responde y exige token, y medir latencia p50/p95.
# Cubre: /health, /chat con token (check-in 15:00 y check-out 11:00), /chat sin token (401), /image con una foto real
# y 30 peticiones a /chat. Guarda la salida en proyecto_final/app/evidencias/prueba_humo_nube.txt.
# El token se lee del .env (API_ENV_FILE, por defecto ~/haru-work/pf-deploy/.env) y NUNCA se imprime ni va en argumentos.
# Uso: proyecto_final/deploy/verificar_nube.sh https://<usuario>-casaguacate-api.hf.space
set -euo pipefail

URL="${1:?Uso: verificar_nube.sh <URL>}"; URL="${URL%/}"
ENV_FILE="${API_ENV_FILE:-$HOME/haru-work/pf-deploy/.env}"
AQUI="$(cd "$(dirname "$0")" && pwd)"
SALIDA="$AQUI/../app/evidencias/prueba_humo_nube.txt"
FOTO="$AQUI/../../avance2/datos/fotos_reales/real_2.jpg"

[ -r "$ENV_FILE" ] || { echo "ERROR: no existe o no se puede leer $ENV_FILE" >&2; exit 1; }
[ -r "$FOTO" ] || { echo "ERROR: falta la foto $FOTO" >&2; exit 1; }
TOKEN_VAL="$(grep -E '^[[:space:]]*(export[[:space:]]+)?API_TOKEN=' "$ENV_FILE" | head -1 \
  | sed -E 's/^[[:space:]]*(export[[:space:]]+)?API_TOKEN=//' | tr -d "\r\"' \t")" || true
[ "${#TOKEN_VAL}" -ge 32 ] || { echo "ERROR: API_TOKEN falta o es corto en $ENV_FILE" >&2; exit 1; }
export TOKEN_VAL URL FOTO

mkdir -p "$(dirname "$SALIDA")"
# Todo el bloque se muestra y se guarda; el token solo viaja por entorno/stdin de curl, no sale en la salida.
{
echo "Prueba de humo en la nube, $(date '+%Y-%m-%d %H:%M:%S %z')"; echo "URL: $URL"
fallos=0
ok() { echo "OK    $1"; }
mal() { echo "FALLA $1"; fallos=$((fallos+1)); }
# curl con el encabezado del token leido por stdin (--config -), asi no aparece en la lista de procesos.
conf() { printf 'header = "X-API-Token: %s"\n' "$TOKEN_VAL"; }

# 1) /health: el Space puede estar dormido; se espera hasta ~4 min a que despierte.
code=000
for i in $(seq 1 24); do
  code="$(curl -s -o /tmp/pfn_health.json -w '%{http_code}' --max-time 20 "$URL/health" || true)"
  [ "$code" = 200 ] && break; sleep 10
done
if [ "$code" = 200 ] && grep -q '"estado":"ok"' /tmp/pfn_health.json && grep -q '"intenciones":true' /tmp/pfn_health.json; then
  ok "GET /health 200 con modelos cargados: $(cat /tmp/pfn_health.json | tr -d '\n')"; else mal "GET /health (HTTP $code)"; fi

# 2) /chat con token
conf | curl -s -K - --max-time 60 -o /tmp/pfn_chat.json -w '%{http_code}' "$URL/chat" -H 'Content-Type: application/json' \
  -d '{"texto":"¿A qué hora es el check-in?"}' > /tmp/pfn_code.txt || true
code="$(cat /tmp/pfn_code.txt)"
if [ "$code" = 200 ] && grep -q '15:00' /tmp/pfn_chat.json && grep -q '11:00' /tmp/pfn_chat.json; then
  ok "POST /chat con token 200, respuesta con check-in 15:00 y check-out 11:00"; else mal "POST /chat con token (HTTP $code)"; fi

# 3) /chat sin token debe dar 401
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$URL/chat" -H 'Content-Type: application/json' -d '{"texto":"hola"}' || true)"
[ "$code" = 401 ] && ok "POST /chat sin token 401" || mal "POST /chat sin token (HTTP $code, se esperaba 401)"

# 4) /image con foto real
conf | curl -s -K - --max-time 120 -o /tmp/pfn_img.json -w '%{http_code}' "$URL/image" -F file=@"$FOTO" > /tmp/pfn_code.txt || true
code="$(cat /tmp/pfn_code.txt)"
if [ "$code" = 200 ] && grep -q '"etiqueta"' /tmp/pfn_img.json; then
  ok "POST /image 200: $(cut -c1-120 /tmp/pfn_img.json | tr -d '\n')..."; else mal "POST /image (HTTP $code)"; fi

# 5) Latencia: 30 peticiones a /chat (el token llega por entorno, no por argumentos)
python3 - <<'PY' || fallos=$((fallos+1))
import json, os, time, urllib.request
ts = []
for _ in range(30):
    r = urllib.request.Request(os.environ["URL"] + "/chat", data=json.dumps({"texto": "¿A qué hora es el check-in?"}).encode(),
                               headers={"X-API-Token": os.environ["TOKEN_VAL"], "Content-Type": "application/json"})
    t = time.time(); urllib.request.urlopen(r, timeout=60).read(); ts.append((time.time() - t) * 1000)
ts.sort()
print("OK    Latencia /chat (n=30): p50 %.0f ms, p95 %.0f ms, max %.0f ms" % (ts[14], ts[28], ts[-1]))
PY
rm -f /tmp/pfn_health.json /tmp/pfn_chat.json /tmp/pfn_img.json /tmp/pfn_code.txt
echo "Resultado: $([ "$fallos" = 0 ] && echo TODO OK || echo "$fallos FALLA(S)")"
[ "$fallos" = 0 ]
} 2>&1 | tee "$SALIDA"
exit "${PIPESTATUS[0]}"
