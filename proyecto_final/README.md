# Proyecto Final: asistente de check-in de Cas'aguacate

Aplicación web con un agente híbrido para el check-in de un hospedaje en Costa Rica. Atiende en español e inglés, clasifica la intención del huésped, reconoce fotos de la entrada y decide cuándo responder, pedir más información o avisar a la anfitriona.

Curso: Desarrollo de aplicaciones con IA (Tecmilenio). Todo el código vive en `proyecto_final/app/`.

## Qué incluye

| Componente | Dónde | Qué hace |
|---|---|---|
| Núcleo de inferencia | `app/casaguacate/texto.py`, `vision.py` | Pipeline del Avance 2 (TF-IDF + SVM calibrado por idioma) y clasificador de fotos con backbone congelado |
| Agente híbrido | `app/casaguacate/reglas.py`, `agente.py` | Capa reactiva (reglas de seguridad) y capa deliberativa (estado de conversación y política) |
| API REST Flask | `app/casaguacate/api.py`, `app/servidor.py` | `/health`, `/chat`, `/image`, `/agent`, `/metrics` con validación, códigos de error y logs estructurados |
| Interfaz Streamlit | `app/ui/streamlit_app.py` | Chat bilingüe, foto, traza del agente, errores amigables; consume la API |
| Pruebas | `app/tests/` | Unitarias, de integración y de regresión contra el Avance 2 |
| Evidencias | `app/evidencias/` | Salidas de pytest, entrenamiento, prueba de humo y regresión |
| Diagramas | `app/docs/` | Flujo de coordinación (Mermaid) y arquitectura final (PNG) |

Estructura:

```
proyecto_final/
  README.md
  app/
    casaguacate/   nucleo (config, texto, vision, reglas, agente, api, validacion, metricas, servicios)
    artefactos/    modelos ligeros (.joblib) y metricas (.json)
    scripts/       entrenamiento, prueba de humo, figura de arquitectura
    ui/            streamlit_app.py, cliente_api.py, requirements.txt
    tests/         pytest
    evidencias/    resultados guardados
    docs/          flujo_coordinacion.md, arquitectura_final.png
    servidor.py    punto de entrada de la API
```

## Cómo funciona

El agente sigue el ciclo percibir, decidir, actuar.

1. Percibir: normaliza el texto, detecta idioma, clasifica la intención (texto) y la foto (visión).
2. Capa reactiva: reglas con prioridad fija que no dependen de los modelos. En orden: urgencia (fuego, humo, emergencia médica, remite al 911), datos sensibles (tarjeta o CVV), petición de hablar con una persona, política de la anfitriona (early check-in y early o late check-out; la llegada tarde la contesta el clasificador) y fuera de alcance.
3. Capa deliberativa: combina texto, imagen y estado (idioma, turnos, fallos, si ya pidió foto, intención pendiente) y elige una acción: `RESPONDER`, `RESPONDER_Y_AVISAR`, `PEDIR_FOTO`, `ACLARAR`, `ESCALAR`, `ESCALAR_URGENTE`, `PROTEGER_DATOS`.
4. Actuar: arma la respuesta bilingüe con los datos reales de la base de conocimiento del repo y devuelve la traza de decisiones.

La API no guarda nada del huésped: el estado de conversación viaja en cada petición y regresa en la respuesta. Los logs no contienen el texto del huésped. Diagrama completo en `app/docs/flujo_coordinacion.md`; arquitectura final en `app/docs/arquitectura_final.png`.

Umbrales: texto 0.40, texto ambiguo 0.60, imagen 0.60, tono 0.60. Máximo 2 aclaraciones fallidas antes de escalar.

## Ejecutar en local

Requiere Python 3.9 o superior.

```bash
cd proyecto_final/app
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # incluye API, UI y pytest

# Opcional: reentrenar (los artefactos ya vienen en el repo)
python scripts/entrenar_texto.py
python scripts/entrenar_vision.py --crops <carpeta_con_recortes_reales>

# API
python servidor.py                          # http://127.0.0.1:5000
# o, como en despliegue:
gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 servidor:app

# UI (otra terminal)
API_URL=http://127.0.0.1:5000 streamlit run ui/streamlit_app.py

# Pruebas
pytest
python scripts/prueba_humo.py --api http://127.0.0.1:5000
```

Los pesos del backbone de visión se descargan con `keras.applications` la primera vez (internet requerido). Si no se pueden cargar, la API arranca igual, `/health` lo informa y las fotos se responden con un mensaje amable.

Ejemplos:

```bash
curl -s localhost:5000/chat -H 'Content-Type: application/json' \
  -d '{"texto": "a que hora es el check-in?"}'

curl -s localhost:5000/image -F file=@foto.jpg
```

## Variables de entorno

| Variable | Dónde | Por defecto | Uso |
|---|---|---|---|
| `API_URL` | UI | `http://127.0.0.1:5000` | URL de la API Flask |
| `ALLOW_API_URL_OVERRIDE` | UI | vacío | Solo desarrollo local: con valor `1` y sin `API_TOKEN` muestra un campo para cambiar la URL. En la UI pública no existe ese campo |
| `API_TOKEN` | UI y API | vacío | Si la API lo define, exige el encabezado `X-API-Token`; la UI lo envía |
| `RATE_LIMIT_PER_MIN` | API | 60 | Peticiones por minuto. Con token válido se cuenta por huésped (encabezado `X-Session-Id`, que la UI genera por sesión); sin token, por IP (`CF-Connecting-IP`). `/metrics` tiene su propio contador y `/health` no cuenta |
| `LOG_LEVEL` | API | `INFO` | Nivel de logs JSON |
| `VISION_BACKBONE` | API | `efficientnetb0` | `resnet50`, `mobilenetv2` o `efficientnetb0` |
| `ARTIFACTS_DIR` | API | `artefactos/` | Carpeta de modelos |
| `HOST`, `PORT` | `servidor.py` | `127.0.0.1`, `5000` | Dirección de escucha |

En Streamlit Community Cloud, `API_URL` y `API_TOKEN` se definen como Secrets. La URL de la API sale solo de la configuración del servidor, y el token solo se envía a esa URL (el cliente no sigue redirecciones).

## API

| Endpoint | Método | Descripción |
|---|---|---|
| `/health` | GET | Estado y modelos cargados |
| `/chat` | POST JSON | `{texto, idioma?}` devuelve respuesta, intención e idioma |
| `/image` | POST multipart | Campo `file` (jpg, jpeg, png, máx. 5 MB) devuelve clase y confianza |
| `/agent` | POST JSON o multipart | `texto`, `estado` y foto opcional en `file` (solo multipart); devuelve acción, respuesta, traza y el estado actualizado |
| `/metrics` | GET | Conteos, latencia p50 y p95 |

Errores con código estable y mensaje en español: 400, 401, 413, 415, 422, 429, 503 y 500. Cada respuesta lleva un `X-Request-Id`.

## Resultados

### Clasificador de intenciones (texto)

TF-IDF (palabras 1-2 gramas y caracteres 2-5) con LinearSVC calibrado, un modelo por idioma, semilla 42, partición 80/20 estratificada. Reproduce el Avance 2.

| Idioma | Frases | Accuracy | F1 macro | F1 macro CV (5) | Cobertura con umbral 0.40 | Accuracy cubierta |
|---|---|---|---|---|---|---|
| ES | 711 | 0.818 | 0.792 | 0.704 ± 0.041 | 0.804 | 0.887 |
| EN | 375 | 0.773 | 0.777 | 0.700 ± 0.056 | 0.720 | 0.833 |

El corpus incluye la intención `cancelacion` (cancelar, reembolso, cambio de fechas). Al agregarla cambió la partición de prueba (ES 137 a 143 frases, EN 70 a 75), así que la accuracy de prueba no es comparable uno a uno con la versión anterior (ES 0.752). La medida estable entre versiones es el F1 macro con validación cruzada de 5 particiones (ES 0.706 a 0.704, EN 0.669 a 0.700); es la que conviene citar en el documento técnico. Las cifras salen de `app/artefactos/metricas_texto.json`.

### Visión: ResNet50 (referencia) contra modelo de despliegue

Backbone congelado a 160x160 con cabeza StandardScaler + regresión logística (6 clases, incluida `otro`). Prueba Wikimedia: 39 imágenes. Fotos reales: 44 recortes con validación dejando una toma fuera (la cabeza de despliegue se entrena con todo).

| Modelo | Acc. Wikimedia | F1 macro Wikimedia | Acc. fotos reales | Peso | Latencia media (CPU) | Memoria pico |
|---|---|---|---|---|---|---|
| ResNet50 (referencia) | 0.974 | 0.975 | 0.818 | 94.4 MB | 112.7 ms | 560 MB |
| MobileNetV2 | 0.897 | 0.898 | 0.795 | 9.0 MB | 51.1 ms | 517 MB |
| EfficientNetB0 (despliegue) | 0.974 | 0.975 | 0.841 | 16.2 MB | 128.1 ms | 378 MB |

Criterio: el más ligero en memoria con accuracy a menos de 0.02 de ResNet50. EfficientNetB0 empata en exactitud, pesa casi 6 veces menos y usa menos memoria pico, lo que cabe mejor en la Mac que sirve la API. No es más rápido que ResNet50. MobileNetV2 es el más veloz pero pierde 7.7 puntos en Wikimedia.

Aclaración sobre los datos: solo se guarda la cabeza (unos 90 KB), los pesos del backbone se descargan. El F1 macro en fotos reales es bajo porque hay clases con 1 a 3 ejemplos; la exactitud en esa prueba es indicativa, no concluyente.

### Pruebas

`pytest`: 218 pruebas pasan (reglas 69, agente 42, API 36, validación 22, interfaz 27, modelos reales 17, regresión Avance 2 5). Salida completa en `app/evidencias/pruebas_pytest.txt`. La prueba de humo contra la API en vivo pasó 9 de 9 (`app/evidencias/prueba_humo_local.txt`). Las pruebas del Avance 1 dan el mismo resultado antes y después (`pruebas_avance1_antes.txt` y `pruebas_avance1_despues.txt`).

Las pruebas con modelos reales llevan la marca `modelos`; para omitirlas: `pytest -m "not modelos"`.

### Regresión contra el Avance 2

Se reproducen los casos de integración del Avance 2 (`app/evidencias/regresion_avance2.csv`). Seis comportamientos cambian a propósito y están documentados en la prueba:

1. Fallas del aire acondicionado (ES y EN): ahora escala a la anfitriona; antes devolvía un consejo inventado.
2. Recomendar restaurantes (dos archivos): fuera de alcance, escala.
3. Late check-out en inglés: política de la anfitriona, escala.
4. Foto de un perro: la clase `otro` la rechaza; el modelo de 5 clases la aceptaba como caseta.

Como la cabeza de visión final se entrena con todas las imágenes, estas pruebas validan la política del agente, no la generalización del modelo. La generalización se mide con la partición Wikimedia y la validación dejando una toma fuera.

## Despliegue

Dos piezas. La UI está publicada en Streamlit Community Cloud (https://hospedaje-asistente-ia.streamlit.app, rama `proyecto-final`, archivo `proyecto_final/app/ui/streamlit_app.py`, dependencias `proyecto_final/app/ui/requirements.txt`) y lleva un aviso fijo de demostración académica. La API Flask tuvo una primera etapa en una computadora local con Cloudflare Tunnel (el 23-nov se publicó con un túnel rápido y su prueba de humo pública dio 9 de 9; no hay evidencia guardada de esa ejecución, y la URL de un túnel rápido cambia en cada reinicio). Para una URL estable, el PR #8 agrega un Dockerfile y scripts para desplegarla en un Space de Hugging Face (puerto 7860, secreto `API_TOKEN`; el Space gratuito se duerme por inactividad y se despierta antes de la demo).

1. API (local o en contenedor):
   ```bash
   cd proyecto_final/app && source .venv/bin/activate
   export API_TOKEN=<token largo aleatorio>
   gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 servidor:app
   ```
   Usar un solo worker: el limitador de tasa y las métricas viven en memoria.
2. Verificar: `python scripts/prueba_humo.py --api https://<url-de-la-api> --token <API_TOKEN>`.
3. Streamlit Community Cloud, Secrets:
   ```toml
   API_URL = "https://<url-de-la-api>"
   API_TOKEN = "<mismo token>"
   ```

Limitación: si la API está dormida o apagada, la UI muestra el aviso de servicio no disponible.

## Limitaciones conocidas

- El aviso a la anfitriona está marcado y registrado (log `aviso_anfitriona` y métrica `avisar_anfitriona:si`), pero no hay un canal de envío conectado. Las respuestas dicen que se avisará; mientras no exista el canal, es trabajo futuro y debe decirse así en la demo.
- El modelo de tono (TF-IDF, F1 0.61) es demasiado débil y no se despliega; el tono lo resuelve el léxico reactivo. Sus métricas están en `artefactos/metricas_texto.json`.
- El ensamble con MiniLM del Avance 2 no se despliega (peso y dependencias); sus resultados se citan solo como referencia.
- Pocas fotos reales: 44 recortes, clases con 1 a 3 ejemplos.
- Alcance: la base de conocimiento es de una sola propiedad.

## Ética y privacidad

- Ningún código de acceso, Wi-Fi, número de propiedad ni dato de huéspedes está en el repo, las respuestas o los logs.
- Las tarjetas y CVV se detectan y se rechazan sin procesarlas ni registrarlas.
- La API no almacena conversaciones; las fotos se procesan en memoria.
- Urgencias se remiten al 911 y no a un chatbot.
- Lo que no sabe, lo escala a una persona.

## Estado de los 8 productos de la rúbrica

| # | Producto | Estado | Evidencia o pendiente |
|---|---|---|---|
| 1 | PDF técnico de 15 a 20 páginas | Listo | `documento/Proyecto_Final_Jose_Pablo_Herrera_Fernandez.pdf` (texto fuente `documento/documento_tecnico.md`; se reconstruye con `documento/construir_pdf.py`). Quedan marcadores de datos de la nube y la nota de uso de IA |
| 2 | Repositorio en GitHub | Listo | Este repo, rama `proyecto-final` |
| 3 | Video demo de 5 a 10 min | En Drive | Grabación de pantalla con guion; el enlace lo entrega JP |
| 4 | URL pública | Parcial | UI en https://hospedaje-asistente-ia.streamlit.app; API en la nube en el PR #8 (Hugging Face Spaces) |
| 5 | Diagrama de arquitectura final | Listo | `app/docs/arquitectura_final.png`, `app/docs/flujo_coordinacion.md` |
| 6 | Evidencia de pruebas y despliegue | Parcial | Pruebas en `app/evidencias/`; falta la prueba de humo contra la API en la nube |
| 7 | Métricas de clasificación y chatbot | Listo | Secciones Resultados, `app/artefactos/*.json`, `app/evidencias/metricas_evaluacion_texto.json`, `matriz_confusion_vision.png` y `gradcam_efficientnetb0.png` |
| 8 | Plan de mejoras futuras | Listo | Sección 16 del PDF (tabla de 10 mejoras) |
