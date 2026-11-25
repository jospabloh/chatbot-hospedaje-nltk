# Documento técnico final: esqueleto

Proyecto Final, Desarrollo de aplicaciones con IA (Tecmilenio). Asistente de check-in de Cas'aguacate.
Entrega: PDF de 15 a 20 cuartillas, vie 4-dic 23:59. Este archivo es el esqueleto de trabajo: secciones en el orden que pide la actividad, extensión objetivo, de dónde sale el contenido y qué falta.

Total objetivo: 18 cuartillas de texto y figuras, sin portada ni referencias.

## 1. Introducción (1 cuartilla)
- Contexto: hospedaje pequeño en La Asunción de Belén, Heredia, Costa Rica; la anfitriona responde a mano el check-in por WhatsApp.
- Propósito del sistema y alcance del certificado (Avance 1, Avance 2, Proyecto Final).
- Fuente: Avance 1 y Avance 2 (Problema, Objetivo, Usuarios, Alcance), corregidos con los datos reales.
- Falta: redactar.

## 2. Descripción del problema (1.5 cuartillas)
- Preguntas repetidas de huéspedes (horarios, acceso, pagos, Wi-Fi), llegadas tarde y fuera de horario, idioma (ES y EN).
- Usuarios: huésped, anfitriona y dueño.
- Objetivo medible: resolver solo lo que se puede contestar con datos del negocio y escalar el resto.
- Campo de aplicación (Tema 12): turismo y hospitalidad; un párrafo que lo compare con retail (recomendación) y salud (monitoreo).
- Falta: redactar; usar preguntas frecuentes reales si llegan (si no, se declara como limitación).

## 3. Arquitectura general (2 cuartillas)
- Diagrama final: `app/docs/arquitectura_final.png` (se regenera con `scripts/figura_arquitectura_final.py`).
- Capas: interfaz Streamlit, API REST Flask, agente híbrido, servicios de NLP y visión, base de conocimiento.
- Flujo de una petición `/agent` con texto y foto.
- Fuente: README del proyecto y `app/docs/flujo_coordinacion.md`.
- Falta: revisar que el diagrama use los nombres finales (EfficientNetB0 en despliegue, ResNet50 de referencia).

## 4. Módulo de NLP (2 cuartillas)
- Preprocesamiento (normalización, stopwords NLTK, stemming), TF-IDF de palabras y caracteres, LinearSVC calibrado, un modelo por idioma.
- Umbral 0.40 y cobertura; frases fuera de alcance.
- Métricas (Tema 10): accuracy, F1 macro, F1 macro CV, precisión, recall y F1 por intención, matriz de confusión, AUC uno contra el resto y precisión promedio (`app/evidencias/metricas_evaluacion_texto.json`, `matriz_confusion_es.png`, `matriz_confusion_en.png`, generados con `scripts/metricas_evaluacion.py`).
- Resultados: ES accuracy 0.752, F1 macro 0.746, AUC OvR macro 0.974, AP macro 0.820; EN accuracy 0.729, F1 macro 0.720, AUC OvR macro 0.966, AP macro 0.826.
- Análisis de errores: confusiones principales (ES: ubicación con llegada tarde, problemas con servicios; EN: horarios con disponibilidad, tarifas con servicios).
- Falta: redactar el análisis y una tabla corta por intención.

## 5. Módulo de visión por computadora (2 cuartillas)
- Clases: teclado peatonal, cerradura, portón, caseta, control, otro.
- Backbone congelado y cabeza de regresión logística con `class_weight="balanced"` (Tema 10, clases desbalanceadas).
- Comparación ResNet50 (referencia) contra EfficientNetB0 (despliegue) y MobileNetV2: exactitud, F1 macro, peso, latencia y memoria.
- Fotos reales: 44 recortes, validación dejando una toma fuera; exactitud 0.841 con EfficientNetB0.
- Falta (Tema 10): matriz de confusión de visión en figura, interpretabilidad (mapa de activación tipo Grad-CAM de 2 o 3 fotos) y nota de hiperparámetros (C=0.1, entrada 160x160, por qué no hubo ajuste fino completo).

## 6. Arquitectura del agente inteligente (2 cuartillas)
- Tema 11: agente que percibe, decide y actúa; entorno, sensores (texto, foto, estado) y actuadores (respuesta, aviso a la anfitriona).
- Tipo híbrido: capa reactiva con reglas de prioridad fija (urgencia, datos sensibles, pedir persona, política de la anfitriona, fuera de alcance) y capa deliberativa con estado de conversación y política.
- Acciones: RESPONDER, RESPONDER_Y_AVISAR, PEDIR_FOTO, ACLARAR, ESCALAR, ESCALAR_URGENTE, PROTEGER_DATOS.
- Por qué no aprendizaje por refuerzo ni multiagente: pocos datos y riesgo para el huésped; quedan como mejora futura.
- Métricas del agente: distribución de acciones, tasa de escalamiento, aclaraciones antes de escalar.
- Falta: tabla de métricas del agente (de `/metrics` y de `tests/test_agente.py`).

## 7. Desarrollo web (1.5 cuartillas)
- Tema 13: Streamlit con `st.session_state` (historial, estado del agente, idioma, id de sesión), `st.file_uploader` con tipos permitidos, errores amables, chat bilingüe y traza del agente visible.
- Diseño de interfaz amigable: claridad, retroalimentación y control del usuario.
- Falta: capturas (chat, foto, aviso de servicio no disponible) y redactar.

## 8. API REST (1.5 cuartillas)
- Tema 13: Flask con `/health`, `/chat`, `/image`, `/agent` y `/metrics`; validación de entradas (extensión, tamaño, píxeles), códigos de error estables (400, 401, 413, 415, 422, 429, 503, 500), `X-Request-Id`, token y límite de peticiones.
- Falta: redactar y pegar dos ejemplos reales de petición y respuesta.

## 9. Despliegue en la nube (1.5 cuartillas)
- Tema 14: API en gunicorn (1 worker, 4 hilos) expuesta con Cloudflare Tunnel; interfaz en Streamlit Community Cloud con Secrets `API_URL` y `API_TOKEN`.
- Evidencias: URL pública, prueba de humo por URL pública (9/9 el 23-nov), capturas de Streamlit Cloud.
- Falta: interfaz en Streamlit Community Cloud, túnel con nombre para URL estable y capturas.

## 10. Resultados (2 cuartillas)
- Pruebas: 204 con pytest (reglas, agente, API, validación, interfaz, modelos reales, regresión contra el Avance 2) y prueba de humo 9/9.
- Métricas de clasificación (texto y visión) y del chatbot (cobertura con umbral, fuera de alcance bajo umbral).
- Análisis que pide la rúbrica (Tema 14): desempeño (latencia p50 y p95 de `/metrics`), escalabilidad (un worker por el limitador en memoria; qué cambiar para más carga) y mantenibilidad (módulos, pruebas, configuración por variables).
- Falta: medir latencia p50 y p95 en el despliegue público y redactar los tres análisis con esos números.

## 11. Consideraciones éticas (1 cuartilla)
- Tema 12: privacidad (sin códigos, Wi-Fi ni datos de huéspedes en repo, respuestas o logs; fotos en memoria), datos sensibles (tarjeta y CVV rechazados), seguridad (urgencias al 911), transparencia (el huésped sabe que habla con un asistente), sesgos (idioma, pocas fotos reales) y supervisión humana (escalamiento).
- Falta: redactar.

## 12. Conclusiones (0.5 cuartilla)
- Qué se logró contra el objetivo, aprendizajes y plan de mejoras futuras.

## 13. Referencias (fuera del conteo)
- Documentación de Streamlit, Flask, scikit-learn, Keras y Cloudflare Tunnel; referencias de los Temas 10 a 14 que se citen; formato APA.

## Plan de mejoras futuras
1. Canal real de aviso a la anfitriona (WhatsApp) para las acciones de escalamiento.
2. Más fotos reales tomadas como huésped (teclado, cerradura, portón, caseta, controles) y ajuste fino del backbone.
3. Preguntas frecuentes reales para ampliar el corpus y medir con conversaciones del piloto.
4. Túnel con nombre o API en un servicio en la nube para no depender de una computadora local.
5. Monitoreo con exportador de métricas y alertas; persistencia de métricas.
6. Interpretabilidad visual en la interfaz (qué zona de la foto decidió la clase).

## Entregables y evidencias (criterio de documentación)

| # | Producto | Estado | Dónde o qué falta |
|---|---|---|---|
| 1 | Documento técnico PDF de 15 a 20 cuartillas | Esqueleto | Este archivo; redactar del 28 al 30-nov |
| 2 | Repositorio GitHub con enlace | Listo | jospabloh/chatbot-hospedaje-nltk, rama proyecto-final |
| 3 | Video de 5 a 10 minutos | Pendiente | Guion 1 y 2-dic |
| 4 | URL pública | Parcial | API publicada el 23-nov por túnel rápido; falta interfaz en Streamlit Cloud y URL estable |
| 5 | Diagrama de arquitectura final | Listo | `app/docs/arquitectura_final.png` |
| 6 | Evidencias de pruebas y despliegue | Parcial | Pruebas en `app/evidencias/`; faltan capturas de despliegue |
| 7 | Métricas de clasificación y del chatbot | Casi listo | `artefactos/*.json` y `evidencias/metricas_evaluacion_texto.json`; falta matriz de visión en figura y métricas del agente |
| 8 | Plan de mejoras futuras | Esqueleto | Sección de arriba |

## Qué debe mostrar el video (producto esperado del Tema 14)
- API Flask respondiendo (`/health`).
- Interfaz publicada con su URL.
- Una foto cargada y su clase con confianza.
- Tres preguntas al chatbot: una que responde, una que escala a la anfitriona y una urgencia.
- La traza del agente y el panel de métricas.
- Cierre con limitaciones, privacidad y mejoras.
