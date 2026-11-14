# Avance 2: asistente multimodal (texto + imagen)

Continuación del chatbot basado en reglas del Avance 1. Curso MBIA2602 Desarrollo de aplicaciones con inteligencia artificial, Universidad Tecmilenio.

## Qué hay aquí

| Ruta | Para qué sirve |
| --- | --- |
| `avance2.ipynb` | Notebook ejecutado: NLP, clasificador de intenciones, clasificador visual e integración |
| `datos/corpus_intenciones.csv` | 681 frases sintéticas en 14 intenciones (separador `\|`) |
| `datos/fuera_de_alcance.csv` | 8 frases fuera de alcance para probar el umbral de confianza |
| `datos/imagenes/<clase>/` | 259 imágenes de Wikimedia Commons en 5 clases (256 únicas tras quitar casi duplicados), revisadas a mano |
| `datos/creditos_imagenes.csv` | Título, autor, licencia y URL de cada imagen |
| `salidas/` | Figuras, métricas (`resumen.json`) y salida de la integración |
| `scripts/` | Cómo se construyeron el corpus y el conjunto de imágenes |

## Resultados (conjunto de prueba)

- Intenciones, SVM lineal calibrada sobre TF-IDF de palabras y caracteres: accuracy 0.752, F1 macro 0.746.
- Imágenes, ResNet50 con transferencia y ajuste fino: accuracy 0.872, F1 macro 0.876. CNN desde cero: accuracy 0.538.

## Cómo correrlo en Colab

```python
!git clone -b avance2 https://github.com/jospabloh/chatbot-hospedaje-nltk
%cd chatbot-hospedaje-nltk/avance2
```

Después, Ejecutar todo. Con GPU tarda unos minutos; los números pueden variar un poco por el no determinismo de la GPU.

## Datos y privacidad

Los datos del hospedaje ("Casa Almendro") y el corpus son sintéticos. No hay mensajes ni datos de huéspedes reales. Las imágenes tienen licencia Creative Commons o son de dominio público; los créditos individuales están en `datos/creditos_imagenes.csv`.
