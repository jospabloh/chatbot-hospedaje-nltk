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
| `datos/fuera_de_dominio/` | 6 fotos fuera de dominio (perros, platillos, automóviles) con sus créditos, para probar qué hace el sistema con una foto que no es de ninguna clase |
| `salidas/` | Figuras, métricas (`resumen.json`) y salida de la integración |
| `scripts/` | Cómo se construyeron el corpus y el conjunto de imágenes |

## Resultados (conjunto de prueba)

- Intenciones, SVM lineal calibrada sobre TF-IDF de palabras y caracteres: accuracy 0.752, F1 macro 0.746.
- Imágenes, ResNet50 con transferencia y ajuste fino: accuracy 0.872, F1 macro 0.876. CNN desde cero: accuracy 0.538.
- Fotos fuera de dominio: 4 de 6 superan el umbral de imagen (0.60). El clasificador no sabe rechazar; queda como limitación y como acción para la versión final (`salidas/fuera_de_dominio.json`).

## Cómo correrlo en Colab

```python
!git clone -b avance2 https://github.com/jospabloh/chatbot-hospedaje-nltk
%cd chatbot-hospedaje-nltk/avance2
```

Después, Ejecutar todo. Con GPU tarda unos minutos; los números pueden variar un poco por el no determinismo de la GPU.

## Datos y privacidad

El caso es Cas'aguacate, un hospedaje vacacional real, usado con la autorización por escrito de sus anfitriones. No hay mensajes ni datos de huéspedes reales: el corpus de intenciones se construyó a partir de las consultas que los anfitriones reciben en el día a día. Las imágenes tienen licencia Creative Commons o son de dominio público; los créditos individuales están en `datos/creditos_imagenes.csv`.
