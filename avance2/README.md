# Avance 2: asistente multimodal (texto + imagen)

Continuación del chatbot basado en reglas del Avance 1. Curso MBIA2602 Desarrollo de aplicaciones con inteligencia artificial, Universidad Tecmilenio.

## Qué hay aquí

| Ruta | Para qué sirve |
| --- | --- |
| `avance2.ipynb` | Notebook ejecutado completo en Google Colab: NLP, clasificador de intenciones, clasificador visual, integración y comparación con otras técnicas del curso |
| `datos/corpus_intenciones.csv` | 681 frases sintéticas en 14 intenciones (separador `\|`) |
| `datos/corpus_intenciones_en.csv` | 350 frases en inglés para las mismas 14 intenciones (versión en inglés del asistente) |
| `datos/corpus_tono.csv` | 180 frases etiquetadas por tono (positivo, neutral, negativo) para el análisis de sentimientos |
| `datos/fuera_de_alcance.csv` | 8 frases fuera de alcance para probar el umbral de confianza |
| `datos/imagenes/<clase>/` | 259 imágenes de Wikimedia Commons en 5 clases (256 únicas tras quitar casi duplicados), revisadas a mano |
| `datos/creditos_imagenes.csv` | Título, autor, licencia y URL de cada imagen |
| `datos/fuera_de_dominio/` | 6 fotos fuera de dominio (perros, platillos, automóviles) con sus créditos, para probar qué hace el sistema con una foto que no es de ninguna clase |
| `datos/fotos_reales/` | 9 fotos de la galería pública de Cas'aguacate, usadas con autorización de los anfitriones, para probar el modelo |
| `salidas/` | Figuras y resultados de la ejecución: `resumen.json`, `comparaciones.json`, `entorno.json` y las salidas de la integración |
| `scripts/` | Cómo se construyeron el corpus y el conjunto de imágenes |

## Resultados (conjunto de prueba, ejecución en Google Colab)

- Intenciones, SVM lineal calibrada sobre TF-IDF de palabras y caracteres: accuracy 0.752, F1 macro 0.746, AUC macro 0.974.
- Imágenes, ResNet50 con transferencia y ajuste fino: accuracy 0.949, F1 macro 0.947, AUC macro 0.997. CNN desde cero: accuracy 0.538.
- Fotos fuera de dominio: 5 de 6 superan el umbral de imagen (0.60). El clasificador no sabe rechazar; queda como limitación y como acción para la versión final (`salidas/fuera_de_dominio.json`).
- Fotos reales de Cas'aguacate: el modelo reconoce 8 de 8 interiores, 8 por encima del umbral. Solo hay fotos reales de una de las cinco clases.
- Versión en inglés: accuracy 0.729, F1 macro 0.720 sobre 70 frases de prueba. El detector de idioma acierta 100.0 % en español y 96.6 % en inglés.

## Otras técnicas del curso sobre el mismo caso (`salidas/comparaciones.json`)

- Bag of Words y TF-IDF de palabras, cada uno con Naive Bayes y SVM: F1 macro en prueba entre 0.567 y 0.645.
- Búsqueda de hiperparámetros de la SVM (24 combinaciones, validación cruzada): la mejor da F1 macro 0.717 contra 0.706 de la configuración elegida, dentro de la variación entre particiones.
- Sobremuestreo y pesos por clase: sin cambio (0.705 y 0.705 contra 0.706). No se adoptan.
- Redes neuronales con word embeddings aprendidos: Embeddings + promedio, F1 macro 0.616; Embeddings + LSTM bidireccional, F1 macro 0.620.
- Vectores de palabras preentrenados (spaCy) + SVM: F1 macro 0.725 en español. No se adoptan.
- Transformer multilingüe (MiniLM) + SVM, un solo modelo para los dos idiomas: F1 macro 0.773 en español y 0.913 en inglés. Es la base de la versión mejorada del asistente (`assist_plus`), comparada con la documentada en `salidas/integracion_mejorada.csv`.
- Tono del mensaje (análisis de sentimientos, Transformer multilingüe (MiniLM) + SVM lineal): accuracy 0.917, F1 macro 0.917 en 36 frases.
- Imágenes con características clásicas: HOG + SVM, accuracy 0.487; SIFT + SVM, accuracy 0.462.
- Redes preentrenadas congeladas: VGG16, accuracy 0.923; ResNet50, accuracy 0.923; EfficientNetB0, accuracy 0.923.
- Padding contra foto estirada (ResNet50 congelada, 5 semillas): accuracy en prueba 0.908 contra 0.949. No se adopta.
- Robustez a la iluminación del modelo integrado (`salidas/iluminacion.png`): accuracy entre 0.923 y 0.949 con fotos oscuras, sobreexpuestas y de bajo contraste. Por eso no se agrega color jitter.
- Curvas ROC (`salidas/curvas_roc.png`) y Grad-CAM (`salidas/grad_cam.png`).

## Idioma y código

El asistente funciona en español o en inglés según el huésped: detecta el idioma del mensaje y contesta en ese idioma, igual que el chatbot del Avance 1. El código está en inglés (funciones, variables y comentarios); las etiquetas de intenciones y clases son datos y se conservan en español.

## Cómo correrlo en Colab

Abre `avance2.ipynb` en Google Colab y elige Ejecutar todo. La primera celda clona este repositorio (rama `avance2`) y se ubica en la carpeta `avance2`. Con GPU tarda unos minutos; los números del componente visual pueden variar un poco entre ejecuciones porque las redes se entrenan con inicialización aleatoria.

## Datos y privacidad

El caso es Cas'aguacate, un hospedaje vacacional real, usado con la autorización por escrito de sus anfitriones. No hay mensajes ni datos de huéspedes reales: los corpus se construyeron a partir de las consultas que los anfitriones reciben en el día a día. Las imágenes tienen licencia Creative Commons o son de dominio público; los créditos individuales están en `datos/creditos_imagenes.csv`.
