# Fotos reales de Cas'aguacate: medición del clasificador visual

Proyecto Final · Desarrollo de aplicaciones con IA (Tecmilenio). Mide el ResNet50 del Avance 2
(`resnet50_hospedaje.keras`, 5 clases: cerradura, porton, caseta, controles, interior) con las imágenes
reales que envió Cas'aguacate el 16-nov-2026.

## Qué llegó
Las 33 imágenes no son fotos sueltas: son páginas del folleto del hotel (1236 x 1600).
- 10 páginas con fotos de interiores (collages de recámaras, baños, sala, cocina, terrazas).
- 23 páginas sin una de las 5 clases: instrucciones de entrada, reglas, servicios, bienvenida, logotipo,
  alrededores y amenidades. Se etiquetan como `otro`.
- `recortar_fotos.py` separa las fotos de cada página. Quedan 44 recortes útiles (se descartan a mano
  4 que son letreros o logotipos y 1 que repite un recorte mayor): 25 interior, 15 otro, 3 porton, 1 cerradura.
  No hay ninguna foto de caseta ni de controles.

Las imágenes no se suben a este repositorio público (muestran accesos del hotel). Solo se guardan las
etiquetas (`etiquetas_paginas.csv`, `etiquetas_recortes.csv`), las predicciones y las métricas.

## Regla de decisión
La misma del Avance 2: si la confianza de la clase ganadora es menor que 0.60, la foto se trata como `otro`
(no se etiqueta y se escala a la anfitriona).

## Resultados (`salidas/resultados_fotos_reales.json`)

| Conjunto | Modelo | n | Accuracy | F1 macro |
|---|---|---|---|---|
| Páginas | ResNet50 Avance 2, sin reentrenar | 33 | 0.697 | 0.301 |
| Páginas | Capa de salida nueva (6 clases, partición por toma) | 33 | 0.818 | 0.516 |
| Recortes | ResNet50 Avance 2, sin reentrenar | 44 | 0.545 | 0.240 |
| Recortes | Capa de salida nueva (6 clases, partición por toma) | 44 | 0.750 | 0.315 |

F1 macro promedia las clases que aparecen en lo real o en lo predicho (con 0 cuando una clase no tiene aciertos),
por eso baja tanto: las clases de exterior tienen 1 a 3 ejemplos.

Por clase, en los 44 recortes:
- interior: 20 de 25 bien sin reentrenar; 25 de 25 con la capa nueva.
- otro: 3 de 15 sin reentrenar (el modelo de 5 clases fuerza una clase con mucha confianza: el árbol del jardín
  sale portón 0.997 y las botellas de shampoo salen cerradura 0.994); 8 de 15 con la capa nueva.
- porton: 0 de 3 en ambos. cerradura: 1 de 1 sin reentrenar (0.934), 0 de 1 con la capa nueva.

Matrices de confusión: `salidas/matrices_fotos_reales.png`. Predicción por imagen: `salidas/predicciones_fotos_reales.csv`.

## Ajuste ligero con partición por toma
No se reentrena la red. Se extraen las características de ResNet50 (capa de promedio global, 2048 valores) y se
entrena una capa de salida nueva de 6 clases (las 5 de antes más `otro`) con regresión logística
(`C=0.1`, pesos balanceados). Entrena con las 259 imágenes de Wikimedia del Avance 2, las 6 fotos fuera de dominio
(como `otro`) y las imágenes reales de las demás tomas. La evaluación deja fuera un grupo a la vez (`toma`: cada
habitación, áreas comunes, exterior, alrededores, etc.), así que fotos casi iguales nunca están en entrenamiento
y prueba al mismo tiempo (por ejemplo, el árbol aparece en dos páginas y ambas quedan en el mismo grupo).

## Lectura
- Interiores reales: el modelo funciona; con la capa nueva no falla ninguno de 25.
- Lo que no es de las 5 clases es el punto débil. Agregar `otro` ayuda (3 a 8 de 15), pero no basta.
- Exteriores (portón, cerradura): 4 fotos, todas de folleto. No alcanzan para medir. Hacen falta fotos tomadas
  como las tomaría un huésped: teclado del acceso peatonal, cerradura de la habitación, portón del parqueo,
  caseta del residencial, control del aire acondicionado.

## Cómo reproducir
```bash
python recortar_fotos.py <carpeta con las 33 páginas> <carpeta de recortes>
python evaluar_fotos_reales.py --model resnet50_hospedaje.keras --pages <carpeta con las 33 páginas> --crops <carpeta de recortes>
```
Entorno de la medición: Python 3.9.6, TensorFlow 2.16.2, Keras 3.10.0, scikit-learn 1.6.1, en CPU (Mac).
El modelo se cargó con `safe_mode=False` porque incluye una capa `Lambda` de preprocesamiento.
