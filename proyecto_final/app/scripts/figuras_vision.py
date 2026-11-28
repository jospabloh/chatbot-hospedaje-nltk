"""Figuras de evaluacion del clasificador visual (Tema 10): matrices de confusion y mapas Grad-CAM.

Que es: genera dos figuras para el documento tecnico del Proyecto Final, usando el modelo de despliegue
(EfficientNetB0 congelado a 160x160 + cabeza StandardScaler + regresion logistica, 6 clases).
Para que sirve:
  1. evidencias/matriz_confusion_vision.png: matriz de confusion en dos paneles.
       - Wikimedia: particion de prueba 70/15/15 (semilla 42), cinco clases, cabeza entrenada con train + validacion
         (mismo procedimiento que scripts/entrenar_vision.py, por eso la exactitud coincide con vision_meta.json).
       - Recortes reales del folleto de Cas'aguacate: matriz guardada en vision_meta.json (validacion dejando una
         toma fuera, regla de confianza < 0.60 -> "otro"). No se recalcula, solo se dibuja.
  2. evidencias/gradcam_efficientnetb0.png: Grad-CAM de 3 imagenes (2 recortes reales y 1 de Wikimedia) con la
     cabeza de despliegue. Muestra que zona de la foto empuja la decision hacia la clase predicha.

Uso (desde proyecto_final/app, con el entorno que tiene TensorFlow):
    python scripts/figuras_vision.py --crops <carpeta_con_recortes_reales>

Los recortes reales no estan en el repositorio (muestran accesos de la propiedad); sin --crops solo se usa Wikimedia
y se dibuja la matriz de fotos reales desde vision_meta.json. Las imagenes de Grad-CAM de fotos reales que se
publican son de interiores y del jardin, no de los accesos.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
SEED = 42
BACKBONE = "efficientnetb0"


def ahash(path, n=8):
    """Hash perceptual de 8x8 para descartar casi duplicados (igual que entrenar_vision.py)."""
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L").resize((n, n)), dtype=np.float32)
    return tuple((a > a.mean()).flatten())


def wikimedia_split():
    """Reconstruye la particion 70/15/15 de Wikimedia con la semilla 42; devuelve rutas y etiquetas por particion."""
    from sklearn.model_selection import train_test_split
    from casaguacate import config, vision

    paths, labels, seen = [], [], set()
    for k, c in enumerate(vision.CLASSES):
        for p in sorted(glob.glob(str(config.DATA_DIR / "imagenes" / c / "*.jpg"))):
            h = ahash(p)
            if h not in seen:
                seen.add(h)
                paths.append(p)
                labels.append(k)
    paths, labels = np.array(paths), np.array(labels)
    r_tr, r_tmp, e_tr, e_tmp = train_test_split(paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
    r_va, r_te, e_va, e_te = train_test_split(r_tmp, e_tmp, test_size=0.50, random_state=SEED, stratify=e_tmp)
    names = lambda e: np.array([vision.CLASSES[k] for k in e])  # noqa: E731
    return (r_tr, names(e_tr)), (r_va, names(e_va)), (r_te, names(e_te))


def plot_matrix(ax, cm, classes, title):
    """Dibuja una matriz de confusion con conteos dentro de cada celda."""
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(title, fontsize=10)
    top = cm.max() if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > top / 2 else "black", fontsize=9)


def confusion_figure(out: Path) -> dict:
    """Calcula la matriz de Wikimedia (prueba) y dibuja junto a la de fotos reales guardada en vision_meta.json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from casaguacate import vision

    (p_tr, y_tr), (p_va, y_va), (p_te, y_te) = wikimedia_split()
    extractor, _ = vision.build_extractor(BACKBONE)
    feat = lambda ps: np.asarray(extractor.predict(np.stack([vision.to_array(Image.open(p)) for p in ps]),  # noqa: E731
                                                   batch_size=32, verbose=0))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced",
                                                             random_state=SEED))
    clf.fit(feat(list(p_tr) + list(p_va)), np.concatenate([y_tr, y_va]))
    pred = clf.predict(feat(p_te))
    acc = float(accuracy_score(y_te, pred))
    cm_wiki = confusion_matrix(y_te, pred, labels=vision.CLASSES)

    meta = json.loads((APP / "artefactos" / "vision_meta.json").read_text(encoding="utf-8"))
    real = meta["comparacion"][BACKBONE]["fotos_reales"]
    cm_real = np.array(real["matriz"]["valores"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    plot_matrix(axes[0], cm_wiki, vision.CLASSES, f"Wikimedia, prueba (n={len(y_te)}, exactitud {acc:.3f})")
    plot_matrix(axes[1], cm_real, real["matriz"]["clases"],
                f"Recortes reales (n={real['n']}, exactitud {real['accuracy']:.3f})")
    fig.suptitle("Clasificador visual EfficientNetB0: matrices de confusion", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "matriz_confusion_vision.png", dpi=170)
    plt.close(fig)
    return {"accuracy_wikimedia": round(acc, 4), "n_wikimedia": int(len(y_te)),
            "matriz_wikimedia": cm_wiki.tolist(), "clases_wikimedia": vision.CLASSES}


def grad_cam(base_nopool, head, img_arr, prepare):
    """Grad-CAM de la clase ganadora: gradiente del puntaje (logit) de la cabeza respecto al ultimo mapa convolucional.

    La cabeza es StandardScaler + regresion logistica sobre el promedio global del mapa, asi que el logit se
    reconstruye en TensorFlow con sus coeficientes y se deriva con GradientTape (Selvaraju et al., 2017).
    Devuelve (mapa 0-1 en la resolucion del mapa convolucional, clase predicha, confianza).
    """
    import tensorflow as tf
    x = tf.convert_to_tensor(img_arr[None].astype("float32"))
    scaler, lr = head.named_steps["standardscaler"], head.named_steps["logisticregression"]
    mean, scale = tf.constant(scaler.mean_, tf.float32), tf.constant(scaler.scale_, tf.float32)
    coef, bias = tf.constant(lr.coef_, tf.float32), tf.constant(lr.intercept_, tf.float32)
    with tf.GradientTape() as tape:
        fmap = base_nopool(prepare(x), training=False)               # (1, h, w, canales)
        tape.watch(fmap)
        pooled = tf.reduce_mean(fmap, axis=(1, 2))                   # igual que pooling="avg"
        logits = ((pooled - mean) / scale) @ tf.transpose(coef) + bias
        k = int(tf.argmax(logits[0]))
        score = logits[0, k]
    grads = tape.gradient(score, fmap)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * fmap, axis=-1))[0].numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    prob = head.predict_proba(pooled.numpy())[0]
    return cam, str(head.classes_[k]), float(prob[k])


def gradcam_figure(out: Path, crops: str) -> list:
    """Dibuja Grad-CAM de 2 recortes reales (si hay --crops) y 1 imagen de Wikimedia de la particion de prueba."""
    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from tensorflow.keras import applications as apps
    from casaguacate import config, vision

    head = joblib.load(config.ARTIFACTS_DIR / f"vision_head_{BACKBONE}.joblib")
    base = apps.EfficientNetB0(weights="imagenet", include_top=False, pooling=None,
                               input_shape=(vision.IMG_SIZE, vision.IMG_SIZE, 3))
    prepare = apps.efficientnet.preprocess_input

    items = []   # (titulo, ruta)
    if crops:
        for archivo, que in (("WA0017_r1.jpg", "Recorte real: recamara y sala"),
                             ("WA0019_r1.jpg", "Recorte real: arbol de aguacate")):
            ruta = os.path.join(os.path.expanduser(crops), archivo)
            if os.path.exists(ruta):
                items.append((que, ruta))
    _, _, (p_te, y_te) = wikimedia_split()
    for p, y in zip(p_te, y_te):
        if y == "porton":
            items.append(("Wikimedia (prueba): porton", str(p)))
            break

    fig, axes = plt.subplots(len(items), 2, figsize=(7.2, 3.3 * len(items)))
    axes = np.atleast_2d(axes)
    resumen = []
    for row, (titulo, ruta) in zip(axes, items):
        img = Image.open(ruta).convert("RGB")
        arr = vision.to_array(img)
        cam, clase, conf = grad_cam(base, head, arr, prepare)
        heat = np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize((vision.IMG_SIZE,) * 2, Image.BILINEAR)) / 255
        row[0].imshow(arr.astype("uint8"))
        row[0].set_title(titulo, fontsize=9)
        row[1].imshow(arr.astype("uint8"))
        row[1].imshow(heat, cmap="jet", alpha=0.45)
        row[1].set_title(f"Grad-CAM: {clase} ({conf:.3f})", fontsize=9)
        for a in row:
            a.axis("off")
        resumen.append({"imagen": titulo, "clase": clase, "confianza": round(conf, 3),
                        "mapa": f"{cam.shape[0]}x{cam.shape[1]}"})
    fig.tight_layout()
    fig.savefig(out / "gradcam_efficientnetb0.png", dpi=170)
    plt.close(fig)
    return resumen


def main() -> None:
    """Punto de entrada: genera la matriz de confusion y el Grad-CAM, y guarda un resumen en JSON."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="", help="carpeta con los recortes reales (no estan en el repositorio)")
    args = ap.parse_args()
    out = APP / "evidencias"
    out.mkdir(exist_ok=True)
    resumen = {"modelo": BACKBONE, "confusion": confusion_figure(out), "gradcam": gradcam_figure(out, args.crops)}
    (out / "figuras_vision_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
