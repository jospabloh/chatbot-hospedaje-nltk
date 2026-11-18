"""Measure the Avance 2 ResNet50 on Cas'aguacate's real images and compare it with a retrained output layer.

Usage:
    python evaluar_fotos_reales.py --model resnet50_hospedaje.keras --pages <dir with the 33 pages> \
        --crops <dir with the photo crops> --wiki ../../avance2/datos

The 33 images sent by the business are brochure pages (photo collages and instruction cards).
`recortar_fotos.py` extracts the individual photos from them. The images themselves are not stored
in this public repository; only labels, predictions and metrics are.

Two evaluations, both with the decision rule of Avance 2 (below 0.60 confidence -> "otro"):
  1. base: the Avance 2 model as is (5 classes, no retraining).
  2. head: frozen ResNet50 features + a new 6-class output layer (adds "otro"), trained on the
     Wikimedia images, the out-of-domain photos and the real images of the OTHER shots/spaces
     (leave-one-group-out by `toma`), so near-identical photos never sit in train and test at once.
"""
import argparse, glob, json, os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

CLASSES = ["cerradura", "porton", "caseta", "controles", "interior"]
ALL = CLASSES + ["otro"]
IMG_SIZE = 160
IMAGE_THRESHOLD = 0.60
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)


def decide(prob, names):
    k = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    return [names[i] if c >= IMAGE_THRESHOLD else "otro" for i, c in zip(k, conf)], conf


def metrics(y_true, y_pred):
    labels = [c for c in ALL if c in set(y_true) | set(y_pred)]
    return {"n": len(y_true), "accuracy": round(accuracy_score(y_true, y_pred), 3),
            "f1_macro": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 3),
            "clases_en_f1": labels,
            "matriz": confusion_matrix(y_true, y_pred, labels=ALL).tolist()}


def plot_cm(ax, m, title):
    cm = np.array(m["matriz"])
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(ALL))); ax.set_xticklabels(ALL, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ALL))); ax.set_yticklabels(ALL, fontsize=8)
    for i in range(len(ALL)):
        for j in range(len(ALL)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicha"); ax.set_ylabel("real")
    ax.set_title(f"{title}\nacc {m['accuracy']:.3f} · F1 macro {m['f1_macro']:.3f} · n={m['n']}", fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pages", required=True)
    ap.add_argument("--crops", required=True)
    ap.add_argument("--wiki", default=os.path.join(HERE, "..", "..", "avance2", "datos"))
    ap.add_argument("--out", default=os.path.join(HERE, "salidas"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = keras.models.load_model(a.model, safe_mode=False, custom_objects={"preprocess_input": preprocess_input})
    features = keras.Model(model.inputs, model.get_layer("global_average_pooling2d_1").output)

    pages = pd.read_csv(os.path.join(HERE, "etiquetas_paginas.csv")); pages["tipo"] = "pagina"
    crops = pd.read_csv(os.path.join(HERE, "etiquetas_recortes.csv")); crops["tipo"] = "recorte"
    pages["ruta"] = [os.path.join(a.pages, f) for f in pages.archivo]
    crops["ruta"] = [os.path.join(a.crops, f) for f in crops.archivo]
    real = pd.concat([pages, crops], ignore_index=True)
    x_real = np.stack([load(p) for p in real.ruta])

    # 1) Avance 2 model as is
    prob = model.predict(x_real, verbose=0)
    real["pred_base"], real["conf_base"] = decide(prob, CLASSES)
    real["top_base"] = [CLASSES[i] for i in prob.argmax(axis=1)]

    # 2) New 6-class output layer on frozen features, leave-one-group-out by shot/space
    wiki = [(p, c) for c in CLASSES for p in sorted(glob.glob(os.path.join(a.wiki, "imagenes", c, "*.jpg")))]
    wiki += [(p, "otro") for p in sorted(glob.glob(os.path.join(a.wiki, "fuera_de_dominio", "*.jpg")))]
    f_wiki = features.predict(np.stack([load(p) for p, _ in wiki]), verbose=0)
    y_wiki = np.array([c for _, c in wiki])
    f_real = features.predict(x_real, verbose=0)
    y_real = real.clase.values

    pred_head, conf_head = [None] * len(real), np.zeros(len(real))
    for g in sorted(real.toma.unique()):
        test = (real.toma == g).values
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced",
                                                                 random_state=SEED))
        clf.fit(np.vstack([f_wiki, f_real[~test]]), np.concatenate([y_wiki, y_real[~test]]))
        p = clf.predict_proba(f_real[test])
        names = list(clf.classes_)
        k = p.argmax(axis=1)
        for idx, kk, pp in zip(np.where(test)[0], k, p):
            c = names[kk]
            pred_head[idx] = c if pp.max() >= IMAGE_THRESHOLD else "otro"
            conf_head[idx] = pp.max()
    real["pred_cabeza"], real["conf_cabeza"] = pred_head, conf_head.round(3)
    real["conf_base"] = real.conf_base.round(3)

    results = {"umbral_imagen": IMAGE_THRESHOLD, "imagenes_wikimedia": int(len(y_wiki)),
               "reparto_real": {t: real[real.tipo == t].clase.value_counts().to_dict() for t in ["pagina", "recorte"]},
               "tomas": sorted(real.toma.unique().tolist())}
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for r, t in enumerate(["pagina", "recorte"]):
        d = real[real.tipo == t]
        for c, (col, title) in enumerate([("pred_base", "ResNet50 Avance 2 (sin reentrenar)"),
                                          ("pred_cabeza", "Capa de salida nueva, 6 clases (partición por toma)")]):
            m = metrics(d.clase.tolist(), d[col].tolist())
            results[f"{t}_{'base' if c == 0 else 'cabeza'}"] = m
            plot_cm(axes[r, c], m, f"{'Páginas' if t == 'pagina' else 'Fotos recortadas'} · {title}")
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "matrices_fotos_reales.png"), dpi=130)
    real.drop(columns=["ruta"]).to_csv(os.path.join(a.out, "predicciones_fotos_reales.csv"), index=False)
    with open(os.path.join(a.out, "resultados_fotos_reales.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    for k, v in results.items():
        if isinstance(v, dict) and "accuracy" in v:
            print(f"{k:16s} n={v['n']:3d} acc={v['accuracy']:.3f} f1_macro={v['f1_macro']:.3f}")
    print(json.dumps(results["reparto_real"], ensure_ascii=False))


if __name__ == "__main__":
    main()
