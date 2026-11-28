"""Evaluation metrics for the intent classifiers, following Tema 10 (confusion matrix, ROC / AUC one-vs-rest,
precision-recall) without retraining anything.

It rebuilds the same 80/20 stratified split (seed 42) used by scripts/entrenar_texto.py, loads the exported
pipelines (artefactos/intenciones_<lang>.joblib, fitted on the training part only) and scores the held-out part.

    python scripts/metricas_evaluacion.py

Writes in evidencias/: metricas_evaluacion_texto.json and matriz_confusion_<lang>.png (if matplotlib is available).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

SEED = 42


def main() -> None:
    import joblib
    from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize

    from casaguacate import config, texto

    out = APP / "evidencias"
    out.mkdir(exist_ok=True)
    result = {"semilla": SEED, "particion": "80/20 estratificada, misma que entrenar_texto.py"}

    for lang, name in [("es", "corpus_intenciones.csv"), ("en", "corpus_intenciones_en.csv")]:
        corpus = pd.read_csv(config.DATA_DIR / name, sep="|")
        x = pd.DataFrame({"norm": [texto.preprocess(t, lang, stem=False, remove_stopwords=False) for t in corpus["texto"]],
                          "proc": [texto.preprocess(t, lang) for t in corpus["texto"]]})
        _, x_te, _, y_te = train_test_split(x, corpus["intencion"], test_size=0.2, random_state=SEED,
                                            stratify=corpus["intencion"])
        model = joblib.load(config.ARTIFACTS_DIR / f"intenciones_{lang}.joblib")
        classes = list(model.classes_)
        proba = model.predict_proba(x_te)
        pred = np.asarray(classes)[proba.argmax(axis=1)]
        y_bin = label_binarize(y_te, classes=classes)
        cm = confusion_matrix(y_te, pred, labels=classes)
        p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=classes, zero_division=0)
        auc_per_class = {c: round(float(roc_auc_score(y_bin[:, i], proba[:, i])), 4)
                         for i, c in enumerate(classes) if 0 < y_bin[:, i].sum() < len(y_bin)}
        pairs = [(classes[i], classes[j], int(cm[i, j])) for i in range(len(classes)) for j in range(len(classes))
                 if i != j and cm[i, j] > 0]
        pairs.sort(key=lambda t: -t[2])
        result[lang] = {
            "n_prueba": int(len(y_te)),
            "clases": classes,
            "auc_ovr_macro": round(float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")), 4),
            "auc_ovr_ponderado": round(float(roc_auc_score(y_bin, proba, average="weighted", multi_class="ovr")), 4),
            "precision_promedio_macro": round(float(average_precision_score(y_bin, proba, average="macro")), 4),
            "auc_por_intencion": auc_per_class,
            "por_intencion": {c: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                                  "f1": round(float(f[i]), 3), "soporte": int(s[i])} for i, c in enumerate(classes)},
            "confusiones_principales": [{"real": a, "predicha": b, "casos": n} for a, b, n in pairs[:8]],
            "matriz_confusion": cm.tolist(),
        }
        print(lang, "AUC OvR macro", result[lang]["auc_ovr_macro"], "AP macro", result[lang]["precision_promedio_macro"])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.imshow(cm, cmap="Greens")
            ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
            ax.set_yticks(range(len(classes)), classes, fontsize=7)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    if cm[i, j]:
                        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
            ax.set_xlabel("Predicha")
            ax.set_ylabel("Real")
            ax.set_title(f"Matriz de confusión, intenciones ({lang.upper()})")
            fig.tight_layout()
            fig.savefig(out / f"matriz_confusion_{lang}.png", dpi=150)
            plt.close(fig)
        except ImportError:
            print("matplotlib no disponible: se omite la figura")

    (out / "metricas_evaluacion_texto.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                        encoding="utf-8")


if __name__ == "__main__":
    main()
