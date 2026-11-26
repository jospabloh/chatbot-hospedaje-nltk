"""Compare ResNet50 (reference) with MobileNetV2 and EfficientNetB0 (deployment candidates) and export the heads.

Every backbone is used frozen (ImageNet weights, 160x160 input) and gets the same head: StandardScaler +
logistic regression with six classes (cerradura, porton, caseta, controles, interior, otro). Measured:
  A. Wikimedia test split (70/15/15 stratified, seed 42, same as the Avance 2 notebook), five classes.
  B. Real Cas'aguacate photos (crops of the hotel brochure), six classes, leave-one-group-out by `toma`
     (a shot never sits in training and test at once), decision rule: confidence < 0.60 -> "otro".
  C. Cost: parameters, weights size, single-image latency on CPU and process memory (each in a clean subprocess).
The head of every backbone is trained with ALL the data and saved as artefactos/vision_head_<name>.joblib;
vision_meta.json records which one is deployed and why.

    python scripts/entrenar_vision.py --crops ~/haru-work/pf-fotos/recortes
    python scripts/entrenar_vision.py --medir mobilenetv2      (internal: cost of one backbone)

The real photos are not in the repository (they show the access points of the property): pass the folder with
--crops, labelled by proyecto_final/fotos_reales/etiquetas_recortes.csv. Without it, only the public data is used.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
SEED = 42
MAX_ACC_GAP = 0.02   # a lighter backbone is accepted if it loses at most this much accuracy vs ResNet50


def measure_cost(name: str) -> dict:
    """Runs in its own process so the memory figure belongs to that backbone only."""
    import joblib
    from PIL import Image
    from casaguacate import config, vision

    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6   # macOS reports bytes
    extractor, base = vision.build_extractor(name)
    head = joblib.load(config.ARTIFACTS_DIR / f"vision_head_{name}.joblib")
    arr = np.asarray(Image.new("RGB", (640, 480), (120, 130, 140)).resize((160, 160)), dtype="float32")[None]
    for _ in range(5):
        head.predict_proba(np.asarray(extractor(arr, training=False)))
    times = []
    for _ in range(40):
        t0 = time.perf_counter()
        head.predict_proba(np.asarray(extractor(arr, training=False)))
        times.append((time.perf_counter() - t0) * 1000)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return {"parametros": int(base.count_params()), "peso_float32_mb": round(base.count_params() * 4 / 1e6, 1),
            "latencia_ms_media": round(float(np.mean(times)), 1), "latencia_ms_p95": round(float(np.percentile(times, 95)), 1),
            "memoria_pico_mb": round(after, 0), "memoria_inicial_mb": round(before, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="", help="folder with the real photo crops (not in the repository)")
    ap.add_argument("--medir", default="", help="internal: print the cost of one backbone as JSON")
    args = ap.parse_args()
    if args.medir:
        print("COSTO " + json.dumps(measure_cost(args.medir)))
        return

    import joblib
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from casaguacate import config, vision

    data = config.DATA_DIR
    out = APP / "artefactos"
    out.mkdir(exist_ok=True)
    classes, all_classes = vision.CLASSES, vision.ALL_CLASSES

    def ahash(path, n=8):
        a = np.asarray(Image.open(path).convert("L").resize((n, n)), dtype=np.float32)
        return tuple((a > a.mean()).flatten())

    paths, labels, seen = [], [], set()
    for k, c in enumerate(classes):          # same near-duplicate removal as the notebook
        for p in sorted(glob.glob(str(data / "imagenes" / c / "*.jpg"))):
            h = ahash(p)
            if h not in seen:
                seen.add(h)
                paths.append(p)
                labels.append(k)
    paths, labels = np.array(paths), np.array(labels)
    r_tr, r_tmp, e_tr, e_tmp = train_test_split(paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
    r_va, r_te, e_va, e_te = train_test_split(r_tmp, e_tmp, test_size=0.50, random_state=SEED, stratify=e_tmp)
    ood = sorted(glob.glob(str(data / "fuera_de_dominio" / "*.jpg")))
    print(f"Wikimedia: {len(paths)} unicas ({len(r_tr)}/{len(r_va)}/{len(r_te)}), fuera de dominio: {len(ood)}")

    real = pd.DataFrame()
    if args.crops:
        real = pd.read_csv(APP.parent / "fotos_reales" / "etiquetas_recortes.csv")
        real["ruta"] = [os.path.join(os.path.expanduser(args.crops), f) for f in real.archivo]
        print(f"Fotos reales (recortes): {len(real)}", real.clase.value_counts().to_dict())

    def load(ps):
        return np.stack([vision.to_array(Image.open(p)) for p in ps])

    x_wiki, x_ood = load(paths), load(ood)
    x_real = load(real.ruta) if len(real) else np.zeros((0, vision.IMG_SIZE, vision.IMG_SIZE, 3), dtype=np.float32)
    idx = {p: i for i, p in enumerate(paths)}
    i_tr, i_va, i_te = ([idx[p] for p in part] for part in (r_tr, r_va, r_te))
    names_wiki = np.array([classes[k] for k in labels])

    def head():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced",
                                                                  random_state=SEED))

    results = {}
    for name in vision.BACKBONES:
        print(f"== {name}")
        extractor, base = vision.build_extractor(name)
        feat = lambda x: np.asarray(extractor.predict(x, batch_size=32, verbose=0))  # noqa: E731
        f_wiki, f_ood, f_real = feat(x_wiki), feat(x_ood), (feat(x_real) if len(real) else np.zeros((0, 1)))

        # A. Wikimedia test split, five classes (head trained on train + validation)
        fit_idx = i_tr + i_va
        clf_a = head().fit(f_wiki[fit_idx], names_wiki[fit_idx])
        pred_a = clf_a.predict(f_wiki[i_te])
        true_a = names_wiki[i_te]
        res = {"wikimedia": {"n_prueba": len(i_te), "accuracy": round(accuracy_score(true_a, pred_a), 4),
                              "f1_macro": round(f1_score(true_a, pred_a, average="macro"), 4),
                              "por_clase": {k: round(v["f1-score"], 3) for k, v in
                                            classification_report(true_a, pred_a, output_dict=True,
                                                                  zero_division=0).items() if k in classes}}}

        # B. real photos, leave-one-group-out by shot, six classes with the 0.60 decision rule
        if len(real):
            y_real = real.clase.values
            pred_b = [None] * len(real)
            for g in sorted(real.toma.unique()):
                test = (real.toma == g).values
                clf = head().fit(np.vstack([f_wiki, f_ood, f_real[~test]]),
                                 np.concatenate([names_wiki, ["otro"] * len(f_ood), y_real[~test]]))
                prob = clf.predict_proba(f_real[test])
                for i, p in zip(np.where(test)[0], prob):
                    d = vision.decide(p, list(clf.classes_), config.IMAGE_THRESHOLD)
                    pred_b[i] = d.label
            present = [c for c in all_classes if c in set(y_real) | set(pred_b)]
            cm = confusion_matrix(y_real, pred_b, labels=all_classes)
            res["fotos_reales"] = {
                "n": len(real), "accuracy": round(accuracy_score(y_real, pred_b), 4),
                "f1_macro": round(f1_score(y_real, pred_b, labels=present, average="macro", zero_division=0), 4),
                "recall_interior": round(float(cm[4, 4] / max(cm[4].sum(), 1)), 3),
                "recall_otro": round(float(cm[5, 5] / max(cm[5].sum(), 1)), 3),
                "matriz": {"clases": all_classes, "valores": cm.tolist()}}

        # Final head with ALL the data (what the API uses)
        parts_x, parts_y = [f_wiki, f_ood], [names_wiki, np.array(["otro"] * len(f_ood))]
        if len(real):
            parts_x.append(f_real)
            parts_y.append(real.clase.values)
        final = head().fit(np.vstack(parts_x), np.concatenate(parts_y))
        joblib.dump(final, out / f"vision_head_{name}.joblib", compress=3)
        res["clases_cabeza"] = list(final.classes_)
        del extractor, base

        sub = subprocess.run([sys.executable, __file__, "--medir", name], capture_output=True, text=True,
                             env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"})
        line = [x for x in sub.stdout.splitlines() if x.startswith("COSTO ")]
        res["costo"] = json.loads(line[0][6:]) if line else {"error": sub.stderr[-300:]}
        results[name] = res
        print(json.dumps({k: v for k, v in res.items() if k != "fotos_reales"}, ensure_ascii=False))
        if "fotos_reales" in res:
            print("   fotos reales:", {k: v for k, v in res["fotos_reales"].items() if k != "matriz"})

    # Reference: the fine-tuned ResNet50 of the Avance 2 (5 classes, no "otro"), numbers from its own outputs
    ref = {}
    try:
        summary = json.loads((data.parent / "salidas" / "resumen.json").read_text(encoding="utf-8"))
        row = [m for m in summary["vision"]["modelos"] if m["modelo"].startswith("ResNet50")][0]
        ref["avance2_resnet50_afinada"] = {"accuracy_wikimedia": row["accuracy"], "f1_macro_wikimedia": row["f1_macro"]}
        real_json = json.loads((APP.parent / "fotos_reales" / "salidas" / "resultados_fotos_reales.json").read_text())
        ref["avance2_resnet50_afinada"]["accuracy_recortes_reales"] = real_json["recorte_base"]["accuracy"]
    except Exception as exc:  # reference numbers are optional
        ref["error"] = str(exc)

    # Choice: the lightest backbone (by memory) whose accuracy is within MAX_ACC_GAP of the ResNet50 reference
    def score(n):
        r = results[n]
        return r.get("fotos_reales", r["wikimedia"])["accuracy"], r["wikimedia"]["accuracy"]

    base_real, base_wiki = score("resnet50")
    candidates = [n for n in ("mobilenetv2", "efficientnetb0")
                  if score(n)[0] >= base_real - MAX_ACC_GAP and score(n)[1] >= base_wiki - MAX_ACC_GAP]
    chosen = min(candidates, key=lambda n: results[n]["costo"].get("memoria_pico_mb", 1e9)) if candidates else "resnet50"
    meta = {"despliegue": chosen, "referencia": "resnet50", "clases": all_classes, "umbral": config.IMAGE_THRESHOLD,
            "tamano_entrada": vision.IMG_SIZE, "criterio": f"el mas ligero en memoria con accuracy a menos de {MAX_ACC_GAP} de ResNet50",
            "fotos_reales_usadas": int(len(real)), "comparacion": results, "referencia_avance2": ref}
    (out / "vision_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DESPLIEGUE:", chosen)


if __name__ == "__main__":
    main()
