"""Train and export the text artifacts: intent classifiers (es, en), tone classifier and stopword lists.

Same recipe as the Avance 2 notebook (TF-IDF of words and characters + calibrated linear SVM, 80/20 stratified
split, seed 42), but exporting the fitted pipelines with joblib so the API does not retrain anything.

    python scripts/entrenar_texto.py

Writes in artefactos/: intenciones_es.joblib, intenciones_en.joblib, tono.joblib, metricas_texto.json
and casaguacate/stopwords.json (NLTK lists, accent-free, so the API needs no corpus download).
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

SEED = 42


def write_stopwords() -> None:
    import nltk
    from nltk.corpus import stopwords
    nltk.download("stopwords", quiet=True)
    es = sorted({unicodedata.normalize("NFD", w).encode("ascii", "ignore").decode() for w in stopwords.words("spanish")})
    en = sorted(set(stopwords.words("english")))
    (APP / "casaguacate" / "stopwords.json").write_text(json.dumps({"es": es, "en": en}, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    write_stopwords()
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.svm import LinearSVC

    from casaguacate import config, texto

    data = config.DATA_DIR
    out = APP / "artefactos"
    out.mkdir(exist_ok=True)

    def tfidf():
        return ColumnTransformer([
            ("palabras", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), "proc"),
            ("caracteres", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True), "norm")])

    def frame(texts, lang):
        return pd.DataFrame({"norm": [texto.preprocess(t, lang, stem=False, remove_stopwords=False) for t in texts],
                             "proc": [texto.preprocess(t, lang) for t in texts]})

    metrics = {"semilla": SEED, "umbral_texto": config.TEXT_THRESHOLD}
    oos = pd.read_csv(data / "fuera_de_alcance.csv", sep="|")["texto"].tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for lang, name in [("es", "corpus_intenciones.csv"), ("en", "corpus_intenciones_en.csv")]:
        corpus = pd.read_csv(data / name, sep="|")
        x = frame(corpus["texto"], lang)
        x_tr, x_te, y_tr, y_te = train_test_split(x, corpus["intencion"], test_size=0.2, random_state=SEED,
                                                  stratify=corpus["intencion"])
        cv_f1 = cross_val_score(make_pipeline(tfidf(), LinearSVC(C=0.5, random_state=SEED)), x_tr, y_tr, cv=cv,
                                scoring="f1_macro")
        model = make_pipeline(tfidf(), CalibratedClassifierCV(LinearSVC(C=0.5, random_state=SEED), cv=5)).fit(x_tr, y_tr)
        pred = model.predict(x_te)
        conf = model.predict_proba(x_te).max(axis=1)
        oos_conf = model.predict_proba(frame(oos, lang)).max(axis=1)
        report = classification_report(y_te, pred, zero_division=0, output_dict=True)
        metrics[lang] = {
            "frases": int(len(corpus)), "n_prueba": int(len(y_te)),
            "accuracy": round(accuracy_score(y_te, pred), 4),
            "f1_macro": round(f1_score(y_te, pred, average="macro"), 4),
            "f1_macro_cv": round(float(cv_f1.mean()), 4), "f1_macro_cv_std": round(float(cv_f1.std()), 4),
            "por_intencion": {k: {m: round(v, 3) for m, v in d.items()} for k, d in report.items()
                              if k in model.classes_},
            "cobertura_umbral": round(float((conf >= config.TEXT_THRESHOLD).mean()), 4),
            "accuracy_cubierta": round(float((pred == y_te.values)[conf >= config.TEXT_THRESHOLD].mean()), 4),
            "fuera_de_alcance_bajo_umbral": round(float((oos_conf < config.TEXT_THRESHOLD).mean()), 4),
        }
        joblib.dump(model, out / f"intenciones_{lang}.joblib", compress=3)
        print(lang, {k: v for k, v in metrics[lang].items() if k != "por_intencion"})

    # Tone: Spanish 80/20 split; English 70/30 split (the Avance 2 English tone set was evaluation only)
    tone_es = pd.read_csv(data / "corpus_tono.csv", sep="|")
    tone_en = pd.read_csv(data / "corpus_tono_en.csv", sep="|")
    es_tr, es_te, ys_tr, ys_te = train_test_split(tone_es["texto"], tone_es["tono"], test_size=0.2, random_state=SEED,
                                                  stratify=tone_es["tono"])
    en_tr, en_te, ye_tr, ye_te = train_test_split(tone_en["texto"], tone_en["tono"], test_size=0.3, random_state=SEED,
                                                  stratify=tone_en["tono"])
    clean = lambda ts: [texto.normalize(t) for t in ts]  # noqa: E731
    tone = make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        CalibratedClassifierCV(LinearSVC(C=1, random_state=SEED), cv=5))
    tone.fit(clean(list(es_tr) + list(en_tr)), list(ys_tr) + list(ye_tr))
    for lang, xs, ys in [("es", es_te, ys_te), ("en", en_te, ye_te)]:
        pred = tone.predict(clean(xs))
        metrics[f"tono_{lang}"] = {"n_prueba": int(len(ys)), "accuracy": round(accuracy_score(ys, pred), 4),
                                    "f1_macro": round(f1_score(ys, pred, average="macro"), 4)}
        print("tono", lang, metrics[f"tono_{lang}"])
    # The tone model is measured but NOT exported: with 180 Spanish phrases a TF-IDF model is far below the MiniLM
    # model of the Avance 2 (F1 0.917), so the agent uses the lexicon rule of the reactive layer for tone.
    metrics["tono_desplegado"] = False
    (out / "metricas_texto.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
