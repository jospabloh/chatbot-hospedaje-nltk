"""NLP component: preprocessing, language detection and the intent / tone classifiers.

The preprocessing is the one of the Avance 2 notebook (accent-free normalization, stopwords without the
question and negation words, Snowball stemming). The classifiers are the TF-IDF (words + characters) and
calibrated linear SVM pipelines, one per language, exported by `scripts/entrenar_texto.py`.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize

from . import config

# Words that carry the intent ("cuanto" is rates, "donde" is location, "no abre" is a failure): never dropped.
KEEP_WORDS = {
    "es": {"no", "cuanto", "cuantos", "cuanta", "donde", "como", "cuando", "antes", "mas", "hasta", "desde", "sin",
           "que", "hay", "puedo", "tarde", "temprano"},
    "en": {"no", "not", "how", "where", "when", "what", "before", "after", "until", "from", "without", "can",
           "late", "early", "more", "there"},
}

ES_MARKERS = {"el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "es", "por", "para", "con", "mi",
              "se", "como", "donde", "cuando", "cuanto", "hay", "esta", "puedo", "al", "hola", "gracias", "buenas",
              "buenos", "tiene", "tienen", "quiero", "hora", "noche", "favor", "abre", "tarde", "ya", "estoy", "lo",
              "le", "su"}
EN_MARKERS = {"the", "is", "are", "do", "does", "you", "i", "we", "my", "our", "can", "how", "what", "where", "when",
              "there", "to", "of", "and", "in", "it", "for", "please", "hi", "hello", "thanks", "thank", "have",
              "not", "with", "at", "on", "this", "be", "will", "time", "night", "wont", "dont", "open", "late"}

_STEMMERS = {"es": SnowballStemmer("spanish"), "en": SnowballStemmer("english")}


def normalize(text: str) -> str:
    """Lowercase, no accents (keeps the n with tilde), no punctuation, no letters repeated more than twice."""
    text = text.lower().replace("ñ", "\0")
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("\0", "ñ")
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return re.sub(r"[^a-zñ0-9 ]", " ", text)


def _load_stopwords() -> dict[str, set[str]]:
    path = Path(__file__).with_name("stopwords.json")
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"es": [], "en": []}
    return {lang: set(words) - KEEP_WORDS[lang] for lang, words in data.items()}


STOPWORDS = _load_stopwords()


def preprocess(text: str, lang: str = "es", stem: bool = True, remove_stopwords: bool = True) -> str:
    tokens = [t for t in wordpunct_tokenize(normalize(text)) if t.isalnum()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS.get(lang, set())]
    if stem:
        tokens = [_STEMMERS[lang].stem(t) for t in tokens]
    return " ".join(tokens)


def features(text: str, lang: str) -> pd.DataFrame:
    """The two columns the TF-IDF ColumnTransformer expects."""
    return pd.DataFrame({"norm": [preprocess(text, lang, stem=False, remove_stopwords=False)],
                         "proc": [preprocess(text, lang)]})


def detect_language(text: str, default: str = "es") -> str:
    """Function-word overlap; a tie keeps `default` (the language the guest used before)."""
    tokens = wordpunct_tokenize(normalize(text))
    es = sum(t in ES_MARKERS for t in tokens)
    en = sum(t in EN_MARKERS for t in tokens)
    if en > es:
        return "en"
    if es > en:
        return "es"
    return default if default in ("es", "en") else "es"


@dataclass
class TextPrediction:
    label: str
    confidence: float
    top: list = field(default_factory=list)  # [(label, probability)] best three

    def as_dict(self) -> dict:
        return {"etiqueta": self.label, "confianza": round(self.confidence, 3),
                "top3": [{"etiqueta": k, "confianza": round(p, 3)} for k, p in self.top]}


class IntentClassifier:
    """One calibrated SVM pipeline per language (labels are the 14 intents of the corpus)."""

    def __init__(self, models: dict):
        self.models = models

    @classmethod
    def load(cls, directory: Path | None = None) -> "IntentClassifier":
        directory = Path(directory or config.ARTIFACTS_DIR)
        return cls({lang: joblib.load(directory / f"intenciones_{lang}.joblib") for lang in ("es", "en")})

    @property
    def labels(self) -> list[str]:
        return list(self.models["es"].classes_)

    def predict(self, text: str, lang: str) -> TextPrediction:
        model = self.models.get(lang, self.models["es"])
        prob = model.predict_proba(features(text, lang))[0]
        order = np.argsort(prob)[::-1][:3]
        return TextPrediction(str(model.classes_[order[0]]), float(prob[order[0]]),
                              [(str(model.classes_[i]), float(prob[i])) for i in order])


class ToneClassifier:
    """Positive / neutral / negative tone (one model for both languages, characters TF-IDF)."""

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, directory: Path | None = None) -> "ToneClassifier":
        return cls(joblib.load(Path(directory or config.ARTIFACTS_DIR) / "tono.joblib"))

    def predict(self, text: str) -> TextPrediction:
        prob = self.model.predict_proba([normalize(text)])[0]
        order = np.argsort(prob)[::-1][:3]
        return TextPrediction(str(self.model.classes_[order[0]]), float(prob[order[0]]),
                              [(str(self.model.classes_[i]), float(prob[i])) for i in order])
