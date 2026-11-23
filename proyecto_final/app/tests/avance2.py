"""Helpers to replay the Avance 2 case battery (avance2/salidas/integracion*.csv) against the final agent."""
from __future__ import annotations

import csv
import glob

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from casaguacate import config

SEED = 42
CLASSES = ["cerradura", "porton", "caseta", "controles", "interior"]
OUT = config.DATA_DIR.parent / "salidas"


def _ahash(path, n=8):
    a = np.asarray(Image.open(path).convert("L").resize((n, n)), dtype=np.float32)
    return tuple((a > a.mean()).flatten())


def notebook_test_split():
    """Same near-duplicate removal and 70/15/15 split as the Avance 2 notebook: returns (paths, labels) of the test set."""
    paths, labels, seen = [], [], set()
    for k, c in enumerate(CLASSES):
        for p in sorted(glob.glob(str(config.DATA_DIR / "imagenes" / c / "*.jpg"))):
            h = _ahash(p)
            if h not in seen:
                seen.add(h)
                paths.append(p)
                labels.append(k)
    paths, labels = np.array(paths), np.array(labels)
    _, p_tmp, _, e_tmp = train_test_split(paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
    _, p_te, _, e_te = train_test_split(p_tmp, e_tmp, test_size=0.50, random_state=SEED, stratify=e_tmp)
    return p_te, e_te


_SPLIT = None


def photo_of_class(class_name: str, n: int = 0) -> Image.Image:
    """The n-th test photo of a class: the same picture the notebook used for `test_photo(class_name)` of the notebook."""
    global _SPLIT
    _SPLIT = _SPLIT or notebook_test_split()
    paths, labels = _SPLIT
    return Image.open(paths[np.where(labels == CLASSES.index(class_name))[0][n]]).convert("RGB")


def read_csv(name: str) -> list[dict]:
    with open(OUT / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def photo_for(row: dict) -> Image.Image | None:
    """Photo of a case row: a test photo of the class, an out-of-domain / real file, or none."""
    if row.get("archivo"):
        folder = "fuera_de_dominio" if row["archivo"].split("_")[0] in ("perro", "comida", "auto") else "fotos_reales"
        return Image.open(config.DATA_DIR / folder / row["archivo"]).convert("RGB")
    cls = row.get("imagen") or row.get("foto")
    return photo_of_class(cls) if cls else None


def cases() -> list[dict]:
    """All the cases of the four Avance 2 CSVs with the expected family of action (RESPONDER* or ESCALAR)."""
    out = []
    for name, action_col, intent_col in [("integracion.csv", "accion", "intencion_final"),
                                         ("integracion_bilingue.csv", "accion", "intencion_final"),
                                         ("integracion_fuera_de_dominio.csv", "accion", "intencion_final"),
                                         ("integracion_mejorada.csv", "mejorado_accion", "mejorado_intencion")]:
        for row in read_csv(name):
            out.append({"fuente": name, "texto": row["texto"], "foto": row.get("archivo") or row.get("imagen") or row.get("foto") or "",
                        "row": row, "accion_esperada": row[action_col], "intencion_esperada": row[intent_col]})
    return out


def family(action: str) -> str:
    return "ESCALAR" if action.startswith("ESCALAR") else "RESPONDER"
