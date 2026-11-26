"""Regression: the case battery of the Avance 2 (integracion*.csv) replayed against the final agent.

The expectation is the decision FAMILY of the Avance 2 (RESPONDER* or ESCALAR). A case may differ only if it is
listed in DOCUMENTED_CHANGES with the design reason; the test also checks that the new behaviour is the documented one.
Set GUARDAR_EVIDENCIAS=1 to write evidencias/regresion_avance2.csv.
"""
import csv
import os
from pathlib import Path

import pytest

from casaguacate.agente import HybridAgent
from tests import avance2

pytestmark = pytest.mark.modelos

# (source file, text, photo) -> (new action family, reason)
DOCUMENTED_CHANGES = {
    ("integracion.csv", "hace mucho calor y no enfria", "controles"):
        ("ESCALAR", "reportes de fallas: la base de conocimiento real no tiene guia para el aire; antes se respondia con un texto inventado"),
    ("integracion_bilingue.csv", "the ac is not cooling", "controles"):
        ("ESCALAR", "igual que el caso anterior, en ingles"),
    ("integracion.csv", "recomiendan algun restaurante por aca", ""):
        ("ESCALAR", "regla reactiva de fuera de alcance: antes se respondia con la lista de servicios"),
    ("integracion_mejorada.csv", "recomiendan algun restaurante por aca", ""):
        ("ESCALAR", "regla reactiva de fuera de alcance"),
    ("integracion_bilingue.csv", "can we check out later, our flight is at night", ""):
        ("ESCALAR", "regla de politica: el late check-out lo decide la anfitriona (antes se contestaba con la info de check-in)"),
    ("integracion_fuera_de_dominio.csv", "mira esto", "perro_1.jpg"):
        ("ESCALAR", "la clase 'otro' rechaza la foto de un perro que el modelo de 5 clases aceptaba como caseta (0.86)"),
}


@pytest.fixture(scope="module")
def agent():
    from casaguacate import config
    if not (config.ARTIFACTS_DIR / "intenciones_es.joblib").exists():
        pytest.skip("faltan los artefactos")
    from casaguacate.servicios import load_services
    services = load_services()
    if services.vision is None:
        pytest.skip("sin clasificador visual")
    return HybridAgent(services)


@pytest.fixture(scope="module")
def replay(agent):
    rows = []
    for case in avance2.cases():
        result = agent.run(case["texto"], avance2.photo_for(case["row"]))
        rows.append({**case, "accion": result.action, "intencion": result.final_intent, "decidido_por": result.decided_by,
                     "familia_ok": avance2.family(result.action) == avance2.family(case["accion_esperada"]),
                     "intencion_ok": result.final_intent == case["intencion_esperada"]})
    if os.environ.get("GUARDAR_EVIDENCIAS"):
        out = Path(__file__).resolve().parents[1] / "evidencias" / "regresion_avance2.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fuente", "texto", "foto", "accion_avance2", "intencion_avance2", "accion_final", "intencion_final",
                             "decidido_por", "misma_familia", "cambio_documentado"])
            for r in rows:
                key = (r["fuente"], r["texto"], r["foto"])
                writer.writerow([r["fuente"], r["texto"], r["foto"], r["accion_esperada"], r["intencion_esperada"], r["accion"], r["intencion"],
                                 r["decidido_por"], "si" if r["familia_ok"] else "no", DOCUMENTED_CHANGES.get(key, ("", ""))[1]])
    return rows


def test_la_bateria_tiene_los_29_casos_del_avance2(replay):
    assert len(replay) == 29


def test_toda_diferencia_esta_documentada_y_es_la_esperada(replay):
    for r in replay:
        key = (r["fuente"], r["texto"], r["foto"])
        if r["familia_ok"]:
            assert key not in DOCUMENTED_CHANGES, f"el cambio documentado ya no ocurre: {key}"
        else:
            assert key in DOCUMENTED_CHANGES, f"diferencia sin documentar: {key} esperado {r['accion_esperada']}, obtuvo {r['accion']}"
            assert avance2.family(r["accion"]) == DOCUMENTED_CHANGES[key][0], key


def test_la_mayoria_conserva_la_decision_del_avance2(replay):
    same = sum(r["familia_ok"] for r in replay)
    assert same >= 23, f"{same} de {len(replay)}"


def test_las_intenciones_coinciden_en_su_mayoria(replay):
    same = sum(r["intencion_ok"] for r in replay)
    assert same / len(replay) >= 0.75, f"{same} de {len(replay)}"


def test_ningun_caso_de_acceso_con_foto_confiable_se_degrada(replay):
    for r in replay:
        if r["foto"] in ("cerradura", "porton", "caseta"):
            assert r["accion"].startswith("RESPONDER"), r["texto"]
