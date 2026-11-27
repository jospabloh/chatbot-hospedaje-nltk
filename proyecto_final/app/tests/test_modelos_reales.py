"""Tests with the real exported artifacts (scikit-learn + TensorFlow). Skipped if the artifacts are missing."""
import csv
import io
import json
import time

import pytest
from PIL import Image

from casaguacate import config
from casaguacate.api import create_app

pytestmark = pytest.mark.modelos


@pytest.fixture(scope="module")
def real_services():
    if not (config.ARTIFACTS_DIR / "intenciones_es.joblib").exists():
        pytest.skip("faltan los artefactos: corre scripts/entrenar_texto.py y scripts/entrenar_vision.py")
    from casaguacate.servicios import load_services
    services = load_services()
    if services.vision is None:
        pytest.skip("no se pudo cargar el clasificador visual (¿sin red para bajar los pesos de ImageNet?)")
    return services


@pytest.fixture(scope="module")
def real_client(real_services):
    return create_app(real_services, {"RATE_LIMIT_PER_MIN": 0}).test_client()


def test_el_clasificador_cargado_reproduce_las_metricas_guardadas(real_services):
    import pandas as pd
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from casaguacate import texto

    metrics = json.loads((config.ARTIFACTS_DIR / "metricas_texto.json").read_text(encoding="utf-8"))
    for lang, name in [("es", "corpus_intenciones.csv"), ("en", "corpus_intenciones_en.csv")]:
        corpus = pd.read_csv(config.DATA_DIR / name, sep="|")
        _, test_texts, _, y_te = train_test_split(corpus["texto"], corpus["intencion"], test_size=0.2, random_state=42,
                                                  stratify=corpus["intencion"])
        pred = [real_services.intents.predict(t, lang).label for t in test_texts]
        accuracy = accuracy_score(y_te, pred)
        assert abs(accuracy - metrics[lang]["accuracy"]) < 0.005
        assert accuracy >= 0.70, f"accuracy del texto en {lang}: {accuracy:.3f}"
    assert texto.STOPWORDS["es"] and texto.STOPWORDS["en"]


def test_el_modelo_visual_de_despliegue_cumple_el_criterio_de_seleccion():
    meta = json.loads((config.ARTIFACTS_DIR / "vision_meta.json").read_text(encoding="utf-8"))
    chosen, ref = meta["comparacion"][meta["despliegue"]], meta["comparacion"][meta["referencia"]]
    assert chosen["wikimedia"]["accuracy"] >= ref["wikimedia"]["accuracy"] - 0.02
    assert chosen["fotos_reales"]["accuracy"] >= ref["fotos_reales"]["accuracy"] - 0.02
    assert chosen["costo"]["peso_float32_mb"] < ref["costo"]["peso_float32_mb"]
    assert chosen["wikimedia"]["accuracy"] >= 0.90


def test_fotos_reales_de_interiores_se_reconocen(real_services):
    with open(config.DATA_DIR / "fotos_reales" / "origen_fotos_reales.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["esperada"] == "interior"]
    hits = sum(real_services.vision.predict(Image.open(config.DATA_DIR / "fotos_reales" / r["archivo"]).convert("RGB")).label == "interior"
               for r in rows)
    assert len(rows) >= 8 and hits / len(rows) >= 0.75


def test_foto_fuera_de_dominio_no_se_acepta_como_clase_del_negocio(real_services):
    flagged = 0
    for name in ["perro_0.jpg", "perro_1.jpg", "comida_0.jpg", "comida_1.jpg", "auto_0.jpg", "auto_1.jpg"]:
        flagged += real_services.vision.predict(Image.open(config.DATA_DIR / "fuera_de_dominio" / name).convert("RGB")).label == "otro"
    assert flagged >= 5  # the Avance 2 model accepted 5 of the 6 as one of its classes; the head with "otro" must reject them


def test_flujo_completo_por_la_api_con_modelos_reales(real_client):
    health = real_client.get("/health").get_json()
    assert health["estado"] == "ok" and health["modelos"]["vision"] == json.loads(
        (config.ARTIFACTS_DIR / "vision_meta.json").read_text(encoding="utf-8"))["despliegue"]
    body = real_client.post("/agent", json={"texto": "¿A qué hora es el check-in?"}).get_json()
    assert body["decision"]["accion"] == "RESPONDER" and "15:00" in body["decision"]["respuesta"]
    buffer = io.BytesIO()
    Image.open(config.DATA_DIR / "fotos_reales" / "real_1.jpg").convert("RGB").save(buffer, format="JPEG")
    image = real_client.post("/image", data={"file": (io.BytesIO(buffer.getvalue()), "real_1.jpg")}, content_type="multipart/form-data")
    assert image.status_code == 200 and image.get_json()["etiqueta"] in ("interior", "otro", "cerradura", "porton", "caseta", "controles")
    mixed = real_client.post("/agent", data={"texto": "mira esto", "file": (io.BytesIO(buffer.getvalue()), "real_1.jpg")},
                             content_type="multipart/form-data").get_json()
    assert [s["componente"] for s in mixed["traza"]] == ["Agente", "Capa reactiva", "Chatbot NLP", "Clasificador visual", "Capa deliberativa"]


def test_latencia_aceptable_en_cpu(real_client):
    buffer = io.BytesIO()
    Image.open(config.DATA_DIR / "fotos_reales" / "real_1.jpg").convert("RGB").save(buffer, format="JPEG")
    timings = []
    for _ in range(8):
        start = time.perf_counter()
        real_client.post("/agent", data={"texto": "no me abre", "file": (io.BytesIO(buffer.getvalue()), "a.jpg")},
                         content_type="multipart/form-data")
        timings.append((time.perf_counter() - start) * 1000)
    assert sorted(timings)[len(timings) // 2] < 1500, timings  # median under 1.5 s on a laptop CPU


# La intencion `cancelacion` existe porque la app viva respondia "Perdon, no entendi" a "¿Como puedo cancelar mi reservacion?".
# Esta prueba usa el modelo entrenado real: si alguien reentrena sin las frases de cancelacion (o las mezcla con otra
# intencion), el huesped vuelve a recibir una aclaracion en lugar de la respuesta, y aqui se nota.
@pytest.mark.parametrize("text", ["¿Cómo puedo cancelar mi reservación?", "quiero un reembolso", "ya no voy a ir",
                                  "puedo cambiar las fechas de mi reserva", "how can I cancel my booking?",
                                  "is there a refund if I cancel"])
def test_la_cancelacion_se_reconoce_con_el_modelo_real(real_services, text):
    from casaguacate.agente import HybridAgent
    result = HybridAgent(real_services).run(text)
    assert result.final_intent == "cancelacion", result.trace
    assert result.action == "RESPONDER_Y_AVISAR" and result.notify_host and "no entend" not in result.reply.lower()


# Agregar una intencion no puede quitarle frases a las otras (check-in y mascotas eran los casos que ya funcionaban).
@pytest.mark.parametrize("text,lang,intent", [("¿A qué hora es el check in?", "es", "horarios"), ("¿aceptan mascotas?", "es", "reglas"),
                                              ("¿cuánto cuesta la noche?", "es", "tarifas"), ("what time is check in", "en", "horarios"),
                                              ("¿hay estacionamiento?", "es", "porton_estacionamiento")])
def test_agregar_cancelacion_no_rompe_otras_intenciones(real_services, text, lang, intent):
    assert real_services.intents.predict(text, lang).label == intent
