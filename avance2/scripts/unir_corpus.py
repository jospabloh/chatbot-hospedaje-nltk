"""Une el corpus v1 (9-nov, 11 intenciones del negocio) con el corpus v2 (14 intenciones
alineadas al Avance 1) y deja un solo archivo sin duplicados: corpus_intenciones.csv."""
import pandas as pd

MAPA_V1 = {
    "como_llegar": "ubicacion", "codigo_acceso": "cerradura", "estacionamiento": "porton_estacionamiento",
    "horarios_checkin_checkout": "horarios", "llegada_tarde": "llegada_tarde",
    "early_late_checkout": "early_late_checkout", "desayuno": "servicios", "factura_electronica": "servicios",
    "tarifa_corporativa": "contactar_anfitrion", "reportar_problema": "reportar_problema",
}
SALUDOS = {"hola buenas tardes", "gracias, muy amable", "buenos dias", "ok perfecto", "ya quedo, gracias",
           "que tengan bonito dia", "hasta luego", "excelente servicio, muchas gracias", "buenas noches",
           "que onda", "una pregunta", "disculpa la molestia"}

v1 = pd.read_csv("intenciones_v1.csv", sep="|")
v1.loc[v1["texto"].isin(SALUDOS), "intencion"] = "saludo_cortesia"
fuera = v1[(v1["intencion"] == "fuera_de_alcance")]
fuera.to_csv("fuera_de_alcance.csv", sep="|", index=False)  # se usan para probar el umbral, no para entrenar
v1 = v1[v1["intencion"] != "fuera_de_alcance"].copy()
v1["intencion"] = v1["intencion"].replace(MAPA_V1)

v2 = pd.read_csv("intenciones_v2_base.csv", sep="|")
corpus = pd.concat([v2, v1], ignore_index=True)
corpus["texto"] = corpus["texto"].str.strip()
corpus = corpus.drop_duplicates(subset="texto", keep="first").reset_index(drop=True)
corpus.to_csv("corpus_intenciones.csv", sep="|", index=False)
print(corpus["intencion"].value_counts().to_string())
print("total", len(corpus), "| fuera de alcance para prueba:", len(fuera))
