"""Final architecture diagram (PNG) of the deployed system: python scripts/figura_arquitectura_final.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "arquitectura_final.png"
fig, ax = plt.subplots(figsize=(14, 7.4))
ax.set_xlim(0, 140); ax.set_ylim(0, 74); ax.axis("off")


def box(x, y, w, h, title, text, color="#E8F0FB"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=10, weight="bold", family="serif")
    ax.text(x + w / 2, y + h - 6.0, text, ha="center", va="top", fontsize=8.2, family="serif", linespacing=1.35)


def arrow(x1, y1, x2, y2, label="", dy=1.3, dx=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
    if label:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, label, ha="center", fontsize=7.8, style="italic", family="serif")


GREY, GREEN, RED, YELLOW = "#EDEDED", "#E3F1E3", "#FBE3E0", "#FFF2CC"
box(1, 28, 15, 17, "Huésped", "Escribe en\nespañol o inglés\ny adjunta\nuna foto", GREY)
box(26, 28, 22, 17, "Interfaz Streamlit", "Chat bilingüe, subida\nde foto, traza del\nagente, estado del\nservicio (Cloud)", GREEN)
box(58, 28, 20, 17, "API REST Flask", "/health /chat /image\n/agent /metrics\nvalidación, logs JSON,\nmétricas, token", YELLOW)
box(86, 46, 22, 22, "Agente híbrido", "Capa reactiva:\nurgencias, datos\nsensibles, pedir humano,\nfuera de alcance, política\nCapa deliberativa:\nestado y decisión", YELLOW)
box(86, 4, 22, 18, "Clasificador visual", "EfficientNetB0 congelada\n(ImageNet) + capa de 6\nclases (5 + «otro»),\numbral 0.60", "#E8F0FB")
box(116, 27, 23, 18, "Chatbot NLP", "TF-IDF (palabras +\ncaracteres) + SVM\ncalibrada por idioma\n15 intenciones, umbral 0.40", "#E8F0FB")
box(116, 52, 23, 16, "Base de conocimiento", "Datos reales de\nCas'aguacate (sin\ncódigos ni Wi-Fi)", GREY)
box(58, 4, 20, 14, "Anfitriona (Lore)", "El agente marca el caso\n(avisar_anfitriona) y lo\nregistra; el canal de\nenvío es mejora futura", RED)
arrow(16, 36.5, 26, 36.5, "texto +\nfoto", dy=1.2)
arrow(48, 36.5, 58, 36.5, "HTTPS", dy=1.2)
arrow(78, 40, 86, 55, "texto, foto,\nestado", dy=2, dx=-6.5)
arrow(86, 50, 78, 36, "decisión\n+ traza", dy=-6.5, dx=2.5)
arrow(97, 46, 97, 22, "foto", dy=0, dx=2.5)
arrow(108, 54, 116, 40, "texto", dy=3.5, dx=1)
arrow(127, 45, 127, 52, "intención", dy=0, dx=5)
arrow(108, 62, 116, 62, "respuesta", dy=1.2)
arrow(68, 28, 68, 18, "escala", dy=0, dx=3)
ax.text(70, 71.5, "Arquitectura final: asistente de check-in de Cas'aguacate", ha="center", fontsize=12.5, weight="bold", family="serif")
ax.text(70, 1.2, "La UI vive en Streamlit Community Cloud; la API Flask y los modelos corren en Hugging Face Spaces (HTTPS, contenedor Docker). "
        "Nada del huésped se guarda en el servidor.", ha="center", fontsize=8, family="serif", style="italic")
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("Guardado", OUT)
