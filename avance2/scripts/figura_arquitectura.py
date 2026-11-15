"""Diagrama de arquitectura del asistente multimodal (Figura del documento)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 57); ax.axis("off")


def caja(x, y, w, h, titulo, texto, gris=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc="#EDEDED" if gris else "#E8F0FB", ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h - 2.6, titulo, ha="center", va="top", fontsize=10.5, weight="bold", family="serif")
    ax.text(x + w / 2, y + h - 6.6, texto, ha="center", va="top", fontsize=8.6, family="serif", linespacing=1.35)


def flecha(x1, y1, x2, y2, t="", dy=1.2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333333"))
    if t:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, t, ha="center", fontsize=8.3, style="italic", family="serif")


caja(1, 19, 15, 15, "Huésped", "Escribe por chat\ny manda una foto", gris=True)
caja(24, 30, 26, 19, "Clasificador de texto", "Normalización, tokenización,\nstopwords y stemming\nTF-IDF (palabras + caracteres)\nSVM lineal calibrada")
caja(24, 4, 26, 19, "Clasificador visual", "Aumento de datos\nResNet50 (ImageNet)\ncon transferencia y\najuste fino del bloque conv5")
caja(58, 17, 22, 19, "Política de decisión", "Umbral de confianza\nLa foto decide en\nmensajes de acceso\nReglas de escalamiento")
caja(88, 30, 21, 17, "Base de conocimiento", "Respuestas exactas\ndel Avance 1\n(datos sintéticos)", gris=True)
caja(88, 5, 21, 17, "Anfitriona", "Recibe solo lo que\nnecesita una persona,\ncon la foto etiquetada", gris=True)
flecha(16, 30, 24, 39, "texto")
flecha(16, 23, 24, 13, "foto")
flecha(50, 39, 58, 31, "intención y p")
flecha(50, 13, 58, 22, "clase y q")
flecha(80, 31, 88, 38, "RESPONDER")
flecha(80, 22, 88, 14, "ESCALAR", dy=-3.2)
ax.plot([98.5, 98.5, 8.5], [47.6, 53.5, 53.5], color="#777777", lw=1.0)
ax.annotate("", xy=(8.5, 34.4), xytext=(8.5, 53.5), arrowprops=dict(arrowstyle="->", lw=1.0, color="#777777"))
ax.text(52, 54.5, "respuesta al huésped", ha="center", fontsize=8.3, style="italic", family="serif")
plt.tight_layout()
plt.savefig("salidas/arquitectura_v2.png", dpi=180)
print("ok")
