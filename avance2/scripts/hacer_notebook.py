"""Convierte avance2.py (celdas marcadas con '# %%') en avance2.ipynb y lo ejecuta completo."""
import sys
import nbformat
from nbclient import NotebookClient

fuente = open("avance2.py", encoding="utf-8").read()
celdas = []
for bloque in fuente.split("# %%")[1:]:
    if bloque.startswith(" [markdown]"):
        lineas = bloque.split("\n")[1:]
        celdas.append(nbformat.v4.new_markdown_cell("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in lineas).strip()))
    else:
        celdas.append(nbformat.v4.new_code_cell(bloque.strip("\n")))
nb = nbformat.v4.new_notebook(cells=celdas)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": []}
if "--ejecutar" in sys.argv:
    NotebookClient(nb, timeout=3600, kernel_name="tec-avance2", resources={"metadata": {"path": "."}}).execute()
nbformat.write(nb, "avance2.ipynb")
print("celdas:", len(celdas))
