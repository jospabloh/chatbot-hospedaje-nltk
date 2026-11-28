"""Convierte documento_tecnico.md en el PDF del Proyecto Final (HTML con estilo + Chrome sin cabeza).

Que es: script de construccion del documento. Lee documento_tecnico.md, lo pasa a HTML con la biblioteca
`markdown` (tablas, bloques de codigo), le aplica una hoja de estilo academica (carta, margenes de 2.5 cm,
numero de pagina al pie, tablas y figuras que no se parten) y lo imprime a PDF con Google Chrome.
Para que sirve: regenerar Proyecto_Final_Jose_Pablo_Herrera_Fernandez.pdf cada vez que cambie el texto o las figuras.

Uso:
    python construir_pdf.py                 # escribe documento.html y el PDF junto a este archivo
    CHROME=/ruta/a/chrome python construir_pdf.py

Requiere: pip install markdown pymupdf (pymupdf es opcional: metadatos), y Google Chrome (o Chromium) instalado.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import markdown

AQUI = Path(__file__).resolve().parent
MD = AQUI / "documento_tecnico.md"
HTML = AQUI / "documento.html"
PDF = AQUI / "Proyecto_Final_Jose_Pablo_Herrera_Fernandez.pdf"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

CSS = """
@page { size: Letter; margin: 2.0cm 2.2cm 2.0cm 2.2cm;
        @bottom-center { content: counter(page); font: 9pt Georgia, serif; color: #444; } }
@page :first { margin: 2.2cm; @bottom-center { content: ''; } }
html { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt; line-height: 1.2; color: #000; }
body { margin: 0; text-align: left; }
h1, h2, h3 { font-family: Georgia, serif; color: #000; text-align: left; break-after: avoid; }
h2 { font-size: 14pt; margin: 13pt 0 4pt; border-bottom: 1px solid #777; padding-bottom: 2pt; }
h3 { font-size: 11.5pt; margin: 8pt 0 3pt; }
p { margin: 0 0 5pt; orphans: 3; widows: 3; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f0f0f0; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f5f5f5; border: 1px solid #bbb; border-radius: 2pt; padding: 5pt 7pt; font-size: 8pt;
      line-height: 1.22; white-space: pre-wrap; text-align: left; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 5pt 0 8pt; font-size: 8.4pt; line-height: 1.15;
        font-family: Georgia, serif; text-align: left; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #d9d9d9; color: #000; padding: 2.5pt 3.5pt; border: 1px solid #777; }
td { padding: 1.6pt 3.5pt; border: 1px solid #999; }
tr:nth-child(even) td { background: #f4f4f4; }
figure { margin: 6pt 0 8pt; text-align: center; break-inside: avoid; }
figure img { max-width: 92%; max-height: 8cm; }
figcaption { font-size: 9pt; font-style: italic; color: #222; margin-top: 3pt; text-align: center; }
pre.flujo { display: inline-block; text-align: left; font-size: 7.2pt; background: #f8f8f8; }
ul, ol { margin: 0 0 5pt 0; padding-left: 20pt; }
li { margin-bottom: 1.5pt; }
.pb { break-after: page; }
.refs p { padding-left: 1.2cm; text-indent: -1.2cm; margin-bottom: 4pt; }
.portada { text-align: center; padding-top: 60pt; }
.portada .inst { font-size: 12pt; letter-spacing: .5pt; color: #000; }
.portada .titulo-doc { font-size: 25pt; line-height: 1.2; margin: 60pt 0 12pt; border: 0; text-align: center; color: #000; }
.portada .subt { font-size: 12.5pt; font-style: italic; margin-bottom: 70pt; text-align: center; }
.portada .datos { font-size: 12pt; line-height: 1.6; text-align: center; margin-bottom: 18pt; }
.portada .aviso-portada { margin-top: 70pt; font-size: 10pt; color: #333; text-align: center; }
"""


def main() -> int:
    """Genera el HTML y el PDF; devuelve 0 si Chrome termino bien."""
    cuerpo = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    marca = "<h2>Referencias</h2>"
    if marca in cuerpo:   # sangria francesa en las referencias
        cuerpo = cuerpo.replace(marca, marca + "<div class='refs'>") + "</div>"
    HTML.write_text(f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Proyecto Final</title>"
                    f"<style>{CSS}</style></head><body>{cuerpo}</body></html>", encoding="utf-8")
    comando = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
               "--virtual-time-budget=10000", f"--user-data-dir=/tmp/pf-doc-chrome-perfil", HTML.as_uri()]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    try:   # metadatos del PDF (autor y titulo)
        import fitz
        doc = fitz.open(PDF)
        doc.set_metadata({"author": "Jose Pablo Herrera Fernandez",
                          "title": "Asistente inteligente de check-in para Cas'aguacate: Proyecto Final",
                          "subject": "Desarrollo de aplicaciones con inteligencia artificial, Tecmilenio"})
        doc.saveIncr()
        print("Paginas:", len(doc))
    except Exception as exc:   # PyMuPDF es opcional
        print("Metadatos no escritos:", exc)
    print(f"PDF: {PDF} (codigo {resultado.returncode})")
    return resultado.returncode


if __name__ == "__main__":
    sys.exit(main())
