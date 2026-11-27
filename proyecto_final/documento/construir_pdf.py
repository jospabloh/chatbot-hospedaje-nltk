"""Convierte documento_tecnico.md en el PDF del Proyecto Final (HTML con estilo + Chrome sin cabeza).

Que es: script de construccion del documento. Lee documento_tecnico.md, lo pasa a HTML con la biblioteca
`markdown` (tablas, bloques de codigo), le aplica una hoja de estilo academica (carta, margenes de 2.5 cm,
numero de pagina al pie, tablas y figuras que no se parten) y lo imprime a PDF con Google Chrome.
Para que sirve: regenerar Proyecto_Final_Jose_Pablo_Herrera_Fernandez.pdf cada vez que cambie el texto o las figuras.

Uso:
    python construir_pdf.py                 # escribe documento.html y el PDF junto a este archivo
    CHROME=/ruta/a/chrome python construir_pdf.py

Requiere: pip install markdown, y Google Chrome (o Chromium) instalado.
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
@page { size: Letter; margin: 2.0cm 2.1cm 2.0cm 2.1cm;
        @bottom-center { content: counter(page); font: 9pt 'Times New Roman', serif; color: #444; } }
@page :first { margin: 2.2cm; @bottom-center { content: ''; } }
html { font-family: 'Times New Roman', Times, serif; font-size: 10.4pt; line-height: 1.3; color: #111; }
body { margin: 0; text-align: justify; hyphens: auto; }
h1, h2, h3 { font-family: Helvetica, Arial, sans-serif; color: #14365d; text-align: left; break-after: avoid; }
h2 { font-size: 14pt; margin: 14pt 0 5pt; border-bottom: 1px solid #b9c6d6; padding-bottom: 2pt; }
h3 { font-size: 11.2pt; margin: 9pt 0 3pt; }
p { margin: 0 0 5.5pt; orphans: 3; widows: 3; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.6pt; background: #f1f3f6; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f5f6f8; border: 1px solid #d5dae1; border-radius: 3pt; padding: 6pt 8pt; font-size: 8pt;
      line-height: 1.28; white-space: pre-wrap; text-align: left; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.2pt; line-height: 1.2;
        font-family: Helvetica, Arial, sans-serif; text-align: left; break-inside: avoid; }
th { background: #14365d; color: #fff; padding: 3pt 4pt; border: 1px solid #14365d; }
td { padding: 1.8pt 3.5pt; border: 1px solid #c8d0db; }
tr:nth-child(even) td { background: #f4f7fb; }
figure { margin: 8pt 0 10pt; text-align: center; break-inside: avoid; }
figure img { max-width: 88%; max-height: 8.5cm; }
figcaption { font-size: 9pt; font-style: italic; color: #333; margin-top: 3pt; text-align: center; }
pre.flujo { display: inline-block; text-align: left; font-size: 7.4pt; background: #f8f9fb; }
ul, ol { margin: 0 0 7pt 0; padding-left: 20pt; }
li { margin-bottom: 2pt; }
.pb { break-after: page; }
.portada { text-align: center; padding-top: 60pt; }
.portada .inst { font: 12pt Helvetica, Arial, sans-serif; letter-spacing: .5pt; color: #14365d; }
.portada .titulo-doc { font-size: 25pt; line-height: 1.2; margin: 60pt 0 12pt; border: 0; text-align: center; color: #14365d; }
.portada .subt { font-size: 12.5pt; font-style: italic; margin-bottom: 70pt; text-align: center; }
.portada .datos { font-size: 12pt; line-height: 1.6; text-align: center; margin-bottom: 18pt; }
.portada .aviso-portada { margin-top: 70pt; font-size: 10pt; color: #7a5200; text-align: center; }
"""


def main() -> int:
    """Genera el HTML y el PDF; devuelve 0 si Chrome termino bien."""
    cuerpo = markdown.markdown(MD.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    HTML.write_text(f"<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Proyecto Final</title>"
                    f"<style>{CSS}</style></head><body>{cuerpo}</body></html>", encoding="utf-8")
    comando = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
               "--virtual-time-budget=10000", f"--user-data-dir=/tmp/pf-doc-chrome-perfil", HTML.as_uri()]
    resultado = subprocess.run(comando, capture_output=True, text=True)
    print(f"PDF: {PDF} (codigo {resultado.returncode})")
    return resultado.returncode


if __name__ == "__main__":
    sys.exit(main())
