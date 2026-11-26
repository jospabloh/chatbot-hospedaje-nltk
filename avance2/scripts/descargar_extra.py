"""Descarga imágenes con licencia libre de Wikimedia Commons por categoría (más limpias que la
búsqueda por texto). Guarda licencia, autor y URL de cada archivo en creditos.csv.
Uso: python descargar_imagenes.py [explorar]"""
import csv, hashlib, os, re, sys, time
import requests

UA = {"User-Agent": "TecmilenioAvance2/1.0 (proyecto academico de maestria)"}
API = "https://commons.wikimedia.org/w/api.php"
DESTINO = "imagenes_raw2"
LIMITE = 90

CATEGORIAS = {
    "cerradura": ["Deadbolts", "Door locks", "Electronic door locks", "Door handles with locks", "Card readers", "Mortise locks"],
}


def api(params, intentos=5):
    params = {"format": "json", "action": "query", **params}
    for k in range(intentos):
        r = requests.get(API, params=params, headers=UA, timeout=40)
        time.sleep(1.0)
        if r.status_code == 200 and r.text.startswith("{"):
            return r.json()
        time.sleep(30 * (k + 1))
    return {}


def miembros(cat, tipo):
    out, cont = [], {}
    while True:
        j = api({"list": "categorymembers", "cmtitle": "Category:" + cat, "cmtype": tipo, "cmlimit": 500, **cont})
        out += j.get("query", {}).get("categorymembers", [])
        if "continue" not in j:
            return out
        cont = j["continue"]


def archivos_de(cat, profundidad=1):
    titulos = [m["title"] for m in miembros(cat, "file")]
    if profundidad:
        for sub in miembros(cat, "subcat")[:15]:
            titulos += archivos_de(sub["title"][9:], profundidad - 1)
    return titulos


def info(titulos):
    j = api({"titles": "|".join(titulos), "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 320})
    return list(j.get("query", {}).get("pages", {}).values())


def limpio(t):
    return re.sub(r"<[^>]+>", "", t or "").strip()[:120]


def main(explorar=False):
    vistos, filas = set(), []
    for clase, cats in CATEGORIAS.items():
        titulos = []
        for c in cats:
            t = archivos_de(c)
            print(f"  {clase} <- {c}: {len(t)}", flush=True)
            titulos += [x for x in t if re.search(r"\.(jpe?g|png)$", x, re.I) and x not in titulos]
        print(clase, "candidatos", len(titulos), flush=True)
        if explorar:
            continue
        os.makedirs(os.path.join(DESTINO, clase), exist_ok=True)
        n = 0
        for i in range(0, len(titulos), 40):
            if n >= LIMITE:
                break
            for pg in info(titulos[i:i + 40]):
                if n >= LIMITE:
                    break
                ii = (pg.get("imageinfo") or [{}])[0]
                meta = ii.get("extmetadata", {})
                lic = limpio(meta.get("LicenseShortName", {}).get("value"))
                if not ii.get("thumburl") or not re.search(r"CC|Public domain|PD", lic, re.I):
                    continue
                try:
                    rr = requests.get(ii["thumburl"], headers=UA, timeout=40)
                    if rr.status_code == 429:
                        time.sleep(30)
                        rr = requests.get(ii["thumburl"], headers=UA, timeout=40)
                    if rr.status_code != 200:
                        continue
                    datos = rr.content
                except Exception:
                    continue
                h = hashlib.md5(datos).hexdigest()
                if h in vistos or len(datos) < 3000:
                    continue
                vistos.add(h)
                nombre = f"{clase}_{n:03d}.jpg"
                with open(os.path.join(DESTINO, clase, nombre), "wb") as f:
                    f.write(datos)
                filas.append([clase, nombre, pg["title"], lic, limpio(meta.get("Artist", {}).get("value")),
                              ii.get("descriptionurl")])
                n += 1
                time.sleep(0.6)
        print(clase, "descargadas", n, flush=True)
    if not explorar:
        with open("creditos2.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clase", "archivo", "titulo_commons", "licencia", "autor", "url"])
            w.writerows(filas)


if __name__ == "__main__":
    main(explorar=len(sys.argv) > 1)
