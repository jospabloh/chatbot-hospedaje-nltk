"""Selección manual tras revisar las hojas de contacto: solo se conservan las fotos que sí muestran
la clase. Copia las elegidas a datos/imagenes/<clase>/ y sus créditos a datos/creditos_imagenes.csv."""
import csv, os, shutil

SEL = [  # (origen, carpeta, clase destino, ids)
    ("imagenes_raw", "cerradura", "cerradura", "000 006 008 014 015 016 017 018 019 026 032 033 039 042 046 047 049 063 065 072 075 082 083 085 094 095 096 097 111 112 116 117 118 119 122 128 129"),
    ("imagenes_raw", "porton", "porton", "002 010 013 019 022 023 024 026 028 032 041 043 044 045 046 047 050 051 053 056 057 058 063 066 076 077 078 079 082 087 089 090 091 093 096 098 099 103 104 106 111 112 119 121 122 123 124 126 129"),
    ("imagenes_raw", "porton", "caseta", "003 004 005 006 007 009 059"),
    ("imagenes_raw", "caseta", "caseta", "003 009 010 011 022 023 024 026 027 028 033 036 038 050 051 052 063 064 067 069 070 075 076 077 086 087 090 092 093 094 099 102 103 109 118 119 124 125 126 127"),
    ("imagenes_raw", "controles", "controles", "004 007 008 010 011 017 018 019 023 032 033 034 039 040 041 043 055 056 057 058 059 063 068 076 077 078 082 083 084 085 086 088 090 097 098 099 100 101 105 106 107 108 110 111 112 119 120 122 128 129"),
    ("imagenes_raw", "interior", "interior", "000 002 003 004 005 006 008 009 010 011 012 013 015 016 021 023 024 025 027 029 030 032 033 034 035 036 038 040 041 043 044 045 046 047 048 049 050 051 052 053 054 055 056 057 058 059 060 062 063 064 065 067 068 070 071 072 073 074 075 077"),
    ("imagenes_raw2", "cerradura", "cerradura", "021 022 023 024 025 026 027 038 041 042 043 044 055 056 079 084"),
]
EXTRA_LISTA = []

if __name__ == "__main__":
    SEL += EXTRA_LISTA
    creditos = {}
    for origen, archivo in [("imagenes_raw", "creditos.csv"), ("imagenes_raw2", "creditos2.csv")]:
        for f in csv.DictReader(open(archivo)):
            creditos[(origen, f["archivo"])] = f
    filas = []
    shutil.rmtree("datos/imagenes", ignore_errors=True)
    for origen, carpeta, clase, ids in SEL:
        os.makedirs(f"datos/imagenes/{clase}", exist_ok=True)
        for i in ids.split():
            src = f"{origen}/{carpeta}/{carpeta}_{i}.jpg"
            dst_nombre = f"{clase}_{carpeta[:3]}{origen[-1] if origen.endswith('2') else ''}{i}.jpg"
            shutil.copy(src, f"datos/imagenes/{clase}/{dst_nombre}")
            c = creditos.get((origen, f"{carpeta}_{i}.jpg"), {})
            filas.append([clase, dst_nombre, c.get("titulo_commons", ""), c.get("licencia", ""), c.get("autor", ""), c.get("url", "")])
    with open("datos/creditos_imagenes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clase", "archivo", "titulo_commons", "licencia", "autor", "url"])
        w.writerows(filas)
    for clase in sorted({x[2] for x in SEL}):
        print(clase, len(os.listdir(f"datos/imagenes/{clase}")))
