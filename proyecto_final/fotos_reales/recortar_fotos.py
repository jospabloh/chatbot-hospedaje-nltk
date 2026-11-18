"""Extract the individual photos from the brochure pages (light background, photos are dense textured blocks).

Usage: python recortar_fotos.py <pages dir> <output dir>. Keep the crops listed in etiquetas_recortes.csv
(banners, logos and sub-crops that repeat a larger crop are left out by hand).
"""
import glob, os, sys
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = []
for f in sorted(glob.glob(os.path.join(src, "*.jpg"))):
    im = np.asarray(Image.open(f).convert("RGB"))
    g = im.astype(np.float32).mean(axis=2)
    fg = g < 228
    fg = ndi.binary_opening(fg, structure=np.ones((9, 9)))
    lab, n = ndi.label(fg)
    base = os.path.basename(f)[13:19]
    k = 0
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        y, x = sl[0].start, sl[1].start
        h, w = sl[0].stop - y, sl[1].stop - x
        if w < 180 or h < 180:
            continue
        fill = (lab[sl] == i).mean()
        if fill < 0.55:
            continue
        crop = im[sl]
        if crop.std() < 35:
            continue
        k += 1
        name = f"{base}_r{k}.jpg"
        Image.fromarray(crop).save(os.path.join(out, name), quality=92)
        rows.append((name, x, y, w, h, round(float(fill), 2)))
for r in rows:
    print(*r)
print(len(rows), "recortes")

fs = sorted(glob.glob(os.path.join(out, "*.jpg")))
T, cols = 250, 6
for s in range(0, len(fs), 24):
    chunk = fs[s:s + 24]
    rws = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * T, rws * (T + 22)), "white")
    d = ImageDraw.Draw(sheet)
    for j, p in enumerate(chunk):
        t = Image.open(p); t.thumbnail((T - 6, T - 6))
        x0, y0 = (j % cols) * T, (j // cols) * (T + 22)
        sheet.paste(t, (x0 + 3, y0 + 22)); d.text((x0 + 4, y0 + 4), os.path.basename(p), fill="red")
    sheet.save(os.path.join(out, f"_hoja_{s // 24 + 1}.png"))
