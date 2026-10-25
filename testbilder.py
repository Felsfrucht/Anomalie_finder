"""
testbilder.py - erzeugt synthetische Testbilder mit bekannter Lösung.

Jede Funktion liefert (PIL-Bild, Liste der wahren Anomalie-Rahmen (x0,y0,x1,y1)).

    python testbilder.py            # schreibt alle Bilder in ./testbilder/
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def _gull_mask(w: int, h: int, cx: float, cy: float, span: float = 42.0, width: float = 5.0):
    """Möwe als 'M'-förmige Flügel, 4-fach überabgetastet und weich verkleinert."""
    ss = 4
    m = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(m)
    pts_l = [(cx - span, cy - 4), (cx - span * 0.55, cy - 20), (cx - span * 0.2, cy - 14), (cx, cy)]
    pts_r = [(cx, cy), (cx + span * 0.2, cy - 14), (cx + span * 0.55, cy - 20), (cx + span, cy - 4)]
    for pts in (pts_l, pts_r):
        d.line([(x * ss, y * ss) for x, y in pts], fill=255, width=int(width * ss), joint="curve")
    d.ellipse(((cx - 3) * ss, (cy - 4) * ss, (cx + 3) * ss, (cy + 4) * ss), fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    return np.asarray(m, np.float32) / 255.0, (cx - span - 3, cy - 24, cx + span + 3, cy + 6)


def moewe(w=1200, h=800, seed=1, wolken=False):
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None, None]
    top = np.array([35, 95, 200], np.float32)
    bot = np.array([150, 200, 245], np.float32)
    sky = top + (bot - top) * y ** 1.3
    sky = np.broadcast_to(sky, (h, w, 3)).copy()
    if wolken:
        n = rng.normal(size=(h // 8, w // 8)).astype(np.float32)
        n = np.asarray(Image.fromarray(n).resize((w, h), Image.BICUBIC), np.float32)
        n = ndi.gaussian_filter(n, 14)
        n = (n - n.mean()) / n.std()
        cloud = np.clip((n - 0.6) / 1.2, 0, 1)[..., None] * 0.55
        sky = sky * (1 - cloud) + np.array([245, 247, 250], np.float32) * cloud
    sky += rng.normal(0, 2.0, sky.shape)
    mask, box = _gull_mask(w, h, cx=150, cy=110)
    gull = np.array([248, 248, 244], np.float32)
    sky = sky * (1 - mask[..., None]) + gull * mask[..., None]
    return _to_img(sky), [tuple(int(v) for v in box)]


def sterne(w=1200, h=800, seed=2, n_stars=1400):
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, h)[:, None, None]
    bg = np.array([5, 7, 15], np.float32) + np.array([6, 6, 10], np.float32) * (yy ** 2)
    img = np.broadcast_to(bg, (h, w, 3)).copy()
    img += rng.normal(0, 1.4, img.shape)

    px, py = 880, 210           # Planet
    pr = 8
    tints = np.array([[255, 255, 255], [255, 238, 205], [200, 220, 255]], np.float32) / 255.0
    for _ in range(n_stars):
        sx, sy = rng.uniform(6, w - 6), rng.uniform(6, h - 6)
        if (sx - px) ** 2 + (sy - py) ** 2 < (pr + 14) ** 2:
            continue
        bright = float(np.clip(rng.lognormal(4.4, 0.7), 35, 255))
        sig = rng.choice([0.6, 0.8, 1.0, 1.3])
        tint = tints[rng.integers(0, 3)]
        x0, y0 = int(sx) - 4, int(sy) - 4
        gy, gx = np.mgrid[y0:y0 + 9, x0:x0 + 9]
        g = np.exp(-((gx + 0.5 - sx) ** 2 + (gy + 0.5 - sy) ** 2) / (2 * sig ** 2))
        img[y0:y0 + 9, x0:x0 + 9] += bright * g[..., None] * tint

    # Planet: weich begrenzte Scheibe mit Schattierung und einem Band
    ss = 4
    S = (2 * pr + 8) * ss
    gy, gx = np.mgrid[0:S, 0:S]
    c = S / 2
    dist = np.hypot(gx + 0.5 - c, gy + 0.5 - c) / ss
    disc = np.clip((pr - dist) * 1.0 + 0.5, 0, 1)
    light = np.clip(1.05 - 0.55 * ((gx - c * 0.7) ** 2 + (gy - c * 0.7) ** 2) ** 0.5 / (c * 1.2), 0.35, 1.1)
    band = 1 - 0.18 * (np.abs(gy - c * 1.05) < 2.2 * ss)
    col = np.array([225, 150, 72], np.float32)
    patch = (col * (light * band)[..., None]).astype(np.float32)
    disc_s = np.asarray(Image.fromarray((disc * 255).astype(np.uint8)).resize((S // ss, S // ss), Image.LANCZOS), np.float32) / 255
    patch_s = np.asarray(Image.fromarray(np.clip(patch, 0, 255).astype(np.uint8)).resize((S // ss, S // ss), Image.LANCZOS), np.float32)
    n = S // ss
    x0, y0 = px - n // 2, py - n // 2
    region = img[y0:y0 + n, x0:x0 + n]
    img[y0:y0 + n, x0:x0 + n] = region * (1 - disc_s[..., None]) + patch_s * disc_s[..., None]
    return _to_img(img), [(px - pr - 2, py - pr - 2, px + pr + 3, py + pr + 3)]


def muster(w=400, h=300, seed=3):
    """Senkrechte Streifen R,G,B,R,G,B... (jede Farbe ist häufig!) mit Reihenfolge-Fehlern."""
    rng = np.random.default_rng(seed)
    cols = np.array([[205, 45, 45], [45, 180, 65], [55, 75, 210]], np.float32)
    xs = np.arange(w) % 3
    img = cols[xs][None, :, :].repeat(h, axis=0).copy()
    truth = []
    # a) einzelnes falsches Pixel (G statt B)
    x, y = 101, 60
    assert xs[x] == 2
    img[y, x] = cols[1]
    truth.append((x - 2, y - 2, x + 3, y + 3))
    # b) Block mit umgekehrter Reihenfolge (R,B,G statt R,G,B)
    bx, by, bw, bh = 249, 150, 12, 8
    for xx in range(bx, bx + bw):
        img[by:by + bh, xx] = cols[[0, 2, 1][(xx - bx) % 3]]
    truth.append((bx - 2, by - 2, bx + bw + 2, by + bh + 2))
    # c) waagrechter Abschnitt, in dem sich die Streifenphase verschiebt (ein Pixel doppelt)
    x, y0, y1 = 330, 200, 230
    img[y0:y1, x] = img[y0:y1, x - 1]
    truth.append((x - 2, y0 - 1, x + 3, y1 + 1))
    img += rng.normal(0, 1.2, img.shape)
    return _to_img(img), truth


def rauschen(w=800, h=600, seed=4):
    """Glatter Verlauf mit Rauschen und leichter Vignette - ohne Anomalie."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([90 + 60 * xx / w, 110 + 50 * yy / h, 150 + 40 * (1 - xx / w)], axis=-1).astype(np.float32)
    vig = 1 - 0.25 * (((xx - w / 2) / w) ** 2 + ((yy - h / 2) / h) ** 2)
    img = base * vig[..., None] + rng.normal(0, 2.2, base.shape)
    return _to_img(img), []


def gelaende(w=1000, h=700, seed=5):
    """Natürliche Textur (Mehrskalen-Rauschen) als JPEG, plus ein rotes Objekt."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w, 3), np.float32)
    amp = 1.0
    for octave in range(2, 8):
        n = rng.normal(size=(h // 2 ** (8 - octave) + 2, w // 2 ** (8 - octave) + 2, 3)).astype(np.float32)
        layer = np.stack([np.asarray(Image.fromarray(n[..., c]).resize((w, h), Image.BICUBIC)) for c in range(3)], -1)
        acc += layer * amp
        amp *= 0.55
    acc = (acc - acc.mean((0, 1))) / acc.std((0, 1))
    img = np.array([95, 110, 70], np.float32) + acc * np.array([22, 20, 16], np.float32)
    ox, oy, r = 720, 430, 11
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.clip(r - np.hypot(xx - ox, yy - oy) + 0.5, 0, 1)[..., None]
    img = img * (1 - m) + np.array([230, 30, 40], np.float32) * m
    pil = _to_img(img)
    import io
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=82)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), [(ox - r - 2, oy - r - 2, ox + r + 3, oy + r + 3)]


ALLE = {
    "moewe": lambda: moewe(),
    "moewe_wolken": lambda: moewe(wolken=True, seed=7),
    "sterne": lambda: sterne(),
    "muster": lambda: muster(),
    "rauschen": lambda: rauschen(),
    "gelaende": lambda: gelaende(),
}


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testbilder")
    os.makedirs(out, exist_ok=True)
    for name, fn in ALLE.items():
        img, truth = fn()
        ext = "jpg" if name == "gelaende" else "png"
        path = os.path.join(out, f"{name}.{ext}")
        img.save(path, quality=90) if ext == "jpg" else img.save(path)
        print(f"{path}  {img.size}  Anomalien: {truth}")


if __name__ == "__main__":
    sys.exit(main())
