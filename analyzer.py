"""
analyzer.py - Engine des Anomalie-Finders
=========================================

Findet in einem Foto Pixel und Pixelabfolgen, die nicht zum Rest des Bildes
passen. Das Bild wird nicht mit festen Regeln verglichen, sondern lernt aus
sich selbst, was "normal" ist.

Drei Detektoren (jeder liefert eine "Überraschungs-Karte" in Bit):

1. Farbseltenheit
   Wie häufig ist die Farbe eines Pixels im ganzen Bild?
   (Möwe im blauen Himmel: Weiß ist selten.)

2. Pixelfolgen
   Das Bild wird auf wenige Farbklassen reduziert. Dann wird gezählt, welche
   Farbklasse in welcher Richtung (waagrecht, senkrecht, 2x diagonal) und in
   welchem Abstand neben welcher anderen vorkommt.
   - Paare:   P(Farbe A | Nachbar B)      "neben B steht normalerweise A"
   - Tripel:  P(Mitte B | links A, rechts C)
   Ein Pixel ist auffällig, wenn es zu seinen Nachbarn nicht passt, selbst
   wenn jede einzelne Farbe im Bild häufig ist.

3. Strukturseltenheit
   Kleine Bildausschnitte (5x5, in zwei Maßstäben) werden per PCA verdichtet.
   Selten ist ein Ausschnitt, wenn es im Bild kaum ähnliche gibt (k-nächste
   Nachbarn).

Jede Karte wird robust (Median/MAD) in z-Werte umgerechnet und danach
gewichtet kombiniert. Die Empfindlichkeit legt nur noch die Schwelle fest,
deshalb reagiert der Regler sofort, ohne das Bild neu zu berechnen.

Treffer werden zu Regionen zusammengefasst. Ein Wiederholungsfilter erklärt
viele gleichartige Regionen (z. B. Sterne) für normal.

Benutzung als Modul:
    res = analyze(Image.open("foto.jpg"))
    det = detect(res, DetectParams(sensitivity=60))
    for r in det.regions: print(r.id, r.x0, r.y0, r.x1, r.y1, r.score)

Benutzung auf der Kommandozeile:
    python analyzer.py foto.jpg --sens 60 --out ergebnis.png --csv funde.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi
try:
    # scipy.spatial ist schnell, wird aber z. B. von Windows "Smart App Control" teils blockiert.
    # Dann (oder mit ANOMALIE_NO_KDTREE=1) rechnet die App mit einer reinen numpy-Variante.
    if os.environ.get("ANOMALIE_NO_KDTREE"):
        raise ImportError("kd-Baum per Umgebungsvariable abgeschaltet")
    from scipy.spatial import cKDTree
except Exception:                                   # noqa: BLE001  (ImportError, DLL-Sperre ...)
    cKDTree = None

Image.MAX_IMAGE_PIXELS = 400_000_000  # große Kamerabilder erlauben

# --------------------------------------------------------------------------
# Einstellungen
# --------------------------------------------------------------------------

# Schwellen-Bereich: Empfindlichkeit 0 -> T_MAX (nur Offensichtliches),
# Empfindlichkeit 100 -> T_MIN (schon kleine Abweichungen).
T_MAX = 40.0
T_MIN = 3.0
# Totzone: z-Werte bis hierhin zählen als normale Schwankung / Rauschen.
DEADBAND = 2.0
# Wiederholungsfilter: so ähnlich (in Merkmals-Einheiten) müssen zwei Regionen sein, um als "gleichartig" zu gelten.
REPEAT_TOL = 1.0


@dataclass
class AnalysisParams:
    """Einstellungen, die eine NEUE Berechnung erfordern."""

    max_side: int = 1200        # längste Bildkante der Analyse (größere Bilder werden verkleinert)
    color_bin: float = 6.0      # Farbgenauigkeit der Farbstatistik (Lab-Einheiten, kleiner = genauer)
    color_smooth: float = 0.7   # leichte Rauschglättung (Pixel) nur für die Farbstatistik
    seq_bin: float = 12.0       # Farbklassen-Breite für Pixelfolgen (Lab-Einheiten)
    seq_reach: int = 2          # Pixelfolgen bis zu so vielen Pixeln Abstand
    seq_power: float = 2.0      # Schärfe der Pixelfolgen-Mittelung (2 = quadratisch, höher = schärfer)
    seq_grids: int = 2          # Anzahl gegeneinander verschobener Klassenraster (1 = schneller)
    use_structure: bool = True  # Strukturdetektor (der langsamste) ein/aus
    struct_scales: Tuple[int, ...] = (1, 3)
    struct_k: int = 8           # k-ter Nachbar für die Strukturseltenheit
    struct_ref: int = 30000     # Größe der Vergleichsstichprobe
    seed: int = 0


@dataclass
class DetectParams:
    """Einstellungen, die sich sofort (ohne Neuberechnung) auswirken."""

    sensitivity: float = 50.0   # 0..100, höher = schlägt schneller an
    w_color: float = 1.0        # Gewicht Farbseltenheit 0..1
    w_seq: float = 1.0          # Gewicht Pixelfolgen 0..1
    w_struct: float = 1.0       # Gewicht Strukturseltenheit 0..1
    min_area: int = 4           # Mindestgröße einer Region (Pixel im Analysebild)
    merge_px: int = 3           # Regionen mit kleinerem Abstand werden verschmolzen
    hole_px: int = 200          # kleine Löcher in Regionen werden gefüllt
    repeat_filter: bool = True  # viele gleichartige Regionen gelten als normal
    repeat_group: int = 6       # ab so vielen ähnlichen Regionen
    max_regions: int = 500


def threshold_from_sensitivity(sensitivity: float) -> float:
    s = min(max(float(sensitivity), 0.0), 100.0) / 100.0
    return T_MAX * (T_MIN / T_MAX) ** s


# --------------------------------------------------------------------------
# Hilfen: Fortschritt, Abbruch
# --------------------------------------------------------------------------

class Cancelled(Exception):
    """Wird geworfen, wenn die Analyse abgebrochen wurde."""


class _Progress:
    def __init__(self, cb=None, cancel=None):
        self.cb = cb
        self.cancel = cancel

    def __call__(self, frac: float, text: Optional[str] = None):
        if self.cancel is not None and self.cancel():
            raise Cancelled()
        if self.cb is not None:
            self.cb(min(max(frac, 0.0), 1.0), text)

    def sub(self, a: float, b: float):
        def f(frac, text=None):
            self(a + (b - a) * frac, text)
        return f


# --------------------------------------------------------------------------
# Bild laden und vorbereiten
# --------------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    """Lädt ein Foto, beachtet die EXIF-Drehung und liefert RGB."""
    img = Image.open(path)
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        bg.alpha_composite(img)
        img = bg
    return img.convert("RGB")


def prepare_image(image: Image.Image, max_side: int) -> Tuple[np.ndarray, float]:
    """Gibt (rgb_uint8, scale) zurück. scale = Originalbreite / Analysebreite."""
    img = image.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if max_side and longest > max_side:
        f = max_side / longest
        nw, nh = max(1, round(w * f)), max(1, round(h * f))
        img = img.resize((nw, nh), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.uint8)
    scale = w / arr.shape[1]
    return arr, scale


_SRGB_LUT = np.array(
    [((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92 for c in range(256)],
    dtype=np.float32,
)
_RGB2XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float32,
)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (uint8, ...,3) -> CIE-Lab (float32, ...,3)."""
    lin = _SRGB_LUT[rgb]
    xyz = (lin @ _RGB2XYZ.T) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0).astype(np.float32)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


def robust_z(x: np.ndarray, floor: float) -> np.ndarray:
    """(x - Median) / max(1.4826*MAD, floor) - robust gegen Ausreißer."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return ((x - med) / max(mad, floor)).astype(np.float32)


# --------------------------------------------------------------------------
# Detektor 1: Farbseltenheit
# --------------------------------------------------------------------------

def _quantize_lab(lab: np.ndarray, bin_size: float, shift: float = 0.0):
    q = np.floor(lab / bin_size + shift).astype(np.int64)
    flat3 = q.reshape(-1, 3)
    q -= flat3.min(axis=0)
    dims = q.reshape(-1, 3).max(axis=0) + 1
    code = (q[..., 0] * dims[1] + q[..., 1]) * dims[2] + q[..., 2]
    return code, dims


def color_surprise(lab: np.ndarray, bin_size: float, smooth_bins: float = 1.0) -> np.ndarray:
    """-log2 P(Farbe) mit geglättetem Farbhistogramm. Ergebnis in Bit."""
    code, dims = _quantize_lab(lab, bin_size)
    n = code.size
    total = int(np.prod(dims))
    if total > 60_000_000:
        raise ValueError("Farbgenauigkeit zu fein für diesen Farbraum - bitte größeren Wert wählen.")
    hist = np.bincount(code.ravel(), minlength=total).astype(np.float32).reshape(tuple(dims))
    sm = ndi.gaussian_filter(hist, smooth_bins, mode="constant")
    p = sm / max(float(sm.sum()), 1.0)
    p = np.maximum(p, 0.5 / n)
    bits = -np.log2(p.ravel()[code.ravel()])
    return bits.reshape(code.shape).astype(np.float32)


# --------------------------------------------------------------------------
# Detektor 2: Pixelfolgen
# --------------------------------------------------------------------------

def quantize_labels(lab: np.ndarray, bin_size: float, shift: float = 0.0):
    """Farbklassen (0..K-1) für die Pixelfolgen-Statistik.

    shift verschiebt das Klassenraster um einen Bruchteil der Klassenbreite.
    """
    code, _ = _quantize_lab(lab, bin_size, shift)
    uniq, inv = np.unique(code.ravel(), return_inverse=True)
    return inv.reshape(code.shape).astype(np.int64), len(uniq)


def _count_of(codes: np.ndarray, size: int) -> np.ndarray:
    """Für jedes Element: wie oft kommt sein Code insgesamt vor (inkl. sich selbst)."""
    if size <= 6_000_000:
        bc = np.bincount(codes, minlength=size)
        return bc[codes]
    _, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    return cnt[inv]


_DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]  # waagrecht, senkrecht, diagonal, Gegendiagonale


def sequence_surprise(labels: np.ndarray, K: int, reach: int, lam: float = 1.0,
                      progress: Optional[Callable] = None, power: float = 2.0):
    """
    Liefert (pair_bits, triple_bits, weight): pro Pixel die Überraschung in Bit,
    gemittelt über alle Richtungen/Abstände mit einem Potenzmittel (power=1:
    Durchschnitt, power=2: quadratisch, große power: nähert sich dem Maximum,
    d. h. eine einzige verletzte Nachbarschaftsbeziehung schlägt durch) und
    ein Gewicht 0..1, das angibt, wie viele Nachbarbeziehungen das Pixel
    hatte (Randpixel haben weniger, also weniger Beweiskraft).

    pair:   -log2 P(Farbe | Nachbar) in beide Richtungen
    triple: -log2 P(Mitte | links, rechts); Rückfall auf die Paarwahrscheinlichkeit
    Jedes Pixel wird "leave-one-out" bewertet: sein eigenes Vorkommen zählt
    nicht mit, sonst würde sich ein einzelner Ausreißer selbst normalisieren.
    """
    H, W = labels.shape
    N = H * W
    marg = np.bincount(labels.ravel(), minlength=K).astype(np.float64) / N

    s2 = np.zeros((H, W), np.float64)
    cnt = np.zeros((H, W), np.float64)
    t2 = np.zeros((H, W), np.float64)
    tcnt = np.zeros((H, W), np.float64)

    steps = [(r, d) for r in range(1, reach + 1) for d in _DIRS]
    for i, (r, (dy, dx)) in enumerate(steps):
        if progress:
            progress(i / len(steps), f"Pixelfolgen (Richtung {i + 1}/{len(steps)}) ...")
        oy, ox = dy * r, dx * r
        if oy >= H or abs(ox) >= W:
            continue

        # ---------------- Paare: Pixel x und Pixel x+o ----------------
        ys0, ys1 = 0, H - oy
        xs0, xs1 = (0, W - ox) if ox >= 0 else (-ox, W)
        A = labels[ys0:ys1, xs0:xs1]
        B = labels[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox]
        a = A.ravel()
        b = B.ravel()
        c_ab = _count_of(a * K + b, K * K).astype(np.float64)
        na = np.bincount(a, minlength=K)[a].astype(np.float64)
        nb = np.bincount(b, minlength=K)[b].astype(np.float64)
        pf = (c_ab - 1.0 + lam * marg[a]) / (nb - 1.0 + lam)   # P(a | b)
        pb = (c_ab - 1.0 + lam * marg[b]) / (na - 1.0 + lam)   # P(b | a)
        sf = (-np.log2(np.clip(pf, 1e-12, 1.0))) ** power
        sb = (-np.log2(np.clip(pb, 1e-12, 1.0))) ** power
        shp = A.shape
        s2[ys0:ys1, xs0:xs1] += sf.reshape(shp)
        cnt[ys0:ys1, xs0:xs1] += 1
        s2[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox] += sb.reshape(shp)
        cnt[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox] += 1

        # ---------------- Tripel: links x-o, Mitte x, rechts x+o -------
        aox = abs(ox)
        if H - 2 * oy <= 0 or W - 2 * aox <= 0:
            continue
        ys0, ys1 = oy, H - oy
        xs0, xs1 = aox, W - aox
        L = labels[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox].ravel()
        M = labels[ys0:ys1, xs0:xs1].ravel()
        R = labels[ys0 + oy:ys1 + oy, xs0 + ox:xs1 + ox].ravel()
        c_lmr = _count_of((L * K + M) * K + R, K * K * K).astype(np.float64)
        c_lr = _count_of(L * K + R, K * K).astype(np.float64)
        c_lm = _count_of(L * K + M, K * K).astype(np.float64)
        c_mr = _count_of(M * K + R, K * K).astype(np.float64)
        n_l = np.bincount(L, minlength=K)[L].astype(np.float64)
        n_r = np.bincount(R, minlength=K)[R].astype(np.float64)
        p_m_l = (c_lm - 1.0 + lam * marg[M]) / (n_l - 1.0 + lam)
        p_m_r = (c_mr - 1.0 + lam * marg[M]) / (n_r - 1.0 + lam)
        prior = 0.5 * (p_m_l + p_m_r)
        p3 = (c_lmr - 1.0 + lam * prior) / (c_lr - 1.0 + lam)
        s3 = (-np.log2(np.clip(p3, 1e-12, 1.0))) ** power
        t2[ys0:ys1, xs0:xs1] += s3.reshape(ys1 - ys0, xs1 - xs0)
        tcnt[ys0:ys1, xs0:xs1] += 1

    pair_bits = (s2 / np.maximum(cnt, 1)) ** (1.0 / power)
    trip_bits = (t2 / np.maximum(tcnt, 1)) ** (1.0 / power)
    pair_bits = pair_bits.astype(np.float32)
    trip_bits = trip_bits.astype(np.float32)
    # Randpixel ohne Tripel: Wert der Paare übernehmen
    trip_bits = np.where(tcnt > 0, trip_bits, pair_bits).astype(np.float32)
    weight = np.sqrt(cnt / max(float(cnt.max()), 1.0)).astype(np.float32)
    return pair_bits, trip_bits, weight


# --------------------------------------------------------------------------
# Detektor 3: Strukturseltenheit
# --------------------------------------------------------------------------

def _kth_dist_numpy(ref: np.ndarray, Q: np.ndarray, kk: int) -> np.ndarray:
    """Abstand zum kk-ten nächsten Nachbarn ohne kd-Baum (blockweise, nur numpy)."""
    ref = ref.astype(np.float32)
    Q = Q.astype(np.float32)
    r2 = (ref * ref).sum(axis=1)
    out = np.empty(len(Q), np.float32)
    step = 2000
    for i in range(0, len(Q), step):
        q = Q[i:i + step]
        d2 = (q * q).sum(axis=1)[:, None] + r2[None, :] - 2.0 * (q @ ref.T)
        np.maximum(d2, 0, out=d2)
        out[i:i + step] = np.sqrt(np.partition(d2, kk - 1, axis=1)[:, kk - 1])
    return out


def _count_within(feat: np.ndarray, tol: float = 1.0) -> np.ndarray:
    """Pro Zeile: Anzahl Zeilen (inkl. sich selbst) mit Abstand <= tol in jeder Dimension."""
    n = len(feat)
    order = np.argsort(feat[:, 0])
    f = feat[order]
    f0 = f[:, 0]
    counts = np.zeros(n, np.int64)
    step = 256
    for i in range(0, n, step):
        j = min(n, i + step)
        lo = np.searchsorted(f0, f0[i] - tol, side="left")
        hi = np.searchsorted(f0, f0[j - 1] + tol, side="right")
        cand = f[lo:hi]
        close = (np.abs(f[i:j, None, :] - cand[None, :, :]).max(axis=2) <= tol)
        counts[order[i:j]] = close.sum(axis=1)
    return counts


def structure_rarity(lab: np.ndarray, step: int, rng: np.random.Generator,
                     n_comp: int = 8, ref_n: int = 30000, kk: int = 8,
                     progress: Optional[Callable] = None) -> np.ndarray:
    """
    log2 des Abstands zum kk-ten ähnlichsten 5x5-Ausschnitt (Abstand der
    PCA-Merkmale). step > 1 spreizt den Ausschnitt (größerer Maßstab).
    """
    H, W, _ = lab.shape
    src = lab if step == 1 else ndi.gaussian_filter(lab, (step * 0.5, step * 0.5, 0))
    offs = [(dy * step, dx * step) for dy in range(-2, 3) for dx in range(-2, 3)]
    pad = 2 * step
    padded = np.pad(src, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

    # PCA-Basis aus einer Stichprobe von Ausschnitten
    n_s = min(20000, H * W)
    ys = rng.integers(0, H, n_s)
    xs = rng.integers(0, W, n_s)
    P = np.concatenate([padded[ys + pad + dy, xs + pad + dx] for dy, dx in offs], axis=1)
    mu = P.mean(axis=0)
    cov = np.cov((P - mu), rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    basis = evecs[:, -n_comp:][:, ::-1].astype(np.float32)   # (75, n_comp)
    mu = mu.astype(np.float32)

    # Projektion aller Ausschnitte = Summe verschobener Matrixprodukte
    proj = np.zeros((H * W, n_comp), np.float32)
    for k, (dy, dx) in enumerate(offs):
        sl = padded[pad + dy:pad + dy + H, pad + dx:pad + dx + W, :].reshape(-1, 3)
        proj += sl @ basis[3 * k:3 * k + 3, :]
    proj -= mu @ basis
    proj = proj.reshape(H, W, n_comp)

    flat = proj.reshape(-1, n_comp)
    fast = cKDTree is not None
    # Ohne kd-Baum: kleinere Vergleichsstichprobe (k im gleichen Verhältnis) und gröberes Raster
    ref_eff = ref_n if fast else min(ref_n, 6000)
    kk_eff = kk if fast else max(1, int(round(kk * ref_eff / ref_n)))
    ref_idx = rng.integers(0, H * W, min(ref_eff, H * W))
    ref = flat[ref_idx]

    if fast:
        stride = 1 if H * W <= 600_000 else 2
    else:
        stride = max(1, int(math.ceil(math.sqrt(H * W / 110_000))))
    sub = proj[::stride, ::stride]
    Q = sub.reshape(-1, n_comp)
    if fast:
        dk = cKDTree(ref).query(Q, k=kk_eff, workers=-1)[0][:, -1]
    else:
        dk = _kth_dist_numpy(ref, Q, kk_eff)
    dk = dk.reshape(sub.shape[:2])
    if stride > 1:
        dk = np.repeat(np.repeat(dk, stride, axis=0), stride, axis=1)[:H, :W]
    return np.log2(dk + 1e-3).astype(np.float32)


# --------------------------------------------------------------------------
# Analyse (rechenintensiv, einmal pro Bild)
# --------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    shape: Tuple[int, int]            # (H, W) des Analysebilds
    orig_size: Tuple[int, int]        # (Breite, Höhe) des Originals
    scale: float                      # Original / Analyse
    rgb: np.ndarray                   # Analysebild
    z_color: np.ndarray
    z_seq: np.ndarray
    z_struct: np.ndarray
    params: AnalysisParams
    timings: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict, repr=False)   # Zwischenspeicher der Erkennung


def analyze(image: Image.Image, params: Optional[AnalysisParams] = None,
            progress: Optional[Callable] = None,
            cancel: Optional[Callable[[], bool]] = None) -> AnalysisResult:
    params = params or AnalysisParams()
    prog = _Progress(progress, cancel)
    timings = {}
    t0 = time.perf_counter()

    prog(0.0, "Bild vorbereiten ...")
    rgb, scale = prepare_image(image, params.max_side)
    H, W = rgb.shape[:2]
    if H < 8 or W < 8:
        raise ValueError("Das Bild ist zu klein für die Analyse (mindestens 8x8 Pixel).")
    lab = rgb_to_lab(rgb)
    timings["vorbereiten"] = time.perf_counter() - t0

    # 1) Farbseltenheit
    t = time.perf_counter()
    prog(0.04, "Farbstatistik ...")
    lab_c = lab
    if params.color_smooth > 0:
        lab_c = ndi.gaussian_filter(lab, (params.color_smooth, params.color_smooth, 0))
    bits_c = color_surprise(lab_c, params.color_bin)
    z_color = robust_z(bits_c, 1.0)
    timings["farbe"] = time.perf_counter() - t

    # 2) Pixelfolgen - auf mehreren gegeneinander verschobenen Klassenrastern.
    #    Pro Pixel gilt der unauffälligste Wert: Rauschen, das zufällig über eine
    #    Klassengrenze springt, ist im anderen Raster normal; eine echte Anomalie
    #    ist in allen Rastern auffällig.
    t = time.perf_counter()
    prog(0.10, "Pixelfolgen ...")
    shifts = [0.0, 0.5][:max(1, params.seq_grids)]
    best_pair = best_trip = None
    K = 0
    for gi, sh in enumerate(shifts):
        labels, K_g = quantize_labels(lab, params.seq_bin, sh)
        K = max(K, K_g)
        lo = 0.10 + 0.45 * gi / len(shifts)
        hi = 0.10 + 0.45 * (gi + 1) / len(shifts)
        pb, tb, weight = sequence_surprise(labels, K_g, max(1, params.seq_reach),
                                           progress=prog.sub(lo, hi), power=params.seq_power)
        best_pair = pb if best_pair is None else np.minimum(best_pair, pb)
        best_trip = tb if best_trip is None else np.minimum(best_trip, tb)
    z_seq = np.maximum(robust_z(best_pair, 1.0), robust_z(best_trip, 1.0))
    z_seq = np.where(z_seq > 0, z_seq * weight, z_seq).astype(np.float32)
    timings["folgen"] = time.perf_counter() - t
    timings["farbklassen"] = K

    # 3) Struktur
    t = time.perf_counter()
    if params.use_structure:
        rng = np.random.default_rng(params.seed)
        zs = []
        scales = list(params.struct_scales)
        for i, st in enumerate(scales):
            a = 0.55 + 0.43 * i / len(scales)
            prog(a, f"Strukturmuster (Maßstab {st}) ...")
            rar = structure_rarity(lab, st, rng, ref_n=params.struct_ref, kk=params.struct_k)
            zs.append(robust_z(rar, 0.3))
        z_struct = np.maximum.reduce(zs)
    else:
        z_struct = np.zeros((H, W), np.float32)
    timings["struktur"] = time.perf_counter() - t

    prog(1.0, "Fertig")
    timings["gesamt"] = time.perf_counter() - t0
    return AnalysisResult(shape=(H, W), orig_size=image.size, scale=scale, rgb=rgb,
                          z_color=z_color, z_seq=z_seq, z_struct=z_struct,
                          params=params, timings=timings)


# --------------------------------------------------------------------------
# Erkennung (schnell, reagiert sofort auf die Regler)
# --------------------------------------------------------------------------

@dataclass
class Region:
    id: int
    x0: int
    y0: int
    x1: int          # exklusiv
    y1: int          # exklusiv
    cx: float
    cy: float
    area: int        # Pixel im Analysebild
    area_orig: int   # geschätzte Pixel im Originalbild
    score: float     # Spitzenwert
    mean_score: float
    reason: str      # "Farbe", "Pixelfolge" oder "Struktur"
    share: Tuple[float, float, float]   # Anteile Farbe / Folge / Struktur
    rgb: Tuple[int, int, int]


@dataclass
class Detection:
    regions: List[Region]
    label_image: np.ndarray       # int32, 0 = nichts, sonst Region-ID (Analysebild)
    score: np.ndarray             # kombinierte Karte
    threshold: float
    scale: float
    orig_size: Tuple[int, int]
    n_found: int = 0              # Regionen vor der Obergrenze
    n_repeated: int = 0           # als "wiederholt" aussortiert
    n_small: int = 0              # zu klein


REASONS = ("Farbe", "Pixelfolge", "Struktur")


def _fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    if max_area <= 0:
        return mask
    bg, n = ndi.label(~mask)
    if n == 0:
        return mask
    areas = np.bincount(bg.ravel(), minlength=n + 1)
    border = np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]]))
    fill = areas <= max_area
    fill[0] = False
    fill[border] = False
    return mask | fill[bg]


@dataclass
class _Seg:
    """Zwischenergebnis: zusammengefasste Treffer oberhalb einer Schwelle."""
    score: np.ndarray
    lab: np.ndarray            # Region-Nummern 1..n (0 = nichts)
    n: int
    n_small: int
    area: np.ndarray
    peak: np.ndarray
    mean: np.ndarray
    peak_pos: np.ndarray       # (n, 2) Zeile, Spalte des Spitzenwerts
    com: np.ndarray
    objs: list
    shares: np.ndarray         # (n, 3) mittlere z-Werte Farbe/Folge/Struktur im Kern
    rgbm: np.ndarray
    feat: np.ndarray           # Merkmale für den Wiederholungsfilter


def _segment(res: AnalysisResult, p: DetectParams, T: float, min_area: int) -> Optional[_Seg]:
    H, W = res.shape
    zc = np.maximum(res.z_color - DEADBAND, 0) * p.w_color
    zs = np.maximum(res.z_seq - DEADBAND, 0) * p.w_seq
    zt = np.maximum(res.z_struct - DEADBAND, 0) * p.w_struct
    score = np.sqrt(zc * zc + zs * zs + zt * zt).astype(np.float32)
    mask = score >= T
    if not mask.any():
        return _Seg(score, np.zeros((H, W), np.int32), 0, 0, *(None,) * 9)

    # Nahe Treffer verschmelzen (Schließen), kleine Löcher füllen
    if p.merge_px > 0:
        k = 2 * int(p.merge_px) + 1
        m8 = mask.astype(np.uint8)
        m8 = ndi.maximum_filter(m8, size=k, mode="constant")
        m8 = ndi.minimum_filter(m8, size=k, mode="nearest")
        mask = mask | m8.astype(bool)
    mask = _fill_small_holes(mask, p.hole_px)

    lab_img, n = ndi.label(mask, structure=np.ones((3, 3), int))
    areas = np.bincount(lab_img.ravel(), minlength=n + 1)
    keep = areas >= max(1, min_area)
    keep[0] = False
    n_small = int(((areas > 0) & ~keep).sum() - 1)
    if not keep.any():
        return _Seg(score, np.zeros((H, W), np.int32), 0, max(n_small, 0), *(None,) * 9)
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    lab_img = remap[lab_img]
    n = int(keep.sum())

    idx = np.arange(1, n + 1)
    area = np.bincount(lab_img.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndi.maximum(score, lab_img, idx))
    mean = np.asarray(ndi.mean(score, lab_img, idx))
    peak_pos = np.asarray(ndi.maximum_position(score, lab_img, idx)).reshape(n, 2)
    com = np.asarray(ndi.center_of_mass(np.ones((H, W), np.float32), lab_img, idx)).reshape(n, 2)
    objs = ndi.find_objects(lab_img)

    # Kern jeder Region: Pixel mit mindestens dem halben Spitzenwert. Der Kern beschreibt die
    # Auffälligkeit selbst; Hof und verschmolzene Nachbarschaft würden Farbe und Größe verfälschen.
    peak_px = np.concatenate([[0.0], peak]).astype(np.float32)[lab_img]
    core_lab = np.where((lab_img > 0) & (score >= 0.5 * peak_px), lab_img, 0)
    core_area = np.bincount(core_lab.ravel(), minlength=n + 1)[1:]
    core_objs = ndi.find_objects(core_lab, max_label=n)
    shares = np.stack([np.asarray(ndi.mean(z, core_lab, idx)) for z in (zc, zs, zt)], axis=1)
    rgbm = np.stack([np.asarray(ndi.mean(res.rgb[..., c], core_lab, idx)) for c in range(3)], axis=1)
    cw = np.array([o[1].stop - o[1].start for o in core_objs], float)
    ch = np.array([o[0].stop - o[0].start for o in core_objs], float)
    labm = rgb_to_lab(np.clip(rgbm, 0, 255).astype(np.uint8).reshape(n, 1, 3)).reshape(n, 3)
    feat = np.column_stack([
        np.log2(np.maximum(core_area, 1)),
        np.log2(cw / np.maximum(ch, 1)),
        labm[:, 0] / 18.0, labm[:, 1] / 18.0, labm[:, 2] / 18.0,
        (core_area / np.maximum(cw * ch, 1)) / 0.35,
    ])
    return _Seg(score, lab_img, n, max(n_small, 0), area, peak, mean, peak_pos, com, objs, shares, rgbm, feat)


def _repeat_map(res: AnalysisResult, p: DetectParams) -> Optional[np.ndarray]:
    """
    Pixelkarte (bool): liegt das Pixel in einer Region, die zu vielen gleichartigen gehört?

    Die Gruppen werden immer bei der höchsten Empfindlichkeit gezählt, nicht bei der gerade
    eingestellten. Sonst würde ein Fund verschwinden, sobald man den Regler weiter aufdreht
    und dadurch mehr ähnliche Funde entstehen. So gilt: höhere Empfindlichkeit zeigt nie
    weniger. Das Ergebnis wird je Analyse zwischengespeichert.
    """
    key = (p.w_color, p.w_seq, p.w_struct, p.merge_px, p.hole_px, p.repeat_group)
    cache = res.cache
    if key in cache:
        return cache[key]
    seg = _segment(res, p, threshold_from_sensitivity(100.0), min_area=1)
    out = None
    if seg.n > p.repeat_group:
        if cKDTree is not None:
            neigh = cKDTree(seg.feat).query_ball_point(seg.feat, r=REPEAT_TOL, p=np.inf, return_length=True)
        else:
            neigh = _count_within(seg.feat, REPEAT_TOL)
        flags = (np.asarray(neigh) - 1) >= p.repeat_group
        out = np.concatenate([[False], flags])[seg.lab]
    cache[key] = out
    return out


def detect(res: AnalysisResult, p: Optional[DetectParams] = None) -> Detection:
    p = p or DetectParams()
    H, W = res.shape
    T = threshold_from_sensitivity(p.sensitivity)
    seg = _segment(res, p, T, p.min_area)
    if seg.n == 0:
        return Detection([], np.zeros((H, W), np.int32), seg.score, T, res.scale, res.orig_size,
                         n_small=seg.n_small)
    n = seg.n

    repeated = np.zeros(n, bool)
    if p.repeat_filter:
        rmap = _repeat_map(res, p)
        if rmap is not None:
            repeated = rmap[seg.peak_pos[:, 0], seg.peak_pos[:, 1]]
    n_rep = int(repeated.sum())

    order = [i for i in np.argsort(-seg.peak) if not repeated[i]]
    n_found = len(order)
    order = order[:max(1, p.max_regions)]

    regions: List[Region] = []
    rank_of = np.zeros(n + 1, np.int32)
    s = res.scale
    for rank, i in enumerate(order, start=1):
        rank_of[i + 1] = rank
        sl = seg.objs[i]
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        sh = seg.shares[i]
        tot = float(sh.sum())
        shf = tuple(float(v) for v in (sh / tot if tot > 0 else sh))
        regions.append(Region(
            id=rank,
            x0=int(math.floor(x0 * s)), y0=int(math.floor(y0 * s)),
            x1=min(res.orig_size[0], int(math.ceil(x1 * s))), y1=min(res.orig_size[1], int(math.ceil(y1 * s))),
            cx=float((seg.com[i, 1] + 0.5) * s), cy=float((seg.com[i, 0] + 0.5) * s),
            area=int(seg.area[i]), area_orig=int(round(seg.area[i] * s * s)),
            score=float(seg.peak[i]), mean_score=float(seg.mean[i]),
            reason=REASONS[int(np.argmax(sh))] if tot > 0 else REASONS[0],
            share=shf,
            rgb=tuple(int(round(v)) for v in seg.rgbm[i]),
        ))
    return Detection(regions=regions, label_image=rank_of[seg.lab], score=seg.score, threshold=T,
                     scale=res.scale, orig_size=res.orig_size,
                     n_found=n_found, n_repeated=n_rep, n_small=seg.n_small)


# --------------------------------------------------------------------------
# Darstellung (für Export und Oberfläche)
# --------------------------------------------------------------------------

_HEAT_STOPS = [(0.0, (0, 0, 0)), (0.2, (40, 10, 80)), (0.4, (160, 30, 90)),
               (0.6, (240, 110, 20)), (0.8, (255, 230, 90)), (1.0, (255, 255, 255))]


def _heat_lut() -> np.ndarray:
    xs = np.array([s[0] for s in _HEAT_STOPS])
    lut = np.zeros((256, 3), np.uint8)
    t = np.linspace(0, 1, 256)
    for c in range(3):
        ys = np.array([s[1][c] for s in _HEAT_STOPS], float)
        lut[:, c] = np.interp(t, xs, ys).astype(np.uint8)
    return lut


_HEAT = _heat_lut()


def render_heatmap(det: Detection) -> Image.Image:
    """Kombinierte Karte; die Schwelle liegt bei 50 % Helligkeit (orange)."""
    t = np.clip(det.score / (2.0 * det.threshold), 0, 1)
    idx = (t * 255).astype(np.uint8)
    return Image.fromarray(_HEAT[idx], "RGB")


def render_mask(det: Detection) -> Image.Image:
    return Image.fromarray(((det.label_image > 0) * 255).astype(np.uint8), "L").convert("RGB")


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_overlay(original: Image.Image, det: Detection, numbers: bool = True,
                   pad: int = 4) -> Image.Image:
    """Original mit Rahmen und Nummern um jede Region."""
    img = original.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    lw = max(2, round(max(img.size) / 500))
    font = _font(max(12, round(max(img.size) / 60)))
    for r in det.regions:
        box = (max(0, r.x0 - pad), max(0, r.y0 - pad), min(img.width - 1, r.x1 + pad), min(img.height - 1, r.y1 + pad))
        d.rectangle(box, outline=(255, 40, 40), width=lw)
        if numbers:
            txt = str(r.id)
            tx, ty = box[0], max(0, box[1] - font.size - 2) if hasattr(font, "size") else box[1]
            try:
                tb = d.textbbox((tx, ty), txt, font=font)
                d.rectangle(tb, fill=(255, 40, 40))
            except Exception:
                pass
            d.text((tx, ty), txt, fill=(255, 255, 255), font=font)
    return img


def regions_to_csv(det: Detection, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Nr", "x0", "y0", "x1", "y1", "Mitte_x", "Mitte_y", "Flaeche_px",
                    "Score", "Grund", "Anteil_Farbe", "Anteil_Folge", "Anteil_Struktur", "R", "G", "B"])
        for r in det.regions:
            w.writerow([r.id, r.x0, r.y0, r.x1, r.y1, round(r.cx, 1), round(r.cy, 1), r.area_orig,
                        round(r.score, 2), r.reason, *(round(s, 3) for s in r.share), *r.rgb])


# --------------------------------------------------------------------------
# Kommandozeile
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Findet auffällige Pixel und Pixelfolgen in einem Foto.")
    ap.add_argument("bild")
    ap.add_argument("--sens", type=float, default=50, help="Empfindlichkeit 0..100 (Standard 50)")
    ap.add_argument("--out", help="Ergebnisbild (PNG) mit Markierungen")
    ap.add_argument("--heat", help="Heatmap als PNG speichern")
    ap.add_argument("--csv", help="Funde als CSV speichern")
    ap.add_argument("--max-side", type=int, default=1200)
    ap.add_argument("--no-struct", action="store_true", help="Strukturdetektor ausschalten (schneller)")
    ap.add_argument("--no-repeat", action="store_true", help="Wiederholungsfilter ausschalten")
    ap.add_argument("--min-area", type=int, default=4)
    args = ap.parse_args(argv)

    img = load_image(args.bild)
    ap_ = AnalysisParams(max_side=args.max_side, use_structure=not args.no_struct)
    print(f"Analysiere {args.bild} ({img.width}x{img.height}) ...")
    res = analyze(img, ap_, progress=lambda f, t: None)
    det = detect(res, DetectParams(sensitivity=args.sens, min_area=args.min_area,
                                   repeat_filter=not args.no_repeat))
    t = res.timings
    print(f"Analyse in {t['gesamt']:.1f} s (Farbe {t['farbe']:.1f}, Folgen {t['folgen']:.1f}, Struktur {t['struktur']:.1f})")
    print(f"Schwelle {det.threshold:.1f}  |  {len(det.regions)} Funde"
          f"  (aussortiert: {det.n_repeated} wiederholt, {det.n_small} zu klein)")
    for r in det.regions[:30]:
        print(f"  #{r.id:<3} x={r.x0}-{r.x1} y={r.y0}-{r.y1}  Fläche {r.area_orig:>6} px  "
              f"Score {r.score:6.1f}  {r.reason}")
    if args.out:
        render_overlay(img, det).save(args.out)
        print("Gespeichert:", args.out)
    if args.heat:
        render_heatmap(det).save(args.heat)
        print("Gespeichert:", args.heat)
    if args.csv:
        regions_to_csv(det, args.csv)
        print("Gespeichert:", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
