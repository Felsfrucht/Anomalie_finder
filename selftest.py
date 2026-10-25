"""
selftest.py - prüft die Engine an Testbildern mit bekannter Lösung.

    python selftest.py

Gibt für jeden Test OK oder FEHLER aus; Rückgabecode 0 = alles bestanden.
"""

from __future__ import annotations

import sys
import time

import numpy as np
from PIL import Image

import testbilder
from analyzer import (AnalysisParams, Cancelled, DetectParams, analyze, detect,
                      threshold_from_sensitivity)

results = []


def check(name: str, ok: bool, info: str = ""):
    results.append(ok)
    print(f"  [{'OK    ' if ok else 'FEHLER'}] {name}" + (f"  ({info})" if info else ""))


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


def overlaps(r, t):
    return min(r.x1, t[2]) > max(r.x0, t[0]) and min(r.y1, t[3]) > max(r.y0, t[1])


def box(r):
    return (r.x0, r.y0, r.x1, r.y1)


def run():
    cache = {}

    def get(name):
        if name not in cache:
            img, truth = testbilder.ALLE[name]()
            t = time.perf_counter()
            res = analyze(img)
            cache[name] = (img, truth, res, time.perf_counter() - t)
        return cache[name]

    # -------- Möwe im Himmel --------------------------------------------------
    for name in ("moewe", "moewe_wolken"):
        print(f"\n{name}: weiße Möwe in der linken oberen Ecke eines blauen Himmels")
        img, truth, res, dt = get(name)
        det = detect(res)
        top = det.regions[0] if det.regions else None
        check("Möwe ist der Fund mit dem höchsten Score", top is not None and iou(box(top), truth[0]) > 0.4,
              f"IoU {iou(box(top), truth[0]):.2f}" if top else "kein Fund")
        check("höchstens 2 Funde insgesamt (keine Fehlalarme)", len(det.regions) <= 2, f"{len(det.regions)} Funde")

    # -------- Sternenhimmel -------------------------------------------------
    print("\nsterne: ~1400 gleichartige Sterne, dazu ein Planet")
    img, truth, res, dt = get("sterne")
    det = detect(res)
    check("Planet ist der Fund mit dem höchsten Score",
          bool(det.regions) and overlaps(det.regions[0], truth[0]),
          f"{len(det.regions)} Funde")
    check("höchstens 3 Funde insgesamt (Sterne gelten als normal)", len(det.regions) <= 3)
    check("Wiederholungsfilter sortiert viele Sterne aus", det.n_repeated > 100, f"{det.n_repeated} aussortiert")
    lost = [s for s in (30, 40, 50, 60, 70, 80, 90, 95)
            if not any(overlaps(r, truth[0]) for r in detect(res, DetectParams(sensitivity=s)).regions)]
    check("Planet geht bei keiner Empfindlichkeit (30-95) verloren", not lost, f"verloren bei {lost}" if lost else "")
    det_off = detect(res, DetectParams(repeat_filter=False))
    check("ohne Wiederholungsfilter erscheinen deutlich mehr Funde", len(det_off.regions) > 20 * max(1, len(det.regions)),
          f"{len(det_off.regions)} statt {len(det.regions)}")

    # -------- Pixelfolgen -----------------------------------------------------
    print("\nmuster: Streifen R,G,B,R,G,B ... - jede Farbe ist häufig, nur die Reihenfolge bricht")
    img, truth, res, dt = get("muster")
    det = detect(res)
    for i, t in enumerate(truth):
        check(f"Fehlerstelle {i + 1} gefunden", any(overlaps(r, t) for r in det.regions))
    check("höchstens 5 Funde insgesamt", len(det.regions) <= 5, f"{len(det.regions)} Funde")
    # Nur Pixelfolgen (Farbe und Struktur ausgeschaltet); Mindestgröße 1, damit auch das
    # einzelne falsche Pixel zählt.
    solo = dict(sensitivity=70, min_area=1)
    only_seq = detect(res, DetectParams(w_color=0.0, w_struct=0.0, **solo))
    hit = sum(any(overlaps(r, t) for r in only_seq.regions) for t in truth)
    check("der Pixelfolgen-Detektor ALLEIN findet alle 3 Stellen (Empfindlichkeit 70)", hit == 3, f"{hit} von 3")
    only_col = detect(res, DetectParams(w_seq=0.0, w_struct=0.0, **solo))
    hit_c = sum(any(overlaps(r, t) for r in only_col.regions) for t in truth)
    check("der Farb-Detektor allein findet hier weniger", hit_c < hit,
          f"Farbe {hit_c}, Folgen {hit}")
    noise_res = get("rauschen")[2]
    check("Pixelfolgen allein melden im Rauschbild nichts (gleiche Einstellung)",
          len(detect(noise_res, DetectParams(w_color=0.0, w_struct=0.0, **solo)).regions) == 0)

    # -------- Rauschen -------------------------------------------------------
    print("\nrauschen: nur Verlauf und Sensorrauschen, keine Anomalie")
    img, truth, res, dt = get("rauschen")
    det = detect(res)
    check("keine Funde bei Standard-Empfindlichkeit", len(det.regions) == 0, f"{len(det.regions)} Funde")

    # -------- natürliche Textur ----------------------------------------------
    print("\ngelaende: natürliche Textur als JPEG, dazu ein rotes Objekt")
    img, truth, res, dt = get("gelaende")
    det = detect(res)
    check("rotes Objekt ist der Fund mit dem höchsten Score",
          bool(det.regions) and iou(box(det.regions[0]), truth[0]) > 0.4)
    check("höchstens 3 Funde insgesamt", len(det.regions) <= 3, f"{len(det.regions)} Funde")

    # -------- Regler ----------------------------------------------------------
    print("\nEmpfindlichkeit")
    res = get("gelaende")[2]
    counts = []
    for s in (10, 30, 50, 70, 90):
        d = detect(res, DetectParams(sensitivity=s))
        counts.append(int((d.score >= d.threshold).sum()))
    check("höhere Empfindlichkeit markiert nie weniger Pixel", counts == sorted(counts), str(counts))
    check("Schwelle sinkt mit steigender Empfindlichkeit",
          threshold_from_sensitivity(0) > threshold_from_sensitivity(50) > threshold_from_sensitivity(100))
    t = time.perf_counter()
    for _ in range(3):
        detect(res, DetectParams(sensitivity=60))
    per = (time.perf_counter() - t) / 3
    check("Regler reagiert schnell (Neuberechnung der Funde < 0,5 s)", per < 0.5, f"{per * 1000:.0f} ms")

    # -------- Koordinaten bei verkleinerter Analyse ---------------------------
    print("\nGroße Bilder")
    img, truth, _, _ = get("moewe")
    big = img.resize((img.width * 2, img.height * 2), Image.BICUBIC)
    res_big = analyze(big, AnalysisParams(max_side=1200))
    det = detect(res_big)
    t2 = tuple(2 * v for v in truth[0])
    check("Analysebild wird verkleinert", res_big.shape[1] == 1200 and abs(res_big.scale - 2.0) < 1e-6)
    check("Rahmen liegen in Koordinaten des Originalbilds",
          bool(det.regions) and iou(box(det.regions[0]), t2) > 0.4,
          f"IoU {iou(box(det.regions[0]), t2):.2f}" if det.regions else "kein Fund")

    # -------- Abbruch ---------------------------------------------------------
    print("\nAbbruch")
    try:
        analyze(img, cancel=lambda: True)
        check("Analyse lässt sich abbrechen", False)
    except Cancelled:
        check("Analyse lässt sich abbrechen", True)

    # -------- Zeiten ---------------------------------------------------------
    print("\nRechenzeiten der Analyse:")
    for name in ("moewe", "sterne", "muster", "rauschen", "gelaende"):
        if name in cache:
            print(f"  {name:10s} {cache[name][2].shape[1]}x{cache[name][2].shape[0]}  {cache[name][3]:.1f} s")

    ok = all(results)
    print(f"\n{sum(results)} von {len(results)} Prüfungen bestanden." + ("" if ok else "  -> FEHLER"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
