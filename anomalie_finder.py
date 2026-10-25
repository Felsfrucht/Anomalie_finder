#!/usr/bin/env python3
"""
anomalie_finder.py - Oberfläche (Tkinter) für den Anomalie-Finder
=================================================================

Start:
    python anomalie_finder.py              # leeres Fenster, Foto über Strg+O öffnen
    python anomalie_finder.py foto.jpg     # Foto direkt öffnen

Die schwere Analyse läuft einmal pro Foto im Hintergrund. Alles, was die
Empfindlichkeit, die Gewichte und die Regionen betrifft, wirkt sofort.
Einstellungen im Reiter "Analyse" brauchen "Neu analysieren".
"""

from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import analyzer as az

APP_TITLE = "Anomalie-Finder"
CANVAS_BG = "#1e1f22"
COL_BOX = "#ff3b30"
COL_SEL = "#ffd400"
MAX_BOXES_DRAWN = 400

HELP_TEXT = """\
WAS DIE APP MACHT
Sie lernt aus dem Foto selbst, was "normal" ist, und markiert alles, was dazu nicht passt:
eine Möwe im blauen Himmel, einen Planeten zwischen lauter gleichartigen Sternen oder eine
Stelle, an der die gewohnte Abfolge der Pixel bricht.

DREI DETEKTOREN
• Farbe: Die Farbe des Pixels ist im ganzen Bild selten.
• Pixelfolge: Das Pixel passt nicht zu seinen Nachbarn. Wenn neben Farbe A sonst immer
  Farbe B steht (waagrecht, senkrecht oder diagonal, bis zu einigen Pixeln Abstand) und hier
  nicht, schlägt das an, auch wenn jede Farbe für sich häufig ist.
• Struktur: Der 5x5-Bildausschnitt um das Pixel ist selten (in zwei Maßstäben).
In der Liste steht, welcher Detektor bei einem Fund am stärksten beteiligt war.

EMPFINDLICHKEIT
Regler nach rechts: Die App schlägt schneller an und zeigt auch kleinere Abweichungen.
Nach links: Nur Offensichtliches. Der Regler wirkt sofort, ohne neue Berechnung. Wenn
Rauschen markiert wird, den Regler etwas zurücknehmen.

WIEDERHOLUNGSFILTER
Viele gleichartige Funde (z. B. Tausende Sterne) gelten als normal und werden ausgeblendet.
Etwas Einzelnes, das heraussticht, bleibt. Mit "Gruppengröße" legst du fest, ab wie vielen
ähnlichen Funden etwas als normal gilt.

EINZELNE PIXEL
Ein einzelner abweichender Pixel ist standardmäßig zu klein (Mindestgröße 4). Für solche Fälle
Mindestgröße im Reiter "Regionen" auf 1 stellen.

BEDIENUNG
Mausrad = Zoomen, Ziehen = Verschieben, Klick auf Rahmen = Fund auswählen,
Doppelklick in der Liste = zum Fund zoomen, Leertaste = Original/Markierung umschalten,
0 = Bild einpassen, 1 = 100 %, Strg+O = Foto öffnen, Strg+S = Ergebnis speichern, F5 = neu analysieren.

GRENZEN
Das Bild definiert "normal" - ein Objekt, das mehr als einige Prozent der Fläche einnimmt,
ist selbst Teil des Normalen. Sehr strukturierte Fotos (Laub, Menschenmengen) liefern bei hoher
Empfindlichkeit viele Treffer. Große Bilder werden für die Analyse verkleinert (Reiter
"Analyse"); die Rahmen gelten trotzdem im Original.
"""


class ToolTip:
    """Kleiner Hilfetext beim Darüberfahren."""

    def __init__(self, widget, text: str, delay: int = 450):
        self.widget, self.text, self.delay = widget, text, delay
        self.tip = None
        self._job = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _e=None):
        self._hide()
        self._job = self.widget.after(self.delay, self._show)

    def _show(self):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 14
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, justify="left", background="#ffffe1", relief="solid",
                 borderwidth=1, wraplength=340, padx=6, pady=4).pack()

    def _hide(self, _e=None):
        if self._job:
            try:
                self.widget.after_cancel(self._job)
            except Exception:
                pass
            self._job = None
        if self.tip:
            self.tip.destroy()
            self.tip = None


class App(tk.Tk):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x840")
        self.minsize(1050, 620)

        # ---- Zustand ----
        self.original: Image.Image | None = None
        self.path: str | None = None
        self.result: az.AnalysisResult | None = None
        self.detection: az.Detection | None = None
        self.zoom = 1.0
        self.ox = 0.0           # Bildkoordinate (Original) der linken oberen Canvas-Ecke
        self.oy = 0.0
        self.fit_mode = True
        self.photo = None       # Referenz auf ImageTk.PhotoImage
        self.selected_id: int | None = None
        self._view_cache: dict = {}
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = False
        self._detect_job = None
        self._detect_ms = 100.0
        self._analysis_dirty = False
        self._press = None
        self._redraw_job = None

        self._init_vars()
        self._init_style()
        self._build_menu()
        self._build_ui()
        self._bind_keys()
        self._draw_placeholder()
        self.after(60, self._poll)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        if path:
            self.after(250, lambda: self.open_path(path))

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------
    def _init_vars(self):
        d, a = az.DetectParams(), az.AnalysisParams()
        self.v_sens = tk.DoubleVar(value=d.sensitivity)
        self.v_w_color = tk.DoubleVar(value=d.w_color * 100)
        self.v_w_seq = tk.DoubleVar(value=d.w_seq * 100)
        self.v_w_struct = tk.DoubleVar(value=d.w_struct * 100)
        self.v_min_area = tk.IntVar(value=d.min_area)
        self.v_merge = tk.IntVar(value=d.merge_px)
        self.v_hole = tk.IntVar(value=d.hole_px)
        self.v_repeat = tk.BooleanVar(value=d.repeat_filter)
        self.v_group = tk.IntVar(value=d.repeat_group)
        self.v_max_regions = tk.IntVar(value=d.max_regions)
        self.v_max_side = tk.IntVar(value=a.max_side)
        self.v_color_bin = tk.DoubleVar(value=a.color_bin)
        self.v_smooth = tk.DoubleVar(value=a.color_smooth)
        self.v_reach = tk.IntVar(value=a.seq_reach)
        self.v_grids = tk.BooleanVar(value=a.seq_grids >= 2)
        self.v_struct_on = tk.BooleanVar(value=a.use_structure)
        self.v_view = tk.StringVar(value="overlay")
        self.v_boxes = tk.BooleanVar(value=True)
        self.v_numbers = tk.BooleanVar(value=True)
        self.v_status = tk.StringVar(value="Bereit. Foto öffnen mit Strg+O.")
        self.v_hover = tk.StringVar(value="")
        self.v_info = tk.StringVar(value="")
        self.v_details = tk.StringVar(value="")
        self.v_file = tk.StringVar(value="kein Foto geladen")
        self.v_sens_text = tk.StringVar(value=f"{int(d.sensitivity)}")

    def _init_style(self):
        st = ttk.Style(self)
        try:
            st.theme_use("vista" if sys.platform.startswith("win") else "clam")
        except tk.TclError:
            pass
        st.configure("Treeview", rowheight=22)
        st.configure("Big.TLabel", font=("TkDefaultFont", 18, "bold"))
        st.configure("Hint.TLabel", foreground="#666666")
        st.configure("Warn.TLabel", foreground="#b35c00")

    def _build_menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Foto öffnen ...", accelerator="Strg+O", command=self.open_dialog)
        f.add_command(label="Neu analysieren", accelerator="F5", command=self.start_analysis)
        f.add_separator()
        f.add_command(label="Ergebnisbild speichern ...", accelerator="Strg+S", command=self.save_result)
        f.add_command(label="Heatmap speichern ...", command=self.save_heatmap)
        f.add_command(label="Funde als CSV speichern ...", command=self.save_csv)
        f.add_separator()
        f.add_command(label="Beenden", command=self._on_close)
        m.add_cascade(label="Datei", menu=f)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Bedienung und Erklärung", command=self.show_help)
        m.add_cascade(label="Hilfe", menu=h)
        self.config(menu=m)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # ---- Werkzeugleiste ----
        bar = ttk.Frame(self, padding=(8, 6))
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.btn_open = ttk.Button(bar, text="Foto öffnen ...", command=self.open_dialog)
        self.btn_open.pack(side="left")
        self.btn_run = ttk.Button(bar, text="Neu analysieren", command=self.start_analysis, state="disabled")
        self.btn_run.pack(side="left", padx=(6, 0))
        self.btn_cancel = ttk.Button(bar, text="Abbrechen", command=self.cancel_analysis, state="disabled")
        self.btn_cancel.pack(side="left", padx=(6, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.btn_save = ttk.Button(bar, text="Ergebnis speichern ...", command=self.save_result, state="disabled")
        self.btn_save.pack(side="left")
        self.btn_csv = ttk.Button(bar, text="Funde als CSV ...", command=self.save_csv, state="disabled")
        self.btn_csv.pack(side="left", padx=(6, 0))
        ttk.Label(bar, textvariable=self.v_file, style="Hint.TLabel").pack(side="right")

        # ---- linke Spalte ----
        left = ttk.Frame(self, padding=(8, 2, 4, 6), width=330)
        left.grid(row=1, column=0, sticky="ns")
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)

        sens = ttk.LabelFrame(left, text="Empfindlichkeit", padding=8)
        sens.grid(row=0, column=0, sticky="ew")
        top = ttk.Frame(sens)
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.v_sens_text, style="Big.TLabel").pack(side="left")
        ttk.Label(top, text="  von 100", style="Hint.TLabel").pack(side="left", pady=(6, 0))
        self.scale_sens = ttk.Scale(sens, from_=0, to=100, variable=self.v_sens, command=self._on_sens)
        self.scale_sens.pack(fill="x", pady=(4, 0))
        row = ttk.Frame(sens)
        row.pack(fill="x")
        ttk.Label(row, text="nur Offensichtliches", style="Hint.TLabel").pack(side="left")
        ttk.Label(row, text="schlägt schnell an", style="Hint.TLabel").pack(side="right")
        # feste Höhe, damit das Panel beim Ziehen des Reglers nicht springt
        info_box = ttk.Frame(sens, height=58)
        info_box.pack(fill="x", pady=(8, 0))
        info_box.pack_propagate(False)
        ttk.Label(info_box, textvariable=self.v_info, wraplength=290, justify="left").pack(anchor="nw")
        ToolTip(self.scale_sens, "Wie schnell die App auf nicht normale Pixel und Pixelfolgen anschlägt.\n"
                                 "Rechts: schon kleine Abweichungen werden markiert (mehr Funde, auch Rauschen).\n"
                                 "Links: nur deutliche Ausreißer. Wirkt sofort.")

        nb = ttk.Notebook(left)
        nb.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        left.rowconfigure(1, weight=1)

        # -- Reiter Detektoren --
        t1 = ttk.Frame(nb, padding=8)
        nb.add(t1, text="Detektoren")
        ttk.Label(t1, text="Wie stark zählt jede Art von Auffälligkeit?", wraplength=290).pack(anchor="w")
        self._weight_row(t1, "Farbe selten", self.v_w_color,
                         "Die Farbe des Pixels kommt im ganzen Bild kaum vor (z. B. weiße Möwe im blauen Himmel).")
        self._weight_row(t1, "Pixelfolge passt nicht", self.v_w_seq,
                         "Das Pixel passt nicht zu seinen Nachbarn: Wenn neben Farbe A sonst immer Farbe B steht "
                         "(in 4 Richtungen, bis zu einigen Pixeln Abstand) und hier nicht.\n"
                         "Findet auch Abweichungen, bei denen jede Farbe für sich häufig ist.")
        self._weight_row(t1, "Struktur selten", self.v_w_struct,
                         "Der 5x5-Bildausschnitt um das Pixel hat im Bild kaum Ähnliche (in zwei Maßstäben).")
        ttk.Label(t1, text="0 % schaltet einen Detektor aus. Wirkt sofort.", style="Hint.TLabel",
                  wraplength=290).pack(anchor="w", pady=(10, 0))

        # -- Reiter Regionen --
        t2 = ttk.Frame(nb, padding=8)
        nb.add(t2, text="Regionen")
        self._spin_row(t2, "Mindestgröße (Pixel)", self.v_min_area, 1, 5000, 1,
                       "Kleinere Funde werden ignoriert. Für einzelne abweichende Pixel auf 1 stellen "
                       "(bei hoher Empfindlichkeit dann evtl. mehr Rauschtreffer).")
        self._spin_row(t2, "Zusammenfassen (Pixel)", self.v_merge, 0, 30, 1,
                       "Treffer, die weniger weit auseinander liegen, werden zu einer Region verschmolzen.")
        self._spin_row(t2, "Löcher füllen bis (Pixel)", self.v_hole, 0, 5000, 10,
                       "Kleine Lücken im Inneren einer Region werden geschlossen.")
        ttk.Separator(t2).pack(fill="x", pady=8)
        cb = ttk.Checkbutton(t2, text="Wiederholtes ausblenden", variable=self.v_repeat, command=self._schedule_detect)
        cb.pack(anchor="w")
        ToolTip(cb, "Viele gleichartige Funde (ähnliche Größe, Form und Farbe) gelten als normal, z. B. "
                    "Tausende Sterne. Einzelnes, das heraussticht, bleibt.")
        self._spin_row(t2, "ab Gruppengröße", self.v_group, 2, 500, 1,
                       "Ab so vielen ähnlichen Funden gilt eine Gruppe als normal.")
        self._spin_row(t2, "Höchstens Funde zeigen", self.v_max_regions, 10, 5000, 10, "Obergrenze für die Liste.")

        # -- Reiter Analyse --
        t3 = ttk.Frame(nb, padding=8)
        nb.add(t3, text="Analyse")
        ttk.Label(t3, text="Diese Einstellungen brauchen eine neue Berechnung:", wraplength=290).pack(anchor="w")
        self._spin_row(t3, "Max. Analysegröße (px)", self.v_max_side, 300, 6000, 100,
                       "Längste Bildkante der Analyse. Größere Fotos werden dafür verkleinert (schneller). "
                       "Die Rahmen gelten trotzdem im Originalbild. Für kleinste Details höher stellen.",
                       analysis=True)
        self._spin_row(t3, "Farbgenauigkeit (kleiner = feiner)", self.v_color_bin, 2, 20, 1,
                       "Breite der Farbklassen für die Farbstatistik (Lab-Einheiten).", analysis=True)
        self._spin_row(t3, "Rauschglättung (px)", self.v_smooth, 0, 3, 0.1,
                       "Leichte Glättung nur für die Farbstatistik, damit Sensorrauschen nicht auffällt. 0 = aus.",
                       analysis=True)
        self._spin_row(t3, "Pixelfolgen: Reichweite", self.v_reach, 1, 4, 1,
                       "Bis zu wie viele Pixel Abstand Nachbarbeziehungen geprüft werden.", analysis=True)
        c1 = ttk.Checkbutton(t3, text="Pixelfolgen: Doppelraster (genauer)", variable=self.v_grids,
                             command=self._mark_dirty)
        c1.pack(anchor="w", pady=(6, 0))
        ToolTip(c1, "Rechnet die Farbklassen auf zwei gegeneinander verschobenen Rastern und nimmt pro "
                    "Pixel den unauffälligeren Wert. Das unterdrückt Fehlalarme durch Rauschen. Dauert etwas länger.")
        c2 = ttk.Checkbutton(t3, text="Strukturdetektor verwenden", variable=self.v_struct_on, command=self._mark_dirty)
        c2.pack(anchor="w")
        ToolTip(c2, "Der langsamste Teil der Analyse. Ausschalten macht sie deutlich schneller.")
        self.lbl_dirty = ttk.Label(t3, text="", style="Warn.TLabel", wraplength=290)
        self.lbl_dirty.pack(anchor="w", pady=(10, 0))
        ttk.Button(t3, text="Neu analysieren (F5)", command=self.start_analysis).pack(anchor="w", pady=(6, 0))

        # -- Reiter Ansicht --
        t4 = ttk.Frame(nb, padding=8)
        nb.add(t4, text="Ansicht")
        for text, val in (("Original mit Markierung", "overlay"), ("Heatmap (Orange = Schwelle)", "heat"),
                          ("Maske der Funde", "mask"), ("Nur Original", "original")):
            ttk.Radiobutton(t4, text=text, value=val, variable=self.v_view, command=self._redraw).pack(anchor="w", pady=1)
        ttk.Separator(t4).pack(fill="x", pady=8)
        ttk.Checkbutton(t4, text="Rahmen zeigen", variable=self.v_boxes, command=self._redraw).pack(anchor="w")
        ttk.Checkbutton(t4, text="Nummern zeigen", variable=self.v_numbers, command=self._redraw).pack(anchor="w")
        ttk.Separator(t4).pack(fill="x", pady=8)
        b = ttk.Frame(t4)
        b.pack(anchor="w")
        ttk.Button(b, text="Einpassen (0)", command=self.fit).pack(side="left")
        ttk.Button(b, text="100 % (1)", command=self.zoom_100).pack(side="left", padx=(6, 0))
        ttk.Label(t4, text="Leertaste schaltet zwischen Original und Markierung um.", style="Hint.TLabel",
                  wraplength=290).pack(anchor="w", pady=(10, 0))

        # ---- Mitte: Bild ----
        mid = ttk.Frame(self)
        mid.grid(row=1, column=1, sticky="nsew", padx=4, pady=(2, 6))
        mid.rowconfigure(0, weight=1)
        mid.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(mid, background=CANVAS_BG, highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # ---- rechts: Funde ----
        right = ttk.Frame(self, padding=(4, 2, 8, 6), width=374)
        right.grid(row=1, column=2, sticky="ns")
        right.grid_propagate(False)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="Funde").grid(row=0, column=0, sticky="w")
        cols = ("nr", "pos", "gr", "sc", "gd")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
        for c, t, w, anchor in (("nr", "#", 32, "e"), ("pos", "Position", 92, "w"), ("gr", "Größe", 72, "w"),
                                ("sc", "Score", 50, "e"), ("gd", "Grund", 92, "w")):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=anchor, stretch=(c == "gd"))
        sb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        sb.grid(row=1, column=1, sticky="ns")
        ttk.Label(right, textvariable=self.v_details, wraplength=350, justify="left").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(right, text="Grund: Farbe = seltene Farbe · Pixelfolge = passt nicht zu den Nachbarpixeln · "
                              "Struktur = seltener Bildausschnitt", style="Hint.TLabel", wraplength=350,
                  justify="left").grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_double)

        # ---- Statusleiste ----
        st = ttk.Frame(self, padding=(8, 2))
        st.grid(row=2, column=0, columnspan=3, sticky="ew")
        st.columnconfigure(0, weight=1)
        ttk.Label(st, textvariable=self.v_status).grid(row=0, column=0, sticky="w")
        ttk.Label(st, textvariable=self.v_hover, style="Hint.TLabel").grid(row=0, column=1, sticky="e", padx=10)
        self.progress = ttk.Progressbar(st, length=200, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=2, sticky="e")

        # Canvas-Ereignisse
        c = self.canvas
        c.bind("<Configure>", self._on_resize)
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", lambda e: self.v_hover.set(""))
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1.25))
        c.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, 1 / 1.25))
        c.bind("<Enter>", lambda e: c.focus_set())

    def _weight_row(self, parent, text, var, tip):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(10, 0))
        lab = ttk.Label(f, text=text)
        lab.pack(side="left")
        val = ttk.Label(f, width=5, anchor="e")
        val.pack(side="right")
        sc = ttk.Scale(parent, from_=0, to=100, variable=var, command=lambda _v: self._on_weight(var, val))
        sc.pack(fill="x")
        val.configure(text=f"{int(var.get())} %")
        ToolTip(lab, tip)
        ToolTip(sc, tip)

    def _on_weight(self, var, val_label):
        val_label.configure(text=f"{int(round(var.get()))} %")
        self._schedule_detect()

    def _spin_row(self, parent, text, var, lo, hi, inc, tip, analysis=False):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(6, 0))
        lab = ttk.Label(f, text=text, wraplength=190, justify="left")
        lab.pack(side="left")
        sp = ttk.Spinbox(f, from_=lo, to=hi, increment=inc, textvariable=var, width=7, justify="right")
        sp.pack(side="right")
        var.trace_add("write", lambda *_: (self._mark_dirty() if analysis else self._schedule_detect()))
        ToolTip(lab, tip)
        ToolTip(sp, tip)

    def _bind_keys(self):
        self.bind_all("<Control-o>", lambda e: self.open_dialog())
        self.bind_all("<Control-s>", lambda e: self.save_result())
        self.bind_all("<F5>", lambda e: self.start_analysis())
        c = self.canvas
        c.bind("<space>", lambda e: self._toggle_original())
        c.bind("<Key-0>", lambda e: self.fit())
        c.bind("<Key-1>", lambda e: self.zoom_100())
        for k in ("<plus>", "<KP_Add>", "<equal>"):
            c.bind(k, lambda e: self._zoom_at(*self._canvas_center(), 1.25))
        for k in ("<minus>", "<KP_Subtract>"):
            c.bind(k, lambda e: self._zoom_at(*self._canvas_center(), 1 / 1.25))

    # ------------------------------------------------------------------
    # Datei / Analyse
    # ------------------------------------------------------------------
    def open_dialog(self):
        if self._busy():
            return
        path = filedialog.askopenfilename(
            title="Foto öffnen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.gif"), ("Alle Dateien", "*.*")])
        if path:
            self.open_path(path)

    def open_path(self, path: str):
        if self._busy():
            return
        try:
            img = az.load_image(path)
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Das Foto konnte nicht geöffnet werden:\n{path}\n\n{e}")
            return
        self.original = img
        self.path = path
        self.result = None
        self.detection = None
        self.selected_id = None
        self._view_cache.clear()
        self._clear_list()
        self.v_info.set("")
        self.v_details.set("")
        self.v_file.set(f"{os.path.basename(path)}  ({img.width} x {img.height})")
        self.title(f"{APP_TITLE} - {os.path.basename(path)}")
        self.btn_run.state(["!disabled"])
        self.btn_save.state(["disabled"])
        self.btn_csv.state(["disabled"])
        self.fit_mode = True
        self.update_idletasks()
        self.fit()
        self.start_analysis()

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _analysis_params(self) -> az.AnalysisParams:
        a = az.AnalysisParams()
        return az.AnalysisParams(
            max_side=int(self._num(self.v_max_side, a.max_side, 300, 6000)),
            color_bin=float(self._num(self.v_color_bin, a.color_bin, 2, 20)),
            color_smooth=float(self._num(self.v_smooth, a.color_smooth, 0, 3)),
            seq_reach=int(self._num(self.v_reach, a.seq_reach, 1, 4)),
            seq_grids=2 if self.v_grids.get() else 1,
            use_structure=bool(self.v_struct_on.get()),
        )

    def _detect_params(self) -> az.DetectParams:
        d = az.DetectParams()
        return az.DetectParams(
            sensitivity=float(self._num(self.v_sens, d.sensitivity, 0, 100)),
            w_color=float(self._num(self.v_w_color, 100, 0, 100)) / 100.0,
            w_seq=float(self._num(self.v_w_seq, 100, 0, 100)) / 100.0,
            w_struct=float(self._num(self.v_w_struct, 100, 0, 100)) / 100.0,
            min_area=int(self._num(self.v_min_area, d.min_area, 1, 100000)),
            merge_px=int(self._num(self.v_merge, d.merge_px, 0, 30)),
            hole_px=int(self._num(self.v_hole, d.hole_px, 0, 100000)),
            repeat_filter=bool(self.v_repeat.get()),
            repeat_group=int(self._num(self.v_group, d.repeat_group, 2, 1000)),
            max_regions=int(self._num(self.v_max_regions, d.max_regions, 10, 5000)),
        )

    @staticmethod
    def _num(var, default, lo, hi):
        try:
            v = float(var.get())
        except (tk.TclError, ValueError):
            return default
        return min(max(v, lo), hi)

    def start_analysis(self):
        if self.original is None or self._busy():
            return
        params = self._analysis_params()
        self._cancel = False
        self._analysis_dirty = False
        self.lbl_dirty.configure(text="")
        self.btn_open.state(["disabled"])
        self.btn_run.state(["disabled"])
        self.btn_cancel.state(["!disabled"])
        self.progress.configure(value=0)
        self.v_status.set("Analyse läuft ...")
        img = self.original
        q = self._queue

        def work():
            try:
                res = az.analyze(img, params, progress=lambda f, t: q.put(("progress", f, t)),
                                 cancel=lambda: self._cancel)
                q.put(("done", res))
            except az.Cancelled:
                q.put(("cancelled",))
            except Exception as e:                      # noqa: BLE001
                q.put(("error", e, traceback.format_exc()))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def cancel_analysis(self):
        if self._busy():
            self._cancel = True
            self.v_status.set("Wird abgebrochen ...")

    def _poll(self):
        last_progress = None
        try:
            while True:
                msg = self._queue.get_nowait()
                kind = msg[0]
                if kind == "progress":
                    last_progress = msg
                elif kind == "done":
                    self._on_done(msg[1])
                    last_progress = None
                elif kind == "cancelled":
                    self._finish_run()
                    self.v_status.set("Analyse abgebrochen.")
                    last_progress = None
                elif kind == "error":
                    self._finish_run()
                    self.v_status.set("Fehler bei der Analyse.")
                    sys.stderr.write(msg[2])
                    messagebox.showerror(APP_TITLE, f"Die Analyse ist fehlgeschlagen:\n\n{msg[1]}")
                    last_progress = None
        except queue.Empty:
            pass
        if last_progress is not None:
            self.progress.configure(value=last_progress[1] * 100)
            if last_progress[2]:
                self.v_status.set(f"Analyse läuft: {last_progress[2]}")
        self.after(60, self._poll)

    def _finish_run(self):
        self.btn_open.state(["!disabled"])
        self.btn_run.state(["!disabled"] if self.original is not None else ["disabled"])
        self.btn_cancel.state(["disabled"])
        self.progress.configure(value=0)

    def _on_done(self, res: az.AnalysisResult):
        self._finish_run()
        self.result = res
        t = res.timings
        h, w = res.shape
        self.v_status.set(
            f"Analyse fertig in {t['gesamt']:.1f} s (Farbe {t['farbe']:.1f} · Folgen {t['folgen']:.1f} · "
            f"Struktur {t['struktur']:.1f}) · Analysebild {w} x {h}" +
            (f" (Faktor {res.scale:.2f})" if res.scale > 1.001 else ""))
        self.progress.configure(value=100)
        self._run_detect()
        self.btn_save.state(["!disabled"])
        self.btn_csv.state(["!disabled"])

    def _mark_dirty(self, *_):
        if self.result is not None:
            self._analysis_dirty = True
            self.lbl_dirty.configure(text="Einstellungen geändert - bitte neu analysieren (F5).")

    # ------------------------------------------------------------------
    # Erkennung (sofort)
    # ------------------------------------------------------------------
    def _on_sens(self, _v=None):
        self.v_sens_text.set(f"{int(round(self.v_sens.get()))}")
        self._schedule_detect()

    def _schedule_detect(self, *_):
        if self.result is None:
            return
        if self._detect_job is not None:
            self.after_cancel(self._detect_job)
        self._detect_job = self.after(max(40, int(self._detect_ms * 1.5)), self._run_detect)

    def _run_detect(self):
        self._detect_job = None
        if self.result is None:
            return
        t = time.perf_counter()
        self.detection = az.detect(self.result, self._detect_params())
        self._detect_ms = (time.perf_counter() - t) * 1000.0
        self._view_cache.pop("heat", None)
        self._view_cache.pop("mask", None)
        self.selected_id = None
        self.v_details.set("")
        self._fill_list()
        self._update_info()
        self._redraw()

    def _update_info(self):
        d = self.detection
        if d is None:
            self.v_info.set("")
            return
        n = len(d.regions)
        txt = f"Schwelle {d.threshold:.1f}  ·  {n} Fund" + ("" if n == 1 else "e")
        if d.n_found > n:
            txt += f" (von {d.n_found})"
        extra = []
        if d.n_repeated:
            extra.append(f"{d.n_repeated} als wiederholt ausgeblendet")
        if d.n_small:
            extra.append(f"{d.n_small} zu klein")
        if extra:
            txt += "\n" + ", ".join(extra)
        self.v_info.set(txt)

    def _clear_list(self):
        self.tree.delete(*self.tree.get_children())

    def _fill_list(self):
        self._clear_list()
        if self.detection is None:
            return
        for r in self.detection.regions:
            self.tree.insert("", "end", iid=str(r.id), values=(
                r.id, f"{int(r.cx)}, {int(r.cy)}", f"{r.x1 - r.x0} x {r.y1 - r.y0}", f"{r.score:.1f}", r.reason))

    def _region(self, rid: int):
        if self.detection:
            for r in self.detection.regions:
                if r.id == rid:
                    return r
        return None

    def _on_tree_select(self, _e=None):
        sel = self.tree.selection()
        if not sel:
            return
        r = self._region(int(sel[0]))
        if r is None:
            return
        self.selected_id = r.id
        s = r.share
        self.v_details.set(
            f"Fund {r.id}: Mitte ({int(r.cx)}, {int(r.cy)}), Rahmen {r.x1 - r.x0} x {r.y1 - r.y0} px, "
            f"ca. {r.area_orig} Pixel, mittlere Farbe RGB{r.rgb}\n"
            f"Score {r.score:.1f} · Beteiligung: Farbe {s[0] * 100:.0f} % · Pixelfolge {s[1] * 100:.0f} % · "
            f"Struktur {s[2] * 100:.0f} %")
        self._center_on(r)
        self._redraw()

    def _on_tree_double(self, _e=None):
        sel = self.tree.selection()
        if sel:
            r = self._region(int(sel[0]))
            if r:
                self._zoom_to_region(r)

    # ------------------------------------------------------------------
    # Ansicht: Zoom, Verschieben, Zeichnen
    # ------------------------------------------------------------------
    def _canvas_size(self):
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def _canvas_center(self):
        w, h = self._canvas_size()
        return w // 2, h // 2

    def _clamp_view(self):
        if self.original is None:
            return
        W, H = self.original.size
        cw, ch = self._canvas_size()
        vw, vh = cw / self.zoom, ch / self.zoom
        self.ox = -(vw - W) / 2 if W <= vw else min(max(self.ox, 0), W - vw)
        self.oy = -(vh - H) / 2 if H <= vh else min(max(self.oy, 0), H - vh)

    def fit(self):
        if self.original is None:
            return
        W, H = self.original.size
        cw, ch = self._canvas_size()
        self.zoom = min(8.0, min(cw / W, ch / H) * 0.98)
        self.ox = self.oy = 0
        self.fit_mode = True
        self._clamp_view()
        self._redraw()

    def zoom_100(self):
        if self.original is None:
            return
        self._zoom_at(*self._canvas_center(), None, absolute=1.0)

    def _zoom_at(self, cx, cy, factor, absolute=None):
        if self.original is None:
            return
        old = self.zoom
        new = absolute if absolute is not None else old * factor
        new = min(max(new, 0.02), 64.0)
        px, py = self.ox + cx / old, self.oy + cy / old
        self.zoom = new
        self.ox, self.oy = px - cx / new, py - cy / new
        self.fit_mode = False
        self._clamp_view()
        self._redraw()

    def _center_on(self, r: az.Region):
        cw, ch = self._canvas_size()
        self.ox = r.cx - cw / (2 * self.zoom)
        self.oy = r.cy - ch / (2 * self.zoom)
        self.fit_mode = False
        self._clamp_view()

    def _zoom_to_region(self, r: az.Region):
        cw, ch = self._canvas_size()
        w, h = max(r.x1 - r.x0, 8), max(r.y1 - r.y0, 8)
        self.zoom = min(max(min(cw * 0.35 / w, ch * 0.35 / h), 0.5), 24.0)
        self._center_on(r)
        self._redraw()

    def _on_resize(self, _e=None):
        if self._redraw_job:
            self.after_cancel(self._redraw_job)
        self._redraw_job = self.after(30, self._after_resize)

    def _after_resize(self):
        self._redraw_job = None
        if self.original is None:
            self._draw_placeholder()
        elif self.fit_mode:
            self.fit()
        else:
            self._clamp_view()
            self._redraw()

    def _on_wheel(self, e):
        self._zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 1 / 1.25)

    def _on_press(self, e):
        self.canvas.focus_set()
        self._press = (e.x, e.y, self.ox, self.oy, False)

    def _on_drag(self, e):
        if self._press is None or self.original is None:
            return
        x0, y0, ox0, oy0, moved = self._press
        if not moved and abs(e.x - x0) + abs(e.y - y0) < 4:
            return
        self._press = (x0, y0, ox0, oy0, True)
        self.ox = ox0 - (e.x - x0) / self.zoom
        self.oy = oy0 - (e.y - y0) / self.zoom
        self.fit_mode = False
        self._clamp_view()
        self._redraw()

    def _on_release(self, e):
        if self._press is not None and not self._press[4]:
            self._click(e.x, e.y)
        self._press = None

    def _click(self, cx, cy):
        if self.detection is None or self.v_view.get() == "original" or not self.v_boxes.get():
            return
        x, y = self.ox + cx / self.zoom, self.oy + cy / self.zoom
        pad = 4
        hits = [r for r in self.detection.regions if r.x0 - pad <= x <= r.x1 + pad and r.y0 - pad <= y <= r.y1 + pad]
        if hits:
            r = min(hits, key=lambda r: (r.x1 - r.x0) * (r.y1 - r.y0))
            self.tree.selection_set(str(r.id))
            self.tree.see(str(r.id))
            # nicht neu zentrieren, wenn per Klick gewählt
            self._redraw()
        else:
            self.tree.selection_remove(self.tree.selection())
            self.selected_id = None
            self.v_details.set("")
            self._redraw()

    def _on_motion(self, e):
        if self.original is None:
            return
        x, y = int(self.ox + e.x / self.zoom), int(self.oy + e.y / self.zoom)
        W, H = self.original.size
        if not (0 <= x < W and 0 <= y < H):
            self.v_hover.set("")
            return
        txt = f"x {x}  y {y}  RGB{self.original.getpixel((x, y))}"
        if self.detection is not None:
            sx = min(int(x / self.detection.scale), self.detection.score.shape[1] - 1)
            sy = min(int(y / self.detection.scale), self.detection.score.shape[0] - 1)
            txt += f"  Score {self.detection.score[sy, sx]:.1f}"
        self.v_hover.set(txt)

    def _toggle_original(self):
        self.v_view.set("original" if self.v_view.get() != "original" else "overlay")
        self._redraw()

    def _draw_placeholder(self):
        c = self.canvas
        c.delete("all")
        w, h = self._canvas_size()
        c.create_text(w / 2, h / 2, fill="#8a8d91", justify="center", font=("TkDefaultFont", 13),
                      text="Foto öffnen mit Strg+O\n\nDie App sucht Pixel und Pixelfolgen,\n"
                           "die nicht zum Rest des Bildes passen.")

    def _base_image(self):
        """(Bild, Maßstab Original-Pixel pro Bild-Pixel) für die gewählte Ansicht."""
        mode = self.v_view.get()
        if mode == "original" or self.detection is None or mode == "overlay":
            return self.original, 1.0
        if mode not in self._view_cache:
            self._view_cache[mode] = az.render_heatmap(self.detection) if mode == "heat" else az.render_mask(self.detection)
        return self._view_cache[mode], self.detection.scale

    def _redraw(self):
        c = self.canvas
        if self.original is None:
            return
        c.delete("all")
        cw, ch = self._canvas_size()
        z, ox, oy = self.zoom, self.ox, self.oy
        base, bs = self._base_image()
        bw, bh = base.size
        bx0 = max(0, int(math.floor(ox / bs)))
        by0 = max(0, int(math.floor(oy / bs)))
        bx1 = min(bw, int(math.ceil((ox + cw / z) / bs)))
        by1 = min(bh, int(math.ceil((oy + ch / z) / bs)))
        if bx1 > bx0 and by1 > by0:
            crop = base.crop((bx0, by0, bx1, by1))
            dw = max(1, round((bx1 - bx0) * bs * z))
            dh = max(1, round((by1 - by0) * bs * z))
            resample = Image.NEAREST if bs * z >= 2.0 else Image.BILINEAR
            if resample == Image.BILINEAR and bs * z < 0.5:
                crop = crop.resize((dw, dh), Image.BOX)
            else:
                crop = crop.resize((dw, dh), resample)
            self.photo = ImageTk.PhotoImage(crop)
            c.create_image(round((bx0 * bs - ox) * z), round((by0 * bs - oy) * z), image=self.photo, anchor="nw")
        if self.detection is not None and self.v_boxes.get() and self.v_view.get() != "original":
            self._draw_boxes()

    def _draw_boxes(self):
        c = self.canvas
        cw, ch = self._canvas_size()
        z, ox, oy = self.zoom, self.ox, self.oy
        pad = 3
        drawn = 0
        selected = None
        for r in self.detection.regions:
            x0, y0 = (r.x0 - pad - ox) * z, (r.y0 - pad - oy) * z
            x1, y1 = (r.x1 + pad - ox) * z, (r.y1 + pad - oy) * z
            if x1 < 0 or y1 < 0 or x0 > cw or y0 > ch:
                continue
            if r.id == self.selected_id:
                selected = (r, x0, y0, x1, y1)
                continue
            self._draw_box(r, x0, y0, x1, y1, COL_BOX, 2)
            drawn += 1
            if drawn >= MAX_BOXES_DRAWN:
                break
        if selected:
            r, x0, y0, x1, y1 = selected
            self._draw_box(r, x0, y0, x1, y1, COL_SEL, 3)

    def _draw_box(self, r, x0, y0, x1, y1, color, width):
        c = self.canvas
        # sehr kleine Rahmen auf eine sichtbare Mindestgröße bringen
        if x1 - x0 < 10:
            m = (x0 + x1) / 2
            x0, x1 = m - 5, m + 5
        if y1 - y0 < 10:
            m = (y0 + y1) / 2
            y0, y1 = m - 5, m + 5
        c.create_rectangle(x0, y0, x1, y1, outline=color, width=width)
        if self.v_numbers.get():
            txt = str(r.id)
            tw = 8 * len(txt) + 6
            c.create_rectangle(x0 - 1, y0 - 16, x0 + tw, y0, fill=color, outline=color)
            c.create_text(x0 + 3, y0 - 8, text=txt, anchor="w", fill="#000000" if color == COL_SEL else "#ffffff",
                          font=("TkDefaultFont", 9, "bold"))

    # ------------------------------------------------------------------
    # Speichern
    # ------------------------------------------------------------------
    def _stem(self):
        return os.path.splitext(os.path.basename(self.path or "foto"))[0]

    def save_result(self):
        if self.detection is None or self.original is None:
            return
        path = filedialog.asksaveasfilename(
            title="Ergebnisbild speichern", defaultextension=".png", initialfile=f"{self._stem()}_anomalien.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Alle Dateien", "*.*")])
        if not path:
            return
        try:
            az.render_overlay(self.original, self.detection, numbers=self.v_numbers.get()).save(path)
            self.v_status.set(f"Gespeichert: {path}")
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Speichern fehlgeschlagen:\n{e}")

    def save_heatmap(self):
        if self.detection is None or self.original is None:
            return
        path = filedialog.asksaveasfilename(
            title="Heatmap speichern", defaultextension=".png", initialfile=f"{self._stem()}_heatmap.png",
            filetypes=[("PNG", "*.png"), ("Alle Dateien", "*.*")])
        if not path:
            return
        try:
            az.render_heatmap(self.detection).resize(self.original.size, Image.BILINEAR).save(path)
            self.v_status.set(f"Gespeichert: {path}")
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Speichern fehlgeschlagen:\n{e}")

    def save_csv(self):
        if self.detection is None:
            return
        path = filedialog.asksaveasfilename(
            title="Funde als CSV speichern", defaultextension=".csv", initialfile=f"{self._stem()}_funde.csv",
            filetypes=[("CSV (Excel, Semikolon)", "*.csv"), ("Alle Dateien", "*.*")])
        if not path:
            return
        try:
            az.regions_to_csv(self.detection, path)
            self.v_status.set(f"Gespeichert: {path}")
        except Exception as e:                          # noqa: BLE001
            messagebox.showerror(APP_TITLE, f"Speichern fehlgeschlagen:\n{e}")

    # ------------------------------------------------------------------
    def show_help(self):
        win = tk.Toplevel(self)
        win.title("Bedienung und Erklärung")
        win.geometry("720x640")
        txt = tk.Text(win, wrap="word", padx=12, pady=10)
        sb = ttk.Scrollbar(win, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")

    def _on_close(self):
        self._cancel = True
        self.destroy()


def _enable_dpi_awareness():
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    _enable_dpi_awareness()
    path = argv[0] if argv else None
    App(path).mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
