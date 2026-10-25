# Anomalie-Finder

Findet in einem Foto Pixel und Pixelabfolgen, die nicht zum Rest des Bildes passen: eine Möwe im blauen
Himmel, einen Planeten zwischen lauter gleichartigen Sternen oder eine Stelle, an der die gewohnte
Abfolge der Pixel bricht.

Das Programm kennt keine Objekte und braucht kein Training. Es lernt aus dem Foto selbst, was dort
"normal" ist, und markiert alles, was unter diesem Modell unwahrscheinlich ist.

## Installation

Voraussetzung: Python 3.9 oder neuer (bei Windows der Installer von python.org, darin ist Tkinter enthalten).

```
pip install -r requirements.txt
```

Linux: Falls `tkinter` fehlt, `sudo apt install python3-tk`.

## Start

```
python anomalie_finder.py              # Fenster öffnen, Foto mit Strg+O laden
python anomalie_finder.py foto.jpg     # Foto direkt laden
```

Unter Windows geht auch ein Doppelklick auf `start_windows.bat`. Ein Foto, das man auf diese Datei zieht,
wird direkt geöffnet.

Ohne Oberfläche (Kommandozeile):

```
python analyzer.py foto.jpg --sens 60 --out ergebnis.png --csv funde.csv
```

## Bedienung

| Was | Wie |
|---|---|
| Foto öffnen | Strg+O oder Knopf "Foto öffnen ..." (die Analyse startet automatisch) |
| **Empfindlichkeit** | Regler links oben, wirkt sofort. Rechts = schlägt schneller an, links = nur Offensichtliches |
| Zoomen / Verschieben | Mausrad / Ziehen mit gedrückter linker Taste |
| Fund auswählen | Klick auf den Rahmen oder Zeile in der Liste. Doppelklick in der Liste zoomt hin |
| Original <-> Markierung | Leertaste |
| Einpassen / 100 % | Taste 0 / Taste 1 |
| Speichern | Strg+S (Bild mit Rahmen), Menü Datei: Heatmap und CSV |
| Neu berechnen | F5 |

Ansichten (Reiter "Ansicht"): Original mit Markierung, Heatmap (Orange = Schwelle, je heller, desto
auffälliger), Maske der Funde, nur Original.

## Wie es funktioniert

Drei Detektoren bewerten jedes Pixel. Jeder misst, wie unwahrscheinlich das Pixel im Vergleich zum Rest
desselben Fotos ist:

1. **Farbe selten.** Wie häufig kommt die Farbe im ganzen Bild vor? (weiße Möwe im blauen Himmel)
2. **Pixelfolge passt nicht.** Das Bild wird auf wenige Farbklassen reduziert. Dann wird gezählt, welche
   Klasse in welcher Richtung (waagrecht, senkrecht, beide Diagonalen) und in welchem Abstand (1 bis
   Reichweite) neben welcher anderen vorkommt. Daraus entsteht für jedes Pixel:
   - Paare: *Wie wahrscheinlich ist Farbe A neben Farbe B?*
   - Tripel: *Wie wahrscheinlich ist die Mitte B zwischen A und C?*

   Das ist deine Idee "neben Pixel A ist normalerweise immer Pixel B": Wo das nicht stimmt, schlägt es an,
   auch wenn jede Farbe für sich häufig ist. Jedes Pixel wird dabei ohne sein eigenes Vorkommen bewertet,
   sonst würde sich ein Ausreißer selbst normalisieren. Die Farbklassen werden auf zwei gegeneinander
   verschobenen Rastern gezählt, damit Rauschen an Klassengrenzen keinen Alarm auslöst.
3. **Struktur selten.** Kleine Bildausschnitte (5x5, in zwei Maßstäben) werden verdichtet. Selten ist ein
   Ausschnitt, wenn es im Bild kaum ähnliche gibt.

Jede Karte wird robust (Median/MAD) in "wie viele Standardabweichungen über dem Üblichen" umgerechnet und
gewichtet kombiniert. Die **Empfindlichkeit** legt nur die Schwelle fest. Deshalb reagiert der Regler
sofort, ohne das Foto neu zu rechnen.

Treffer werden zu **Regionen** verschmolzen. Der **Wiederholungsfilter** erklärt viele gleichartige
Regionen (ähnliche Größe, Form und Farbe) für normal, z. B. tausend Sterne. Etwas Einzelnes, das
heraussticht, bleibt.

## Einstellungen

Sofort wirksam:

| Einstellung | Bedeutung |
|---|---|
| Empfindlichkeit | 0 bis 100. Schwelle sinkt exponentiell von 40 auf 3 |
| Gewichte der Detektoren | 0 % schaltet einen Detektor aus |
| Mindestgröße | Kleinere Funde werden ignoriert (Standard 4 Pixel) |
| Zusammenfassen | Treffer mit kleinerem Abstand werden eine Region |
| Wiederholtes ausblenden / Gruppengröße | Ab so vielen ähnlichen Funden gilt eine Gruppe als normal (Standard 6) |

Braucht "Neu analysieren" (Reiter "Analyse"): maximale Analysegröße, Farbgenauigkeit, Rauschglättung,
Reichweite der Pixelfolgen, Doppelraster, Strukturdetektor an/aus.

### Tipps

- **Einzelnes abweichendes Pixel:** Mindestgröße auf 1 stellen (Standard sind 4 Pixel, damit Rauschen nicht
  als Fund zählt).
- **Nur Pixelabfolgen prüfen:** Gewichte "Farbe" und "Struktur" auf 0 stellen, Empfindlichkeit etwa 70.
- **Zu viele Funde / Rauschen wird markiert:** Empfindlichkeit zurücknehmen oder Mindestgröße erhöhen.
- **Zu wenig Funde:** Empfindlichkeit erhöhen. Auf strukturreichen Fotos (Laub, Menschenmengen)
  entstehen dabei schnell viele Treffer.
- **Schneller:** Strukturdetektor ausschalten (spart etwa zwei Drittel der Zeit) oder Analysegröße senken.

## Grenzen

- Das Foto definiert "normal". Ein Objekt, das mehr als einige Prozent der Bildfläche einnimmt, ist selbst
  Teil des Normalen und wird nur bei hoher Empfindlichkeit oder über seine Ränder erkannt.
- "Auffällig" heißt statistisch selten, nicht "wichtig". Die App erkennt keine Gegenstände und sagt nicht,
  *was* ein Fund ist.
- Große Fotos werden für die Analyse auf höchstens 1200 Pixel Kantenlänge verkleinert (einstellbar).
  Einzelpixel-Details des Originals gehen dabei verloren, die Rahmen gelten aber im Originalbild.
- Sensorrauschen und JPEG-Artefakte werden durch Glättung und die Totzone der Bewertung unterdrückt.
  Bei sehr hoher Empfindlichkeit tauchen trotzdem Rauschtreffer auf, vor allem am Bildrand.

## Dateien

| Datei | Inhalt |
|---|---|
| `anomalie_finder.py` | Oberfläche (Tkinter) |
| `analyzer.py` | Analyse-Engine, auch als Modul und auf der Kommandozeile nutzbar |
| `selftest.py` | Automatische Prüfung der Engine an Testbildern mit bekannter Lösung |
| `testbilder.py` | Erzeugt die Testbilder (Möwe, Sternenhimmel mit Planet, Streifenmuster, Rauschen, Textur) |
| `testbilder/` | Fertige Testbilder zum Ausprobieren |

Selbsttest: `python selftest.py` (je nach Rechner 30 bis 60 Sekunden).

Als Modul:

```python
from PIL import Image
import analyzer as az

res = az.analyze(Image.open("foto.jpg"))                       # rechenintensiv, einmal pro Foto
det = az.detect(res, az.DetectParams(sensitivity=60))          # schnell, beliebig oft
for r in det.regions:
    print(r.id, r.x0, r.y0, r.x1, r.y1, r.score, r.reason)
```
