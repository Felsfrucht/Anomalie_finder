@echo off
rem Startet den Anomalie-Finder. Ein auf diese Datei gezogenes Foto wird direkt geoeffnet.
cd /d "%~dp0"
python anomalie_finder.py %*
if errorlevel 1 (
    echo.
    echo Start fehlgeschlagen. Ist Python installiert und wurde "pip install -r requirements.txt" ausgefuehrt?
    pause
)
