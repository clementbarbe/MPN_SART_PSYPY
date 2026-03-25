@echo off

REM Activation de l’environnement conda via Miniforge
call "conda.bat" activate Sart

REM Aller dans le dossier du projet
cd /d "MPN_SART_PSYPY"

REM Exécuter le script Python
python main.py

pause
