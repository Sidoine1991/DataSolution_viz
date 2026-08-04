@echo off
cd /d "%~dp0"

echo Demarrage du tableau de bord CCR-B...
echo.

if not exist ".venv" (
    echo Creation de l'environnement virtuel...
    python -m venv .venv
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERREUR: impossible d'activer .venv
    pause
    exit /b 1
)

pip install -r requirements.txt
streamlit run dashboard_app.py
pause
