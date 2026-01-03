@echo off
REM ========================================
REM  Assistant de Triage Medical ESI
REM  Version 4.0 - Multilingue & Audio IA
REM ========================================
echo.
echo ========================================
echo  Assistant de Triage Medical ESI v4.0
echo ========================================
echo.
echo Nouvelle version avec:
echo  [+] Traduction intelligente (FR/EN/AR)
echo  [+] Transcription audio multilingue
echo  [+] Detection automatique de langue
echo  [+] Traduction audio automatique
echo  [+] Interface moderne et reactive
echo.

REM Corriger le conflit OpenMP
set KMP_DUPLICATE_LIB_OK=TRUE

REM Configuration de l'encodage UTF-8 pour Windows
chcp 65001 > nul 2>&1

echo [1/3] Activation de l'environnement...
call conda activate esi_training
if errorlevel 1 (
    echo ERREUR: Impossible d'activer l'environnement conda 'esi_training'
    echo Verifiez que l'environnement existe avec: conda env list
    pause
    exit /b 1
)

echo [2/3] Verification des dependances...
python -c "import streamlit, torch, transformers, faster_whisper, deep_translator" 2>nul
if errorlevel 1 (
    echo ATTENTION: Certaines dependances sont manquantes.
    echo Installation en cours...
    pip install -q -r requirements.txt
)

echo [3/3] Lancement de l'application...
echo.
echo ========================================
echo  Interface disponible sur:
echo  - Local:   http://localhost:8501
echo  - Reseau:  http://192.168.1.16:8501
echo ========================================
echo.
echo Fonctionnalites principales:
echo  [*] Analyse de triage ESI intelligente
echo  [*] Transcription audio multilingue (Whisper)
echo  [*] Traduction automatique (FR/EN/AR)
echo  [*] Detection de drapeaux rouges
echo  [*] Recommandations d'examens
echo  [*] Raisonnement clinique IA
echo  [*] Export PDF des rapports
echo.
echo ========================================
echo  L'application demarre...
echo  Appuyez sur Ctrl+C pour arreter
echo ========================================
echo.

REM Lancer Streamlit
streamlit run app.py

REM Si l'application se ferme avec une erreur
if errorlevel 1 (
    echo.
    echo ========================================
    echo  ERREUR lors du lancement
    echo ========================================
    echo.
    echo Solutions possibles:
    echo  1. Verifiez que tous les fichiers sont presents
    echo  2. Reinstallez les dependances: pip install -r requirements.txt
    echo  3. Verifiez les logs ci-dessus pour plus de details
    echo.
    pause
)
