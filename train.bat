@echo off
REM Script pour lancer le fine-tuning avec correction OpenMP
echo ========================================
echo LANCEMENT FINE-TUNING ESI
echo ========================================
echo.

REM Corriger le conflit OpenMP
set KMP_DUPLICATE_LIB_OK=TRUE

echo [1/3] Configuration OpenMP... OK
echo [2/3] Activation environnement conda...

REM Activer l'environnement conda
call conda activate nlp_medical

echo [3/3] Lancement du fine-tuning...
echo.
echo ========================================
echo Debut du fine-tuning (15-30 minutes)
echo ========================================
echo.

REM Lancer le script
python finetune_custom_esi.py

echo.
echo ========================================
echo Fine-tuning termine!
echo ========================================
pause
