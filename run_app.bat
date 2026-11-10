@echo off
title Societal Challenge Theme Discovery Web App
echo ======================================================================
echo Launching Societal Challenge Theme Discovery Platform...
echo ======================================================================

set PYTHON_EXE=%~dp0.venv\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=C:\Users\LENOVO\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe
)

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=C:\Users\LENOVO\AppData\Local\Programs\Python\Python314\python.exe
)

if not exist "%PYTHON_EXE%" (
    echo Local Python runtime not found at expected paths. Trying global python...
    set PYTHON_EXE=python
)

echo Using Python: "%PYTHON_EXE%"
"%PYTHON_EXE%" -m streamlit run app.py --server.port=8501 --server.headless=false
pause
