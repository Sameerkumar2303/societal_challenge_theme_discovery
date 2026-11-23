@echo off
title Societal Challenge Theme Discovery REST API
echo ======================================================================
echo Launching Societal Challenge Theme Discovery REST API (FastAPI)...
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
echo Starting Uvicorn server on http://127.0.0.1:8000 ...
echo Swagger documentation available at: http://127.0.0.1:8000/docs
"%PYTHON_EXE%" -m uvicorn api:app --reload --port 8000
pause
