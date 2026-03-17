@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
cd /d "%PROJECT_DIR%" || exit /b 1

if not exist "%PROJECT_DIR%\hub_data\logs" mkdir "%PROJECT_DIR%\hub_data\logs"
if not exist "%PROJECT_DIR%\hub_data\.mplconfig" mkdir "%PROJECT_DIR%\hub_data\.mplconfig"

set "POWERTRADER_PROJECT_DIR=%PROJECT_DIR%"
set "POWERTRADER_GUI_SETTINGS=%PROJECT_DIR%\gui_settings.json"
set "POWERTRADER_HUB_DIR=%PROJECT_DIR%\hub_data"
set "MPLCONFIGDIR=%PROJECT_DIR%\hub_data\.mplconfig"
if defined PYTHONPATH (
    set "PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%"
) else (
    set "PYTHONPATH=%PROJECT_DIR%"
)

set "VENV_DIR=%PROJECT_DIR%\venv"
set "PY_BIN=%VENV_DIR%\Scripts\python.exe"

if not exist "%PY_BIN%" (
    echo [launch] venv not found; creating at "%VENV_DIR%"
    py -3 -m venv "%VENV_DIR%" 2>nul
    if errorlevel 1 (
        python -m venv "%VENV_DIR%"
    )
)

"%PY_BIN%" -c "import matplotlib" >nul 2>nul
if errorlevel 1 (
    echo [launch] installing dependencies from requirements.txt
    "%PY_BIN%" -m pip install --upgrade pip setuptools wheel
    "%PY_BIN%" -m pip install -r "%PROJECT_DIR%\requirements.txt"
)

"%PY_BIN%" -m ui.pt_hub
exit /b %errorlevel%
