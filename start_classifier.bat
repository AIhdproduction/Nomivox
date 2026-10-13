@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"
set "PIP_CACHE_DIR=%~dp0.cache\pip"
set "TEMP=%~dp0.cache\temp"
set "TMP=%TEMP%"
if not exist "%TEMP%" mkdir "%TEMP%"
set "VENV_PY=%~dp0.cache\venv\Scripts\python.exe"
if exist "%VENV_PY%" goto check_version
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
    py -3.12 -m venv "%~dp0.cache\venv"
    if errorlevel 1 goto setup_error
    goto check_version
)
python -c "import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 goto python_error
python -m venv "%~dp0.cache\venv"
if errorlevel 1 goto setup_error
:check_version
"%VENV_PY%" -c "import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 goto check_packages
py -3.12 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 goto python_error
py -3.12 -m venv --upgrade "%~dp0.cache\venv"
if errorlevel 1 goto setup_error
"%VENV_PY%" -c "import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 goto setup_error
:check_packages
"%VENV_PY%" -m e57cls.bootstrap
if errorlevel 1 goto setup_error
:launch
"%VENV_PY%" -m e57cls.launcher %*
if errorlevel 1 goto launch_error
exit /b 0
:python_error
set "E57CLS_MESSAGE_KEY=python_missing"
goto show_error
:setup_error
set "E57CLS_MESSAGE_KEY=setup_failed"
goto show_error
:launch_error
set "E57CLS_MESSAGE_KEY=launch_failed"
:show_error
powershell -NoProfile -Command "$s=Get-Content -LiteralPath 'e57cls/gui/strings_de.json' -Encoding UTF8 -Raw | ConvertFrom-Json; Write-Host $s.($env:E57CLS_MESSAGE_KEY)"
pause
exit /b 1
