@echo off
setlocal
cd /d "%~dp0"
set "INSTALL_CHECK="
set "INSTALL_NO_PAUSE="
:arguments
if "%~1"=="" goto run
if /i "%~1"=="--check" (
    set "INSTALL_CHECK=-CheckOnly"
    shift
    goto arguments
)
if /i "%~1"=="--no-pause" (
    set "INSTALL_NO_PAUSE=1"
    shift
    goto arguments
)
echo Aufruf: install.bat [--check] [--no-pause]
exit /b 1
:run
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %INSTALL_CHECK%
set "INSTALL_RESULT=%ERRORLEVEL%"
echo.
if not defined INSTALL_NO_PAUSE pause
exit /b %INSTALL_RESULT%
