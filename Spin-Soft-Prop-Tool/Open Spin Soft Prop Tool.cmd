@echo off
rem Spin Tools - Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
rem SPDX-License-Identifier: AGPL-3.0-only - see LICENSE and ADDITIONAL_TERMS.md.
rem The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
rem Starts the tool with the Python set in settings.json (app\spin_launch.py in the suite). No PowerShell needed.
set "SPIN_PY=%~dp0..\.venv\Scripts\pythonw.exe"
if not exist "%SPIN_PY%" (
    echo Spin Tools is not set up yet.
    choice /C YN /M "Run Install Spin Tools.cmd now"
    if errorlevel 2 exit /b 1
    call "%~dp0..\Install Spin Tools.cmd"
    if not exist "%SPIN_PY%" exit /b 1
)
"%SPIN_PY%" "%~dp0..\app\spin_launch.py" "%~dp0." soft_app.py %*
exit /b %ERRORLEVEL%
