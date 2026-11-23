@echo off
rem Spin Tools - Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
rem SPDX-License-Identifier: AGPL-3.0-only - see LICENSE and ADDITIONAL_TERMS.md.
rem The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
rem Sets up the shared Python environment (.venv) for every Spin tool. Plain batch: no PowerShell needed.
rem Options are passed to app\setup_environment.py: --wheelhouse <folder> (offline wheels), --skip-gpu-check.
rem To use a particular Python, run instead:  "C:\Path\To\python.exe" app\setup_environment.py
setlocal EnableExtensions
set "SPIN_ROOT=%~dp0"
set "SPIN_CHECK=import sys,struct; assert sys.platform=='win32' and struct.calcsize('P')==8 and (3,10)<=sys.version_info[:2]<=(3,13)"
set "SPIN_PY="
if exist "%SPIN_ROOT%.venv\Scripts\python.exe" call :try "%SPIN_ROOT%.venv\Scripts\python.exe"
for %%v in (3.10 3.11 3.12 3.13) do if not defined SPIN_PY call :try py -%%v
if not defined SPIN_PY call :try python
if not defined SPIN_PY call :try python3
if not defined SPIN_PY (
    echo Setup failed: Install 64-bit CPython 3.10-3.13 first, then rerun setup. Python 3.10 is the tested configuration.
    set "SPIN_RESULT=1"
    goto done
)
echo Using Python: %SPIN_PY%
%SPIN_PY% "%SPIN_ROOT%app\setup_environment.py" %*
set "SPIN_RESULT=%ERRORLEVEL%"
:done
if not "%SPIN_RESULT%"=="0" echo Installation failed. See temp\install.log.
if not defined SPIN_NO_PAUSE pause
exit /b %SPIN_RESULT%

:try
%* -c "%SPIN_CHECK%" >nul 2>&1 && set SPIN_PY=%*
exit /b 0
