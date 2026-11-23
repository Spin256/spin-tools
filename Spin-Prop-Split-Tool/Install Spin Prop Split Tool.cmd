@echo off
rem Spin Tools - Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
rem SPDX-License-Identifier: AGPL-3.0-only - see LICENSE and ADDITIONAL_TERMS.md.
rem The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
rem Every tool shares the suite's Python environment: this runs the suite installer.
call "%~dp0..\Install Spin Tools.cmd" %*
exit /b %ERRORLEVEL%
