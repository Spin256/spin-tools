# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Taskbar and window icon for every Spin tool (Spin256 · spin256.com/tools).

set_app_icon gives the process its
own AppUserModelID (one per tool, so each tool has its own taskbar button) and sets the suite's icon.ico on every
window. Call it right after the QApplication is created, before any window is shown."""
import sys
from pathlib import Path

ICON = Path(__file__).resolve().parents[1] / "icon.ico"


def app_id(tool):
    return f"Spin256.SpinTools.{tool}"


def set_app_icon(app, tool):
    """Own taskbar identity for ``tool`` (Windows) and the suite icon on every window. Never raises."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id(tool))
        except (AttributeError, OSError):
            pass
    found = ICON.is_file()
    if found:
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(str(ICON)))
    return found

