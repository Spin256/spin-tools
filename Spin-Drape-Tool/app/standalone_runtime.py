# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Provide diagnostic streams when Windows pythonw starts without a console."""
import os
import sys
import tempfile
import atexit

from project_settings import session_directory

_gui_log = None


def _close_gui_log():
    global _gui_log
    if _gui_log is not None:
        if sys.stdout is _gui_log: sys.stdout = None
        if sys.stderr is _gui_log: sys.stderr = None
        _gui_log.close()
        _gui_log = None


atexit.register(_close_gui_log)


def ensure_gui_streams():
    """Keep real console/redirection streams; supply only missing GUI streams."""
    global _gui_log
    if sys.stdout is not None and sys.stderr is not None:
        return None
    if _gui_log is None or _gui_log.closed:
        try:
            _gui_log = tempfile.NamedTemporaryFile(
                mode='a', encoding='utf-8', errors='replace', buffering=1,
                prefix=f'standalone-{os.getpid()}-', suffix='.log',
                dir=session_directory(), delete=False,
            )
        except OSError:
            _gui_log = open(os.devnull, 'w', encoding='utf-8', buffering=1)
    if sys.stdout is None: sys.stdout = _gui_log
    if sys.stderr is None: sys.stderr = _gui_log
    return _gui_log.name
