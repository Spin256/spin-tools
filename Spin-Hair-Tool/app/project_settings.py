# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Project settings shared with the DAZ capture script via settings.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def session_directory(root=None):
    root = Path(root) if root is not None else ROOT
    settings_path = root / 'settings.json'
    settings = json.loads(settings_path.read_text(encoding='utf-8-sig')) if settings_path.exists() else {}
    if not isinstance(settings, dict):
        raise ValueError('settings.json must contain an object.')
    folder = settings.get('session_folder', 'temp')
    if not isinstance(folder, str) or not folder.strip():
        raise ValueError('session_folder must be a nonempty relative or absolute folder path.')
    path = Path(folder)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
