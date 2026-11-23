# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Portable tool-root settings shared with the DAZ launcher."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_settings(root=ROOT):
    path = Path(root)/'settings.json'
    data = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    if not isinstance(data, dict): raise ValueError('settings.json must contain an object.')
    return data


def session_directory(root=ROOT):
    value = load_settings(root).get('session_folder', 'temp')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('session_folder must be a nonempty path.')
    path = Path(value)
    if not path.is_absolute(): path = Path(root)/path
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
