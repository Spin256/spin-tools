# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Shared keyboard shortcuts for the Spin Tools suite.

All tools read one file, keybindings.json in the suite folder: a "common" section used by
every tool and one section per tool ("drape", "hair", ...) that adds or overrides actions.
"""
import json
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
PATH = SUITE / "keybindings.json"


def load(tool, path=PATH):
    """Return ({action: [key, ...]}, errors) for ``tool``: its section merged over common.
    A missing or malformed file gives no bindings and an error, never an exception."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("Expected an object with 'common' and per-tool sections")
    except (OSError, ValueError) as error:
        return {}, [str(error)]
    merged, errors = {}, []
    for section in ("common", tool):
        values = data.get(section, {})
        if not isinstance(values, dict):
            errors.append(f"{section}: expected an object")
            continue
        for action, keys in values.items():
            keys = [keys] if isinstance(keys, str) else keys
            if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
                errors.append(f"{section}.{action}")
                continue
            merged[action] = keys
    return merged, errors
