# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Spin Soft Prop Tool branding."""

APP_NAME = 'Spin Soft Prop Tool'
VERSION = '1.0.0'          # this tool's own version, shown in the footer


def footer(parent=None):
    import sys
    from pathlib import Path
    suite_app = Path(__file__).resolve().parents[2] / 'app'
    if str(suite_app) not in sys.path:
        sys.path.append(str(suite_app))
    import spin_ui
    return spin_ui.footer(parent, VERSION, Path(__file__).resolve().parent.parent)
