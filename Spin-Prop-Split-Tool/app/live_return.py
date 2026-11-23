# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Detached selector launch for the Spin Tools receive window.

``Split Selected Prop.dsa`` runs this and returns control to DAZ straight away.
The selector writes the split with the ``split_obj`` writer, then
points ``<session>.latest`` at the result; ``Receive Prop Split Result.dsa``
imports it. A selector closed without splitting removes its capture files.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def launch(source, result, session):
    entry = Path(__file__).with_name("selector.py")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(
        [sys.executable, str(entry), "--bridge-source", str(source), "--bridge-result", str(result),
         "--bridge-session", str(session)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, start_new_session=os.name != "nt",
    )


def publish(session, result):
    """Point the receive window at a completed split result."""
    session = Path(session)
    pending = session.with_suffix(".latest.tmp")
    pending.write_text(str(Path(result)), encoding="utf-8")
    os.replace(pending, session.with_suffix(".latest"))


def release_unsent(session, source):
    """A selector closed without a split leaves nothing for the receive window."""
    session = Path(session)
    if session.with_suffix(".latest").is_file():
        return
    for path in (Path(source), session):
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge-source", required=True)
    parser.add_argument("--bridge-result", required=True)
    parser.add_argument("--bridge-session", required=True)
    args = parser.parse_args()
    launch(args.bridge_source, args.bridge_result, args.bridge_session)
