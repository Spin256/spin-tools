# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Detached DAZ helper launch and immutable return revisions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def launch(session):
    entry = Path(__file__).with_name("body_app.py")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen(
        [sys.executable, str(entry), "--session", str(session)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, start_new_session=os.name != "nt",
    )


def publish_revision(session_path, spec, write_result):
    """Publish the revision only after its complete set of files exists."""
    session_path = Path(session_path).absolute()
    session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    revision = Path(tempfile.mkdtemp(prefix=session_path.stem + "-return-", dir=session_path.parent))
    latest = session_path.with_suffix(".latest")
    pending = revision / "latest.tmp"

    def output_path(value):
        return str(revision / Path(value).name)

    try:
        result = dict(spec, result_path=output_path(spec["result_path"]), garments=[])
        for entry in spec["garments"]:
            result["garments"].append(dict(
                entry, return_path=output_path(entry["return_path"]),
                animation_path=output_path(entry["animation_path"]),
            ))
        revision_entries = {entry["id"]: entry for entry in result["garments"]}
        session["meshes"] = [revision_entries.get(entry["id"], entry) for entry in session["meshes"]]
        session["result_path"] = result["result_path"]
        manifest = revision / "session.json"
        write_result(result)
        manifest.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
        pending.write_text(str(manifest), encoding="utf-8")
        os.replace(pending, latest)
        return manifest
    except Exception:
        if revision.resolve().parent == session_path.parent.resolve():
            for path in revision.iterdir():
                try:
                    path.unlink()
                except OSError:
                    pass
            try:
                revision.rmdir()
            except OSError:
                pass
        raise


def release_capture(session_path):
    """Remove bulk samples after exit, retaining templates for a pending receive."""
    session_path = Path(session_path).resolve()
    session = json.loads(session_path.read_text(encoding="utf-8-sig"))
    has_result = session_path.with_suffix(".latest").is_file()
    keep = {Path(entry["path"]).resolve() for entry in session["meshes"]
            if has_result and entry["role"] == "garment"}
    for entry in session["meshes"]:
        for value in [entry["path"], *entry.get("samples", [])]:
            path = Path(value).resolve()
            if (path not in keep and path.parent == session_path.parent
                    and path.name.startswith(session_path.stem + "-")):
                path.unlink(missing_ok=True)
    if not has_result:
        session_path.unlink(missing_ok=True)


if __name__ == "__main__":
    launch(sys.argv[1])
