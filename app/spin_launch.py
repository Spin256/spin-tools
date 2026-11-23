# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Start a tool's app with the Python set in its settings.json. The "Open ... Tool.cmd" files run this.

    spin_launch.py <tool folder> <entry .py in the tool's app/> [arguments for the app...]
    spin_launch.py <tool folder> <entry> --check-launcher <output.json>   (writes the resolved paths; for tests)

python_executable in settings.json is relative to the tool folder (default .venv/Scripts/pythonw.exe). Arguments are
passed on as a list, so paths with spaces and quotes arrive unchanged. Errors show a message box (pythonw has no
console).

Before starting, the tool's packages (its requirements.txt) are looked up with that Python, without importing them.
The app's error output goes to temp/startup.log in the tool folder; if it stops within a few seconds, the end of that
log is shown, and missing or damaged packages offer to run Install Spin Tools.cmd."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SUITE = Path(__file__).resolve().parents[1]
MODULES = {"numpy": "numpy", "pyside6-essentials": "PySide6", "vtk": "vtkmodules", "warp-lang": "warp",
           "ufbx": "ufbx"}
STARTUP_SECONDS = 4.
BROKEN = ("ModuleNotFoundError", "ImportError", "DLL load failed", "No module named")


def resolve(tool, entry):
    tool = Path(tool).resolve()
    settings_file = tool / "settings.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8-sig")) if settings_file.is_file() else {}
    python = settings.get("python_executable", ".venv/Scripts/pythonw.exe")
    if not isinstance(python, str) or not python.strip():
        raise ValueError("python_executable in settings.json must be a nonempty path.")
    python = Path(python)
    python = (python if python.is_absolute() else tool / python).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python was not found at {python}. Run Install Spin Tools.cmd, "
                                "or set python_executable in settings.json.")
    return python, tool / "app" / entry, tool


def show_error(message, title):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, str(message), title, 0x10)
    except (AttributeError, OSError):
        print(message, file=sys.stderr)


def ask(message, title):
    """Yes/No message box; True for Yes."""
    try:
        import ctypes
        return ctypes.windll.user32.MessageBoxW(None, str(message), title, 0x34) == 6
    except (AttributeError, OSError):
        print(message, file=sys.stderr)
        return False


def offer_install(message, title):
    if ask(message + "\n\nRun Install Spin Tools.cmd now? It repairs the shared Python environment.", title):
        subprocess.Popen(["cmd", "/c", "start", "", str(SUITE / "Install Spin Tools.cmd")], cwd=SUITE)


def missing_packages(python, tool):
    """Packages from the tool's requirements.txt that ``python`` cannot find (looked up, not imported)."""
    names = []
    requirements = tool / "requirements.txt"
    if requirements.is_file():
        for line in requirements.read_text(encoding="utf-8-sig").splitlines():
            name = line.partition("#")[0].split("==")[0].strip().casefold().replace("_", "-")
            if name in MODULES:
                names.append(MODULES[name])
    if not names:
        return []
    code = ("import importlib.util, sys; "
            "print(' '.join(n for n in sys.argv[1:] if importlib.util.find_spec(n) is None))")
    run = subprocess.run([str(python), "-c", code, *names], capture_output=True, text=True, timeout=60,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if run.returncode:
        return names
    return run.stdout.split()


def startup_log(tool):
    log = tool / "temp" / "startup.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.is_file() and log.stat().st_size > 1_000_000:
        try:
            log.unlink()
        except OSError:
            pass
    return log


def main(argv):
    if len(argv) < 2:
        show_error("Usage: spin_launch.py <tool folder> <entry> [arguments]", "Spin Tools launcher")
        return 2
    tool, entry, arguments = argv[0], argv[1], argv[2:]
    try:
        python, script, folder = resolve(tool, entry)
        if arguments[:1] == ["--check-launcher"]:
            Path(arguments[1]).write_text(json.dumps(dict(python=str(python), entry=str(script),
                                                          working_directory=str(folder))), encoding="utf-8")
            return 0
        title = f"{folder.name} launcher"
        missing = missing_packages(python, folder)
        if missing:
            offer_install(f"{folder.name} cannot start: these packages are missing: {', '.join(missing)}.", title)
            return 1
        log = startup_log(folder)
        with log.open("ab") as output:
            start = output.tell()
            process = subprocess.Popen([str(python), str(script), *arguments], cwd=folder, stdout=output,
                                       stderr=output, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        deadline = time.monotonic() + STARTUP_SECONDS
        while time.monotonic() < deadline and process.poll() is None:
            time.sleep(.1)
        if process.poll() not in (None, 0):
            with log.open("rb") as source:
                source.seek(start)
                text = source.read().decode("utf-8", "replace").strip()
            tail = "\n".join(text.splitlines()[-12:]) or f"exit code {process.returncode}"
            if any(word in text for word in BROKEN):
                offer_install(f"{folder.name} stopped while starting:\n\n{tail}", title)
            else:
                show_error(f"{folder.name} stopped while starting:\n\n{tail}\n\nFull output: {log}", title)
            return 1
        return 0
    except FileNotFoundError as error:
        offer_install(str(error), f"{Path(tool).resolve().name} launcher")
        return 1
    except Exception as error:
        show_error(error, f"{Path(tool).resolve().name} launcher")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
