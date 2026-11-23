# Spin Tools — Originally developed by Spin256 (spin256.com/tools). Copyright (C) 2026 Spin256.
# SPDX-License-Identifier: AGPL-3.0-only — see LICENSE and ADDITIONAL_TERMS.md.
# The Spin256 name, logo and icon are not licensed for reuse; see ADDITIONAL_TERMS.md.
"""Spin Tools launcher: one window with a button per standalone tool (Spin256 · spin256.com/tools).

Each button starts the tool's own "Open ... Tool.cmd" route (spin_launch.py with the tool's settings.json Python) as
a separate process, so the launcher can stay open or be closed while tools run.

"Clear temporary files" empties each tool's session folder (captures, results, frames, logs) and the suite's temp
folder (keeping the install log and report), only while no Spin tool is running."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

SUITE = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'          # the suite's version (release zip name, launcher footer)
TOOLS = (
    ('Spin Drape Tool', 'Spin-Drape-Tool', 'drape_app.py',
     'Cloth simulation: drape garments over still or animated bodies and props, with wind, water and pinning.'),
    ('Spin Hair Tool', 'Spin-Hair-Tool', 'hair_app.py',
     'Settle card hair on an avatar, still or animated, with freezing, gravity, wind, water and grown hair.'),
    ('Spin Soft Body Tool', 'Spin-Soft-Body-Tool', 'body_app.py',
     'Contact deformation, jiggle and example-driven motion in painted areas of a body.'),
    ('Spin Soft Prop Tool', 'Spin-Soft-Prop-Tool', 'soft_app.py',
     'Cushions, mattresses and other soft props pressed by still or moving objects.'),
    ('Spin Prop Split Tool', 'Spin-Prop-Split-Tool', 'selector.py',
     'Select polygons of a prop and export them as separate pieces with their own pivots.'),
)
NOTE = ('Each tool opens its own file window (OBJ and FBX). In DAZ Studio, use Scripts > Spin Tools instead; '
        'Current Frame to Zero is available there only.')


KEEP = {'install.log', 'install-report.json'}         # suite temp/: the last installation's record
OWN = ('.session-', '.bridge-', '.drape-', '.hair-', 'standalone-', 'tmp', 'startup.log', '.warp-check-cache')


def temp_folders():
    """(folder, clear everything) for each tool's session folder and the suite's temp folder. A session folder set
    outside its tool in settings.json is shared with other files, so only the tools' own files are cleared there."""
    folders = []
    for _, tool, _, _ in TOOLS:
        root = SUITE / tool
        try:
            settings = json.loads((root / 'settings.json').read_text(encoding='utf-8-sig'))
            folder = Path(settings.get('session_folder', 'temp'))
        except (OSError, ValueError, AttributeError, TypeError):
            folder = Path('temp')
        folder = (folder if folder.is_absolute() else root / folder).resolve()
        folders.append((folder, folder == (root / 'temp').resolve()))
    folders.append(((SUITE / 'temp').resolve(), True))
    return folders


def clearable(folder, everything):
    if not folder.is_dir():
        return []
    items = [p for p in folder.iterdir() if p.name not in KEEP or folder != (SUITE / 'temp').resolve()]
    return items if everything else [p for p in items if p.name.startswith(OWN)]


def size_of(path):
    try:
        if path.is_symlink() or path.is_file():
            return path.stat().st_size
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file() and not f.is_symlink())
    except OSError:
        return 0


def temp_usage():
    return sum(size_of(p) for folder, everything in temp_folders() for p in clearable(folder, everything))


def clear_temp():
    """Delete everything clearable; files Windows reports as in use are skipped. Returns (bytes freed, skipped)."""
    freed, skipped = 0, 0
    for folder, everything in temp_folders():
        for path in clearable(folder, everything):
            size = size_of(path)
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                freed += size
            except OSError:
                skipped += 1
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
    return freed, skipped


def running_tools():
    """Names of Spin tools running from this suite (their command lines), read with a CIM query; [] if unavailable."""
    query = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             "ForEach-Object { $_.CommandLine }")
    try:
        run = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', query], capture_output=True,
                             text=True, timeout=20, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    except (OSError, subprocess.SubprocessError):
        return []
    names = []
    for line in run.stdout.splitlines():
        for name, tool, entry, _ in TOOLS:
            if str(SUITE / tool / 'app' / entry).casefold() in line.casefold() and name not in names:
                names.append(name)
    return names


def readable(size):
    for unit in ('bytes', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'bytes' else f'{size:.1f} {unit}'
        size /= 1024.


def launch_command(folder, entry):
    """Run the tool through spin_launch.py, which uses the Python set in the tool's settings.json."""
    python = Path(sys.executable)
    if python.name.casefold() == 'python.exe' and python.with_name('pythonw.exe').is_file():
        python = python.with_name('pythonw.exe')
    return [str(python), str(SUITE / 'app' / 'spin_launch.py'), str(SUITE / folder), entry]


def start_tool(folder, entry):
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == 'nt' else 0
    return subprocess.Popen(launch_command(folder, entry), cwd=SUITE / folder, stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
                            start_new_session=os.name != 'nt')


class Launcher(QWidget):
    def __init__(self):
        super().__init__()
        import spin_ui
        self.setWindowTitle('Spin Tools')
        self.settings = QSettings('Spin256', 'Spin Tools/Launcher')
        layout = QVBoxLayout(self)
        title = QLabel('Spin Tools')
        title.setStyleSheet('font-size: 18pt; font-weight: bold;')
        layout.addWidget(title)
        intro = QLabel(NOTE)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        self.buttons = {}
        for row, (name, folder, entry, text) in enumerate(TOOLS):
            button = QPushButton(name)
            button.setMinimumWidth(190)
            button.setMinimumHeight(36)
            button.setToolTip(f'Open {name} ({folder}\\{entry})')
            button.clicked.connect(lambda checked=False, f=folder, e=entry, n=name: self.open_tool(n, f, e))
            description = QLabel(text)
            description.setWordWrap(True)
            description.setMinimumWidth(420)
            description.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            grid.addWidget(button, row, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(description, row, 1)
            self.buttons[name] = button
        layout.addLayout(grid)
        row = QHBoxLayout()
        self.usage = QLabel()
        self.clear_button = QPushButton('Clear temporary files…')
        self.clear_button.setToolTip("Delete captured scenes, results, frames and logs in every tool's temp folder.")
        self.clear_button.clicked.connect(self.clear)
        row.addWidget(self.usage, 1)
        row.addWidget(self.clear_button)
        layout.addLayout(row)
        self.refresh_usage()
        self.status = QLabel('')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch()
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        layout.addWidget(spin_ui.footer(self, VERSION))
        # Room for every description; a remembered size never makes it smaller than its contents need.
        self.setMinimumSize(max(760, self.sizeHint().width()), max(540, self.sizeHint().height()))
        self.resize(self.minimumSize())
        if self.settings.value('geometry'):
            self.restoreGeometry(self.settings.value('geometry'))

    def open_tool(self, name, folder, entry):
        try:
            start_tool(folder, entry)
        except OSError as error:
            QMessageBox.critical(self, 'Spin Tools', f'Could not start {name}: {error}')
            return
        self.status.setText(f'Starting {name}…')

    def refresh_usage(self):
        self.usage.setText('Temporary files: ' + readable(temp_usage()))

    def clear(self):
        running = running_tools()
        if running:
            QMessageBox.information(self, 'Spin Tools', 'Close these tools first, then clear the temporary files:\n\n'
                                    + '\n'.join(running))
            return
        answer = QMessageBox.question(
            self, 'Clear temporary files',
            f'Delete {readable(temp_usage())} of temporary files from every Spin tool?\n\nThis removes captured '
            'scenes, results, animation frames and logs. A result sent to DAZ but not yet imported there is lost, '
            'so import it first. Settings, presets and the installation are kept.')
        if answer != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            freed, skipped = clear_temp()
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_usage()
        self.status.setText(f'Cleared {readable(freed)}.' + (f' {skipped} item(s) in use were skipped.' if skipped else ''))

    def closeEvent(self, event):
        self.settings.setValue('geometry', self.saveGeometry())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv[:1])
    import spin_icon
    spin_icon.set_app_icon(app, 'launcher')
    window = Launcher()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
